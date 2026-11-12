"""Company-grouped splits and a frozen manifest guarding the test set.

Splits are assigned by hashing the ticker, so every fiscal year of a company lands in the
same split and near-duplicate reports cannot leak across train/dev/test (spec section 06).
Freezing records a content hash per case; evaluation refuses to run on a test case whose
report, labels, or sources changed after the freeze.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SPLIT_FRACTIONS = (("train", 0.6), ("dev", 0.2), ("test", 0.2))


class IntegrityError(RuntimeError):
    pass


def assign_split(ticker: str, seed: str = "auditrl-v1") -> str:
    bucket = (
        int(hashlib.sha256(f"{seed}:{ticker.upper()}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    )
    edge = 0.0
    for name, frac in SPLIT_FRACTIONS:
        edge += frac
        if bucket <= edge:
            return name
    return SPLIT_FRACTIONS[-1][0]


def case_hash(case_dir: Path) -> str:
    h = hashlib.sha256()
    files = [case_dir / "report.md", case_dir / "expected.json"]
    files += sorted((case_dir / "sources").glob("*"))
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def _ticker(case_dir: Path) -> str:
    return json.loads((case_dir / "task.json").read_text())["ticker"]


def freeze(bench: Path, case_dirs: list[Path], seed: str = "auditrl-v1") -> dict[str, Any]:
    """Create or extend the manifest. Frozen cases keep their split and hash."""
    path = bench / "splits.json"
    manifest = json.loads(path.read_text()) if path.exists() else None
    if manifest:
        verify(bench, manifest, case_dirs)
        seed = manifest["seed"]
    else:
        manifest = {"seed": seed, "frozen_at": datetime.now(UTC).isoformat(), "cases": {}}
    for case_dir in case_dirs:
        if not (case_dir / "expected.json").exists() or case_dir.name in manifest["cases"]:
            continue
        manifest["cases"][case_dir.name] = {
            "split": assign_split(_ticker(case_dir), seed),
            "ticker": _ticker(case_dir),
            "sha256": case_hash(case_dir),
            "added_at": datetime.now(UTC).isoformat(),
        }
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def load_manifest(bench: Path) -> dict[str, Any]:
    path = bench / "splits.json"
    if not path.exists():
        raise IntegrityError("no splits.json; run `auditrl bench freeze` first")
    return json.loads(path.read_text())


def verify(bench: Path, manifest: dict[str, Any], case_dirs: list[Path]) -> None:
    by_name = {c.name: c for c in case_dirs}
    changed = [
        name
        for name, entry in manifest["cases"].items()
        if entry["split"] == "test"
        and (name not in by_name or case_hash(by_name[name]) != entry["sha256"])
    ]
    if changed:
        raise IntegrityError(f"frozen test cases changed or missing: {', '.join(sorted(changed))}")


def manifest_hash(bench: Path) -> str:
    return hashlib.sha256((bench / "splits.json").read_bytes()).hexdigest()[:16]


def cases_in_split(bench: Path, split: str, case_dirs: list[Path]) -> list[Path]:
    manifest = load_manifest(bench)
    if split == "test":
        verify(bench, manifest, case_dirs)
    names = {n for n, e in manifest["cases"].items() if split == "all" or e["split"] == split}
    return [c for c in case_dirs if c.name in names]
