"""Build the judge SFT/RL dataset from synthetic, silver, and reviewer-feedback examples.

Only companies in the benchmark's train split feed `train.jsonl` and only dev-split companies
feed `val.jsonl`; test-split companies are never read. Feedback examples whose claim appears
in a test-split report are dropped. `manifest.json` records counts, file hashes, and a
dataset hash that training runs log for reproducibility (T04).
"""

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from auditrl.benchmark.agent import Task
from auditrl.benchmark.dataset import case_dirs, load_tasks
from auditrl.benchmark.metrics import BY_KEY, MARGINS
from auditrl.benchmark.splits import assign_split, manifest_hash
from auditrl.findings.aggregate import apply_replacements
from auditrl.intake import parse_path
from auditrl.judge.evidence import index_from_parsed, retrieve
from auditrl.judge.format import (
    JUDGE_PROMPT_VERSION,
    EvidenceItem,
    JudgeOutput,
    build_messages,
    render_target,
)
from auditrl.review.feedback import load_feedback, to_example
from auditrl.training.synthetic import Example, generate_for_task

SPLIT_TO_FILE = {"train": "train", "dev": "val"}
_CITATION = re.compile(r"\(\s*(?:source|sources)\s*:\s*([^)]+)\)", re.I)


@dataclass
class BuildConfig:
    name: str = "judge-v1"
    seed: int = 0
    per_task: int = 40
    top_k: int = 8
    include_silver: bool = True
    include_feedback: bool = True


def _split_of(bench: Path, ticker: str) -> str:
    path = bench / "splits.json"
    seed = json.loads(path.read_text())["seed"] if path.exists() else "auditrl-v1"
    return assign_split(ticker, seed)


def _record(ex: Example, source: str, split: str, rid: str) -> dict[str, Any]:
    return {
        "id": rid,
        "source": source,
        "split": split,
        "kind": ex.kind,
        "prompt": build_messages(ex.claim, ex.cited_source, ex.evidence),
        "completion": [{"role": "assistant", "content": render_target(ex.target)}],
        "claim": ex.claim,
        "cited_source": ex.cited_source,
        "evidence": [e.to_dict() for e in ex.evidence],
        "target": ex.target.normalized().model_dump(),
        "meta": ex.meta,
    }


# --- silver: benchmark cases labeled by the deterministic labeler ---------------


def _silver_examples(bench: Path, case_dir: Path, top_k: int) -> list[Example]:
    expected = json.loads((case_dir / "expected.json").read_text())
    docs = [parse_path(p) for p in sorted((case_dir / "sources").iterdir())]
    index = index_from_parsed(docs)
    out = []
    for lab in expected.get("detail", []):
        if lab["verdict"] not in ("valid", "invalid"):
            continue
        claim = lab["quote"]
        cited = _CITATION.search(claim)
        evidence = retrieve(index, claim, k=top_k)
        eids = _silver_evidence(lab, evidence)
        reasons = "; ".join(f["reason"] for f in lab["figures"] if f["status"] != "unlabeled")
        if lab["verdict"] == "valid":
            target = JudgeOutput(
                verdict="valid", evidence=eids, rationale=f"Matches the filings: {reasons}."
            )
        else:
            fixes = [
                (f["value_text"], f["correction"])
                for f in lab["figures"]
                if f["status"] == "error" and f.get("correction")
            ]
            correction = apply_replacements(claim, fixes)
            target = JudgeOutput(
                verdict="invalid",
                error_type=lab["error_type"],
                evidence=eids,
                correction=correction if correction != claim else None,
                rationale=f"Contradicted by the filings: {reasons}.",
            )
        if not eids:
            target = JudgeOutput(
                verdict="uncertain", rationale="The retrieved passages do not contain the figures."
            )
        out.append(
            Example(
                claim,
                cited.group(1).strip() if cited else None,
                evidence,
                target,
                f"silver:{lab['verdict']}",
                {"case_id": case_dir.name},
            )  # fmt: skip
        )
    return out


def _silver_evidence(label: dict[str, Any], evidence: list[EvidenceItem]) -> list[str]:
    wanted: set[str] = set()
    for f in label["figures"]:
        if f["status"] == "unlabeled":
            continue
        if f.get("metric") in BY_KEY:
            wanted.add(BY_KEY[f["metric"]].label)
        if f.get("kind") == "margin":
            for name, (num, den) in MARGINS.items():
                if name in label["quote"].lower():
                    wanted |= {BY_KEY[num].label, BY_KEY[den].label}
    return [e.eid for e in evidence if e.row_label in wanted]


# --- feedback: reviewer decisions ----------------------------------------------


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _feedback_examples(log: Path, blocked_text: str) -> tuple[list[Example], int]:
    out, dropped = [], 0
    for record in load_feedback(log):
        ex = to_example(record)
        if not ex or ex["target"] is None:
            continue
        claim = ex["input"]["claim"]
        if _squash(claim) in blocked_text:
            dropped += 1
            continue
        evidence = [
            EvidenceItem(eid=f"E{i}", document="source", text=q)
            for i, q in enumerate(ex["input"]["evidence"], start=1)
        ]
        t = ex["target"]
        target = JudgeOutput(
            verdict=t["verdict"],
            error_type=t.get("error_type"),
            evidence=[e.eid for e in evidence] if t["verdict"] != "uncertain" else [],
            correction=t.get("correction"),
            rationale=t.get("rationale") or "",
        )
        out.append(
            Example(
                claim,
                ex["input"].get("cited_source"),
                evidence,
                target,
                f"feedback:{ex['kind']}",
                ex["provenance"],
            )  # fmt: skip
        )
    return out, dropped


# --- build -----------------------------------------------------------------------


def build_dataset(
    bench: Path, out_dir: Path, config: BuildConfig, feedback_log: Path | None = None
) -> dict[str, Any]:
    tasks = load_tasks(bench)
    splits = {t.ticker: _split_of(bench, t.ticker) for t in tasks}
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}

    for task in tasks:
        split = SPLIT_TO_FILE.get(splits[task.ticker])
        if split is None:
            continue
        for i, ex in enumerate(
            generate_for_task(bench, task, config.per_task, config.seed, config.top_k)
        ):
            rows[split].append(_record(ex, "synthetic", split, f"syn-{task.task_id}-{i:03d}"))

    test_text = ""
    for case_dir in case_dirs(bench):
        if not (case_dir / "task.json").exists():
            continue
        task = Task.from_dict(json.loads((case_dir / "task.json").read_text()))
        split = SPLIT_TO_FILE.get(_split_of(bench, task.ticker))
        if split is None:
            test_text += " " + _squash((case_dir / "report.md").read_text())
            continue
        if config.include_silver and (case_dir / "expected.json").exists():
            for i, ex in enumerate(_silver_examples(bench, case_dir, config.top_k)):
                rows[split].append(_record(ex, "silver", split, f"silver-{case_dir.name}-{i:03d}"))

    dropped = 0
    if config.include_feedback and feedback_log is not None:
        examples, dropped = _feedback_examples(feedback_log, test_text)
        for i, ex in enumerate(examples):
            rows["train"].append(_record(ex, "feedback", "train", f"fb-{i:04d}"))

    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, items in rows.items():
        path = out_dir / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items))
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    excluded = sorted({t for t, s in splits.items() if s == "test"})
    used = {r["meta"].get("ticker") for items in rows.values() for r in items}
    assert not used & set(excluded), "test-split company leaked into training data"

    manifest = {
        "name": config.name,
        "created_at": datetime.now(UTC).isoformat(),
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "config": config.__dict__,
        "bench_manifest_hash": manifest_hash(bench) if (bench / "splits.json").exists() else None,
        "tickers": {
            "train": sorted(t for t, s in splits.items() if s == "train"),
            "val": sorted(t for t, s in splits.items() if s == "dev"),
            "excluded_test": excluded,
        },
        "feedback_dropped_test_overlap": dropped,
        "counts": {split: _counts(items) for split, items in rows.items()},
        "files": files,
        "dataset_hash": hashlib.sha256("".join(sorted(files.values())).encode()).hexdigest()[:16],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def _counts(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(items),
        "source": dict(Counter(r["source"] for r in items)),
        "verdict": dict(Counter(r["target"]["verdict"] for r in items)),
        "error_type": dict(
            Counter(r["target"]["error_type"] for r in items if r["target"]["error_type"])
        ),
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
