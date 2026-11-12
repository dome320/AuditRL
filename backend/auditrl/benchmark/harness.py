"""Evaluation harness (T04): run the auditor on a split and save a reproducible run record."""

import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from auditrl.benchmark.dataset import case_dirs
from auditrl.benchmark.labeler import LABELER_VERSION
from auditrl.benchmark.splits import cases_in_split, manifest_hash
from auditrl.config import Settings, get_settings
from auditrl.evaluation import evaluate_cases
from auditrl.judge.format import JUDGE_PROMPT_VERSION
from auditrl.llm import LLMClient
from auditrl.llm.prompts import PROMPT_VERSION

COMPARE_METRICS = (
    ("f1", "F1"),
    ("precision", "P"),
    ("recall", "R"),
    ("error_type_accuracy", "type acc"),
    ("correction_accuracy", "corr acc"),
    ("false_positive_rate", "FP rate"),
    ("uncertain_or_missing_rate", "unc/miss"),
    ("mean_latency_s", "lat s"),
    ("mean_cost_usd", "$/report"),
)


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def run_eval(
    bench: Path,
    model: str | None = None,
    split: str = "test",
    limit: int | None = None,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
    name: str | None = None,
) -> tuple[dict[str, Any], Path]:
    settings = settings or get_settings()
    if model:
        settings = settings.model_copy(update={"model": model})
    selected = cases_in_split(bench, split, case_dirs(bench))
    if limit:
        selected = selected[:limit]
    if not selected:
        raise ValueError(f"no frozen cases in split {split!r}")
    started = datetime.now(UTC)
    result = evaluate_cases(selected, model=settings.model, llm=llm, settings=settings)
    record = {
        "run_id": f"{started:%Y%m%dT%H%M%S}_{_slug(name or result['model'])}_{split}",
        "name": name or result["model"],
        "created_at": started.isoformat(),
        "model": result["model"],
        "split": split,
        "cases": len(selected),
        "auditor_prompt_version": PROMPT_VERSION,
        "judge_model": settings.judge_model or None,
        "judge_api_base": settings.judge_api_base or None,
        "judge_prompt_version": JUDGE_PROMPT_VERSION if settings.judge_model else None,
        "config_hash": settings.config_hash(),
        "labeler_version": LABELER_VERSION,
        "manifest_hash": manifest_hash(bench),
        "git_commit": git_commit(),
        "summary": result["summary"],
        "per_case": result["cases"],
    }
    runs = bench / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / f"{record['run_id']}.json"
    path.write_text(json.dumps(record, indent=1) + "\n")
    return record, path


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", text).strip("-")[:40]


def load_runs(bench: Path, ids: list[str] | None = None) -> list[dict[str, Any]]:
    paths = sorted((bench / "runs").glob("*.json"))
    runs = [json.loads(p.read_text()) for p in paths]
    if ids:
        runs = [r for r in runs if any(i in r["run_id"] for i in ids)]
    return runs


def compare(runs: list[dict[str, Any]]) -> str:
    """Side-by-side table; flags runs evaluated on a different manifest or labeler."""
    failed = [r for r in runs if "error" in r]
    runs = [r for r in runs if "error" not in r]
    notes = [f"{r['name']}: failed ({r['error'][:120]})" for r in failed]
    if not runs:
        return "\n".join(notes) or "no runs"
    header = ["run", "split", "n"] + [label for _, label in COMPARE_METRICS]
    rows = [header]
    for r in runs:
        s = r["summary"]
        rows.append(
            [r["name"][:32], r["split"], str(r["cases"])]
            + [
                f"{s.get(key, 0):.4f}" if key == "mean_cost_usd" else f"{s.get(key, 0):.2f}"
                for key, _ in COMPARE_METRICS
            ]
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    keys = {(r["manifest_hash"], r["labeler_version"], r["split"]) for r in runs}
    if len(keys) > 1:
        lines.append("\nwarning: runs use different manifests, labeler versions, or splits")
    return "\n".join(lines + notes)


def load_baselines(path: Path) -> list[dict[str, Any]]:
    """`baselines.yaml`: a list of {name, model, ...} entries; other keys are Settings
    overrides such as temperature, judge_model, judge_api_base."""
    data = yaml.safe_load(path.read_text()) or {}
    return list(data.get("baselines", []))


def run_baselines(
    bench: Path, config: Path, split: str = "test", limit: int | None = None
) -> list[dict[str, Any]]:
    """Run each baseline; a baseline that fails (e.g. missing API key) is reported and skipped."""
    records = []
    base = get_settings()
    for entry in load_baselines(config):
        update = {k: v for k, v in entry.items() if k != "name"}
        unknown = set(update) - set(type(base).model_fields)
        if unknown:
            records.append({"name": entry["name"], "error": f"unknown keys {sorted(unknown)}"})
            continue
        settings = base.model_copy(update=update)
        try:
            record, _ = run_eval(
                bench, split=split, limit=limit, settings=settings, name=entry["name"]
            )
        except RuntimeError as exc:
            records.append({"name": entry["name"], "error": str(exc)})
            continue
        records.append(record)
    return records
