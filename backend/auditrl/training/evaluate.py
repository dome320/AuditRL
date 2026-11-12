"""Offline judge evaluation on a labeled dataset split (fast check before the benchmark)."""

import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from auditrl.judge.client import JudgeClient
from auditrl.judge.format import JUDGE_PROMPT_VERSION, EvidenceItem, render_target
from auditrl.training.rewards import summarize


def eval_judge(
    rows: list[dict[str, Any]], client: JudgeClient, name: str, dataset: dict[str, Any]
) -> dict[str, Any]:
    started = time.perf_counter()
    completions = []
    for r in rows:
        out, raw = client.judge(
            r["claim"], r.get("cited_source"), [EvidenceItem(**e) for e in r["evidence"]]
        )
        completions.append(render_target(out) if out else raw)
    elapsed = time.perf_counter() - started
    return {
        "name": name,
        "created_at": datetime.now(UTC).isoformat(),
        "model": client.model,
        "api_base": client.api_base,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "dataset": dataset.get("name"),
        "dataset_hash": dataset.get("dataset_hash"),
        "metrics": {
            **summarize(rows, completions),
            "mean_latency_s": elapsed / max(len(rows), 1),
            "mean_cost_usd": client.usage.cost_usd / max(len(rows), 1),
        },
        "predictions": [
            {"id": r["id"], "kind": r["kind"], "target": r["target"]["verdict"], "output": c}
            for r, c in zip(rows, completions, strict=True)
        ],
    }


def save_eval(record: dict[str, Any], evals_dir: Path) -> Path:
    evals_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9.-]+", "-", record["name"]).strip("-")[:40]
    path = evals_dir / f"{datetime.now(UTC):%Y%m%dT%H%M%S}_{slug}.json"
    path.write_text(json.dumps(record, indent=1) + "\n")
    return path


def compare_evals(records: list[dict[str, Any]]) -> str:
    keys = [
        ("verdict_accuracy", "acc"),
        ("invalid_f1", "F1"),
        ("invalid_precision", "P"),
        ("invalid_recall", "R"),
        ("false_accusation_rate", "false acc"),
        ("correction_rate", "corr"),
        ("unsupported_correction_rate", "unsup corr"),
        ("parse_rate", "parse"),
        ("mean_reward", "reward"),
    ]
    rows = [["name", "dataset", "n"] + [label for _, label in keys]]
    for r in records:
        m = r["metrics"]
        rows.append(
            [r["name"][:30], str(r["dataset"]), str(m["examples"])]
            + [f"{m[k]:.2f}" for k, _ in keys]
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    if len({r.get("dataset_hash") for r in records}) > 1:
        lines.append("\nwarning: evaluations used different dataset builds")
    return "\n".join(lines)
