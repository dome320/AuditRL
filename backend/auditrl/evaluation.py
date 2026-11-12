"""Phase 1 smoke evaluation against the curated dataset (`data/curated/<case>/`).

Each case directory holds `report.md`, `sources/`, and `expected.json` with claim-level labels.
Every injected error is labeled, so an `invalid` finding with no matching label counts as a
false positive.
"""

import json
import re
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine

from auditrl.config import Settings, get_settings
from auditrl.export import build_export
from auditrl.intake import parse_path
from auditrl.llm import LLMClient, get_client
from auditrl.models import init_db
from auditrl.pipeline import create_audit, run_audit

SOURCE_SUFFIXES = {".md", ".txt", ".pdf", ".html", ".htm", ".csv", ".xlsx"}


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _matches(label_quote: str, claim_text: str) -> bool:
    a, b = _squash(label_quote), _squash(claim_text)
    return a in b or b in a or fuzz.partial_ratio(a, b) >= 92


def run_case(case_dir: Path, llm: LLMClient, settings: Settings) -> dict[str, Any]:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    init_db(engine)
    sources = sorted(p for p in (case_dir / "sources").iterdir() if p.suffix in SOURCE_SUFFIXES)
    with Session(engine) as session:
        run = create_audit(
            session,
            parse_path(case_dir / "report.md"),
            [parse_path(p) for p in sources],
            model=llm.name,
            settings=settings,
        )
    run = run_audit(run.id, llm=llm, engine=engine, settings=settings)
    with Session(engine) as session:
        export = build_export(session, run.id)
    return export


def score_case(export: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    findings = export["findings"]
    labels = expected["labels"]
    rows = []
    matched_findings: set[int] = set()
    for label in labels:
        f = next((f for f in findings if _matches(label["quote"], f["claim_text"])), None)
        if f:
            matched_findings.add(f["finding_id"])
        predicted = f["verdict"] if f else "missing"
        row = {
            "quote": label["quote"],
            "expected": label["verdict"],
            "expected_error_type": label.get("error_type"),
            "predicted": predicted,
            "predicted_error_type": f["error_type"] if f else None,
            "correction": f["correction"] if f else None,
            "has_evidence": bool(f and f["evidence"]),
        }
        if label.get("correction_contains"):
            row["correction_ok"] = bool(
                f and f["correction"] and label["correction_contains"] in f["correction"]
            )
        rows.append(row)
    unlabeled_invalid = [
        f["claim_text"]
        for f in findings
        if f["verdict"] == "invalid" and f["finding_id"] not in matched_findings
    ]
    return {
        "rows": rows,
        "unlabeled_invalid": unlabeled_invalid,
        "latency_s": export["audit"].get("latency_s"),
        "cost_usd": export["audit"].get("cost_usd") or 0.0,
    }


def aggregate(cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = [r for c in cases.values() for r in c["rows"]]
    tp = sum(1 for r in rows if r["expected"] == "invalid" and r["predicted"] == "invalid")
    fn = sum(1 for r in rows if r["expected"] == "invalid" and r["predicted"] != "invalid")
    fp_labeled = sum(1 for r in rows if r["expected"] == "valid" and r["predicted"] == "invalid")
    fp = fp_labeled + sum(len(c["unlabeled_invalid"]) for c in cases.values())
    valid_rows = [r for r in rows if r["expected"] == "valid"]
    tps = [r for r in rows if r["expected"] == "invalid" and r["predicted"] == "invalid"]
    corrections = [r for r in tps if "correction_ok" in r]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    n_cases = len(cases) or 1
    return {
        "cases": len(cases),
        "labeled_claims": len(rows),
        "errors": tp + fn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "error_type_accuracy": (
            sum(1 for r in tps if r["predicted_error_type"] == r["expected_error_type"]) / len(tps)
            if tps
            else 0.0
        ),
        "correction_accuracy": (
            sum(1 for r in corrections if r["correction_ok"]) / len(corrections)
            if corrections
            else 0.0
        ),
        "valid_claim_accuracy": (
            sum(1 for r in valid_rows if r["predicted"] == "valid") / len(valid_rows)
            if valid_rows
            else 0.0
        ),
        "false_positive_rate": fp_labeled / len(valid_rows) if valid_rows else 0.0,
        "uncertain_or_missing_rate": (
            sum(1 for r in rows if r["predicted"] in ("uncertain", "missing")) / len(rows)
            if rows
            else 0.0
        ),
        "evidence_support": (sum(1 for r in tps if r["has_evidence"]) / len(tps) if tps else 0.0),
        "mean_latency_s": sum(c["latency_s"] or 0 for c in cases.values()) / n_cases,
        "total_cost_usd": sum(c["cost_usd"] for c in cases.values()),
    }


def evaluate_dataset(
    dataset: Path, model: str | None = None, llm: LLMClient | None = None
) -> dict[str, Any]:
    settings = get_settings()
    cases: dict[str, dict[str, Any]] = {}
    for case_dir in sorted(p for p in dataset.iterdir() if (p / "expected.json").exists()):
        client = llm or get_client(model, settings)
        expected = json.loads((case_dir / "expected.json").read_text())
        export = run_case(case_dir, client, settings)
        cases[case_dir.name] = score_case(export, expected)
    return {
        "model": model or (llm.name if llm else settings.model),
        "summary": aggregate(cases),
        "cases": cases,
    }


def format_summary(result: dict[str, Any]) -> str:
    s = result["summary"]
    lines = [
        f"Model: {result['model']}",
        f"Cases: {s['cases']}  labeled claims: {s['labeled_claims']}  "
        f"injected errors: {s['errors']}",
        f"Error detection  P={s['precision']:.2f}  R={s['recall']:.2f}  F1={s['f1']:.2f}  "
        f"(tp={s['tp']} fp={s['fp']} fn={s['fn']})",
        f"Error-type accuracy {s['error_type_accuracy']:.2f}   "
        f"correction accuracy {s['correction_accuracy']:.2f}   "
        f"evidence support {s['evidence_support']:.2f}",
        f"Valid-claim accuracy {s['valid_claim_accuracy']:.2f}   "
        f"false-positive rate {s['false_positive_rate']:.2f}   "
        f"uncertain/missing {s['uncertain_or_missing_rate']:.2f}",
        f"Mean latency {s['mean_latency_s']:.2f}s   total cost ${s['total_cost_usd']:.4f}",
        "",
    ]
    for name, case in result["cases"].items():
        misses = [r for r in case["rows"] if r["predicted"] != r["expected"]]
        if not misses and not case["unlabeled_invalid"]:
            continue
        lines.append(f"[{name}]")
        for r in misses:
            lines.append(f"  expected {r['expected']:<8} got {r['predicted']:<9} {r['quote'][:80]}")
        for text in case["unlabeled_invalid"]:
            lines.append(f"  unlabeled invalid: {text[:80]}")
    return "\n".join(lines)
