"""Turn the reviewer feedback log into candidate training/eval examples (spec T05).

The latest review of each finding wins. Accepted findings become targets as-is; edits become
targets with the reviewer's verdict/correction, keeping the model output as a rejected
alternative (a preference pair); rejections record only the rejected output. Every example is
marked `candidate`: nothing enters training or evaluation without a separate promotion step.
"""

import json
from pathlib import Path
from typing import Any


def load_feedback(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    latest: dict[int, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            latest[record["finding"]["id"]] = record
    return list(latest.values())


def to_example(record: dict[str, Any]) -> dict[str, Any] | None:
    claim, finding, review = record.get("claim"), record["finding"], record["review"]
    if claim is None:
        return None
    model_output = {
        "verdict": finding["verdict"],
        "error_type": finding["error_type"],
        "correction": finding["correction"],
        "rationale": finding["rationale"],
    }
    action = review["action"]
    target: dict[str, Any] | None
    rejected: dict[str, Any] | None = None
    if action == "accept":
        target = model_output
    elif action == "edit":
        verdict = review.get("edited_verdict") or finding["verdict"]
        target = {
            "verdict": verdict,
            "error_type": finding["error_type"] if verdict == "invalid" else None,
            "correction": (review.get("edited_correction") or finding["correction"])
            if verdict == "invalid"
            else None,
            "rationale": review.get("notes"),
        }
        rejected = model_output
    else:
        target, rejected = None, model_output
    return {
        "id": f"fb-{record['audit_id']}-{finding['id']}",
        "status": "candidate",
        "kind": {"accept": "accepted", "edit": "edited"}.get(action, "rejected"),
        "input": {
            "claim": claim["text"],
            "claim_type": claim["claim_type"],
            "cited_source": claim["cited_source"],
            "evidence": finding["evidence_quotes"],
        },
        "target": target,
        "rejected": rejected,
        "provenance": {
            "audit_id": record["audit_id"],
            "finding_id": finding["id"],
            "model": record.get("model"),
            "prompt_version": record.get("prompt_version"),
            "reviewer": review.get("reviewer"),
            "reviewed_at": review.get("created_at"),
            "notes": review.get("notes"),
        },
    }


def export_feedback(log: Path, out: Path) -> dict[str, int]:
    examples = [e for e in map(to_example, load_feedback(log)) if e]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(e, default=str) + "\n" for e in examples))
    counts = {"total": len(examples)}
    for e in examples:
        counts[e["kind"]] = counts.get(e["kind"], 0) + 1
    return counts
