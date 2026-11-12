"""RL gate: GRPO uses the deterministic verifiers as its reward, so it is only worth running
if those verifiers agree with human reviewers (plan: "attempted only if verifier agreement
with human labels is high enough"). Agreement is measured on reviewed findings that the
verifiers (not a judge model) produced.
"""

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from auditrl.review.feedback import load_feedback


def verifier_agreement(log: Path) -> dict[str, Any]:
    agree, total = 0, 0
    by_verdict: Counter[str] = Counter()
    by_verdict_agree: Counter[str] = Counter()
    for record in load_feedback(log):
        finding, review = record["finding"], record["review"]
        if finding.get("verifier") == "judge":
            continue
        verdict = finding["verdict"]
        if review["action"] == "accept":
            ok = True
        elif review["action"] == "edit":
            ok = (review.get("edited_verdict") or verdict) == verdict
        else:
            ok = False
        total += 1
        agree += ok
        by_verdict[verdict] += 1
        by_verdict_agree[verdict] += ok
    return {
        "reviews": total,
        "agreement": agree / total if total else None,
        "by_verdict": {
            v: {"reviews": n, "agreement": by_verdict_agree[v] / n} for v, n in by_verdict.items()
        },
    }


def rl_gate(log: Path, out: Path, min_reviews: int = 50, threshold: float = 0.9) -> dict[str, Any]:
    stats = verifier_agreement(log)
    passed = stats["reviews"] >= min_reviews and (stats["agreement"] or 0) >= threshold
    if stats["reviews"] < min_reviews:
        reason = f"only {stats['reviews']} verifier findings reviewed (need {min_reviews})"
    elif not passed:
        reason = f"agreement {stats['agreement']:.2f} below {threshold}"
    else:
        reason = "agreement high enough to use verifiers as RL reward"
    result = {
        **stats,
        "min_reviews": min_reviews,
        "threshold": threshold,
        "passed": passed,
        "reason": reason,
        "checked_at": datetime.now(UTC).isoformat(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    return result
