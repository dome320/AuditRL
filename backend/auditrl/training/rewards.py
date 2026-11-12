"""Rewards for judge outputs (spec T02), built on the Phase 1 normalizer and grounding checks.

Correct, evidence-supported findings earn reward; false accusations cost the most; corrections
whose figures the evidence does not support are penalized; abstaining when the evidence is
insufficient earns a small reward, while abstaining when it is sufficient costs a little.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import ValidationError

from auditrl.judge.format import EvidenceItem, JudgeOutput, parse_output
from auditrl.judge.grounding import unsupported_figures
from auditrl.verify.normalize import find_quantities, values_match

# (target verdict, predicted verdict) -> base reward
BASE = {
    ("invalid", "invalid"): 1.0,
    ("invalid", "valid"): -1.0,
    ("invalid", "uncertain"): -0.25,
    ("valid", "valid"): 1.0,
    ("valid", "invalid"): -1.5,
    ("valid", "uncertain"): -0.25,
    ("uncertain", "uncertain"): 0.5,
    ("uncertain", "valid"): -0.5,
    ("uncertain", "invalid"): -1.0,
}
PARSE_FAILURE = -1.0


@dataclass
class RewardBreakdown:
    total: float
    parsed: bool
    predicted: str | None = None
    components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def correction_fixes(claim: str, target_correction: str | None, predicted: str | None) -> bool:
    """The prediction contains every figure the gold correction introduces."""
    if not target_correction or not predicted:
        return False
    in_claim = {q.raw for q in find_quantities(claim)}
    needed = [q for q in find_quantities(target_correction) if q.raw not in in_claim]
    got = find_quantities(predicted)
    if not needed:  # non-numeric fix, e.g. a citation: compare text loosely
        return target_correction.strip().lower() == predicted.strip().lower()
    return all(any(values_match(n, g) and n.unit == g.unit for g in got) for n in needed)


def score(
    completion: str, target: dict[str, Any], claim: str, evidence: list[EvidenceItem]
) -> RewardBreakdown:
    try:
        pred: JudgeOutput = parse_output(completion)
    except (ValueError, ValidationError):
        return RewardBreakdown(PARSE_FAILURE, parsed=False, components={"parse": PARSE_FAILURE})
    t_verdict = target["verdict"]
    parts: dict[str, float] = {"verdict": BASE[(t_verdict, pred.verdict)]}
    known = {e.eid for e in evidence}
    cited = [eid for eid in pred.evidence if eid in known]
    if pred.verdict in ("valid", "invalid"):
        if not cited:
            parts["no_evidence"] = -0.25
        elif set(cited) & set(target.get("evidence") or []):
            parts["evidence"] = 0.1
    if pred.verdict == "invalid" and t_verdict == "invalid":
        if pred.error_type == target.get("error_type"):
            parts["error_type"] = 0.25
        if correction_fixes(claim, target.get("correction"), pred.correction):
            parts["correction"] = 0.25
    if pred.verdict == "invalid" and pred.correction:
        items = [e for e in evidence if e.eid in cited] or evidence
        if unsupported_figures(claim, pred.correction, items):
            parts["unsupported_correction"] = -0.5
    return RewardBreakdown(round(sum(parts.values()), 4), True, pred.verdict, parts)


def _content(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return completion[-1].get("content") or ""
    return str(completion)


def judge_reward(
    prompts: list[Any],
    completions: list[Any],
    claim: list[str],
    evidence: list[list[dict[str, Any]]],
    target: list[dict[str, Any]],
    **_: Any,
) -> list[float]:
    """TRL GRPO reward function; dataset columns arrive as keyword lists."""
    return [
        score(_content(c), t, cl, [EvidenceItem(**e) for e in ev]).total
        for c, cl, ev, t in zip(completions, claim, evidence, target, strict=True)
    ]


def summarize(rows: list[dict[str, Any]], completions: list[str]) -> dict[str, Any]:
    """Offline judge metrics on a labeled split (val) for quick model comparison."""
    scored = [
        score(c, r["target"], r["claim"], [EvidenceItem(**e) for e in r["evidence"]])
        for r, c in zip(rows, completions, strict=True)
    ]
    n = len(scored) or 1
    tp = sum(
        1
        for r, s in zip(rows, scored, strict=True)
        if r["target"]["verdict"] == "invalid" and s.predicted == "invalid"
    )
    pred_inv = sum(1 for s in scored if s.predicted == "invalid")
    gold_inv = sum(1 for r in rows if r["target"]["verdict"] == "invalid")
    precision = tp / pred_inv if pred_inv else 0.0
    recall = tp / gold_inv if gold_inv else 0.0
    gold_valid = [s for r, s in zip(rows, scored, strict=True) if r["target"]["verdict"] == "valid"]
    return {
        "examples": len(rows),
        "parse_rate": sum(1 for s in scored if s.parsed) / n,
        "verdict_accuracy": sum(
            1 for r, s in zip(rows, scored, strict=True) if s.predicted == r["target"]["verdict"]
        )
        / n,
        "invalid_precision": precision,
        "invalid_recall": recall,
        "invalid_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_accusation_rate": (
            sum(1 for s in gold_valid if s.predicted == "invalid") / len(gold_valid)
            if gold_valid
            else 0.0
        ),
        "correction_rate": (
            sum(1 for s in scored if "correction" in s.components) / tp if tp else 0.0
        ),
        "unsupported_correction_rate": (
            sum(1 for s in scored if "unsupported_correction" in s.components) / pred_inv
            if pred_inv
            else 0.0
        ),
        "mean_reward": sum(s.total for s in scored) / n,
    }
