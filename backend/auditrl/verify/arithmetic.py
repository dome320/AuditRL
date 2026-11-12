import re

from auditrl.llm import ExtractedFigure, LLMClient
from auditrl.models import ErrorType, Verdict
from auditrl.retrieval import EvidenceIndex
from auditrl.verify.figure import contains, parse_evidence_value
from auditrl.verify.normalize import Quantity, format_like, parse_quantity, value_matches_interval
from auditrl.verify.safe_eval import UnsafeExpression, evaluate_interval
from auditrl.verify.types import CheckResult, Evidence, uncertain

_DOWN = re.compile(
    r"\b(declin\w*|decreas\w*|fell|fall\w*|down|drop\w*|contract\w*|shr[ai]nk\w*)\b", re.I
)
_UP = re.compile(r"\b(gr[eo]w\w*|increas\w*|rose|rise\w*|up|jump\w*|climb\w*|expand\w*)\b", re.I)


def _fmt(value: float, template: Quantity) -> str:
    return format_like(template, value)


def verify_arithmetic(
    claim_text: str,
    figures: list[ExtractedFigure],
    index: EvidenceIndex,
    llm: LLMClient,
    *,
    rel_tol: float = 1e-6,
    top_k: int = 5,
) -> CheckResult:
    kind = "arithmetic"
    query = " ".join([claim_text, *(f.metric for f in figures)])
    passages = index.search(query, k=top_k)
    proposal = llm.propose_computation(claim_text, figures, index.candidates(passages))
    if proposal is None:
        return uncertain(kind, "Could not determine how the stated value was computed.")

    intervals: dict[str, tuple[float, float]] = {}
    evidence: list[Evidence] = []
    operand_desc: list[str] = []
    for op in proposal.operands:
        if op.source == "claim":
            if not contains(claim_text, op.value_text):
                return uncertain(kind, f"Operand '{op.value_text}' is not in the claim.")
            q = parse_quantity(op.value_text)
        else:
            passage = index.by_id.get(op.passage_id) if op.passage_id is not None else None
            if passage is None or not contains(passage.text, op.value_text):
                return uncertain(kind, f"Operand '{op.value_text}' is not in the cited evidence.")
            q = parse_evidence_value(op.value_text, passage)
            evidence.append(Evidence(passage.id, passage.text))
        if q is None:
            return uncertain(kind, f"Could not parse operand '{op.value_text}'.")
        intervals[op.name] = q.interval
        source = "report" if op.source == "claim" else index.doc_name(op.passage_id)
        operand_desc.append(f"{op.name}={op.value_text} ({source})")

    if not contains(claim_text, proposal.result_value_text):
        return uncertain(kind, f"Stated result '{proposal.result_value_text}' is not in the claim.")
    stated = parse_quantity(proposal.result_value_text)
    if stated is None:
        return uncertain(kind, f"Could not parse '{proposal.result_value_text}'.")

    try:
        lo, hi, point = evaluate_interval(proposal.expression, intervals)
    except (UnsafeExpression, SyntaxError, KeyError, ZeroDivisionError, OverflowError) as exc:
        return uncertain(kind, f"Could not evaluate the proposed computation ({exc}).")

    details = {
        "operation": proposal.operation,
        "expression": proposal.expression,
        "operands": [o.model_dump() for o in proposal.operands],
        "computed": point,
        "computed_interval": [lo, hi],
        "stated": stated.value,
    }
    computed_desc = (
        f"Recomputed {proposal.operation} as {proposal.expression} with "
        f"{', '.join(operand_desc)} = {_fmt(point, stated)}"
    )

    point_for_fix = point
    matched = value_matches_interval(stated, (lo, hi), rel_tol)
    if not matched and stated.value > 0 > point and _DOWN.search(claim_text):
        # "declined 2.8%" states the magnitude of a negative change.
        matched = value_matches_interval(stated, (-hi, -lo), rel_tol)
        point_for_fix = -point

    if matched:
        return CheckResult(
            kind,
            Verdict.valid,
            facts=[f"{computed_desc}, consistent with the stated {proposal.result_value_text}."],
            evidence=evidence,
            confidence=0.95,
            details=details,
        )

    direction_conflict = (
        point < 0 and _UP.search(claim_text) and not _DOWN.search(claim_text)
    ) or (point > 0 and _DOWN.search(claim_text) and not _UP.search(claim_text))
    facts = [f"{computed_desc}, not {proposal.result_value_text}."]
    if direction_conflict:
        facts.append("The figures imply a change in the opposite direction to the one stated.")
    return CheckResult(
        kind,
        Verdict.invalid,
        error_type=ErrorType.arithmetic_error,
        facts=facts,
        evidence=evidence,
        replacements=[(proposal.result_value_text, _fmt(point_for_fix, stated))],
        confidence=0.9,
        details=details,
    )
