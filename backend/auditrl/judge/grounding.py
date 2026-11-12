"""Deterministic checks on judge output: cited evidence must exist and corrections must be
supported by evidence values. Shared by live audits (as a guard) and RL rewards (as a penalty).
"""

from itertools import permutations

from auditrl.judge.format import EvidenceItem, JudgeOutput
from auditrl.verify.normalize import (
    Quantity,
    find_quantities,
    parse_quantity,
    value_matches_interval,
    values_match,
)


def evidence_values(evidence: list[EvidenceItem]) -> list[Quantity]:
    values: list[Quantity] = []
    for e in evidence:
        if e.cells:
            for cell in e.cells:
                q = parse_quantity(cell, default_scale=e.unit_scale)
                if q is not None:
                    values.append(q)
        else:
            values.extend(find_quantities(e.text))
    return values


def supported(q: Quantity, values: list[Quantity], rel_tol: float = 1e-6) -> bool:
    """A figure is supported if it matches an evidence value, or (for percentages) a ratio or
    growth rate computed from two evidence values. Declines may be stated as magnitudes
    ("fell 4.2%"), so growth matches in absolute value."""
    if q.unit == "percent":
        money = [v.value for v in values if v.unit in ("currency", "number") and v.value]
        for a, b in permutations(money, 2):
            growth = (a - b) / abs(b) * 100
            for derived in (a / b * 100, growth, abs(growth)):
                if value_matches_interval(q, (derived, derived), rel_tol):
                    return True
        return any(v.unit == "percent" and values_match(q, v, rel_tol) for v in values)
    return any(values_match(q, v, rel_tol) for v in values)


def unsupported_figures(
    claim: str, correction: str, evidence: list[EvidenceItem], rel_tol: float = 1e-6
) -> list[str]:
    """Figures introduced by the correction (not already in the claim) that evidence lacks."""
    in_claim = {q.raw for q in find_quantities(claim)}
    values = evidence_values(evidence)
    return [
        q.raw
        for q in find_quantities(correction)
        if q.raw not in in_claim and not supported(q, values, rel_tol)
    ]


def ground(
    out: JudgeOutput, claim: str, evidence: list[EvidenceItem], rel_tol: float = 1e-6
) -> tuple[JudgeOutput, list[str]]:
    """Return the output to act on plus notes on anything that was downgraded."""
    notes: list[str] = []
    known = {e.eid for e in evidence}
    cited = [eid for eid in out.evidence if eid in known]
    if len(cited) != len(out.evidence):
        notes.append(f"unknown evidence ids dropped: {sorted(set(out.evidence) - known)}")
    out = out.model_copy(update={"evidence": cited})
    if out.verdict != "invalid":
        return out.normalized(), notes
    if not cited:
        notes.append("invalid verdict cites no evidence; downgraded to uncertain")
        return _uncertain(out), notes
    if out.correction:
        cited_items = [e for e in evidence if e.eid in cited] or evidence
        missing = unsupported_figures(claim, out.correction, cited_items, rel_tol)
        if missing:
            notes.append(f"correction figures not supported by cited evidence: {missing}")
            return _uncertain(out), notes
    return out, notes


def _uncertain(out: JudgeOutput) -> JudgeOutput:
    return out.model_copy(update={"verdict": "uncertain", "error_type": None, "correction": None})
