import logging

from auditrl.llm import ExtractedClaim, LLMClient
from auditrl.models import Claim, Finding, Verdict
from auditrl.retrieval import EvidenceIndex
from auditrl.verify.arithmetic import verify_arithmetic
from auditrl.verify.citation import verify_citation
from auditrl.verify.figure import verify_figure
from auditrl.verify.types import CheckResult, uncertain

log = logging.getLogger(__name__)


def check_claim(
    claim: ExtractedClaim,
    index: EvidenceIndex,
    llm: LLMClient,
    *,
    rel_tol: float = 1e-6,
    top_k: int = 5,
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    figures = claim.figures
    if claim.cited_source:
        checks += verify_citation(
            claim.text, claim.cited_source, figures, index, llm, rel_tol=rel_tol, top_k=top_k
        )
    else:
        for fig in figures:
            if fig.role == "result":
                continue
            checks.append(verify_figure(claim.text, fig, index, llm, rel_tol=rel_tol, top_k=top_k))
    if claim.claim_type == "arithmetic":
        checks.append(
            verify_arithmetic(claim.text, figures, index, llm, rel_tol=rel_tol, top_k=top_k)
        )
    if not checks:
        checks.append(uncertain("figure", "The claim contains no figure that could be checked."))
    return checks


def merge_verdict(checks: list[CheckResult]) -> tuple[Verdict, float]:
    decisive = [c for c in checks if c.decisive or c.verdict == Verdict.invalid]
    invalid = [c for c in decisive if c.verdict == Verdict.invalid]
    if invalid:
        return Verdict.invalid, max(c.confidence for c in invalid)
    if decisive and all(c.verdict == Verdict.valid for c in decisive):
        return Verdict.valid, min(c.confidence for c in decisive)
    return Verdict.uncertain, 0.3


def apply_replacements(text: str, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        if old and old in text:
            text = text.replace(old, new, 1)
    return text


def build_finding(
    claim_row: Claim,
    checks: list[CheckResult],
    llm: LLMClient,
    *,
    llm_rationale: bool = True,
) -> Finding:
    verdict, confidence = merge_verdict(checks)
    invalid = [c for c in checks if c.verdict == Verdict.invalid]
    primary = invalid[0] if invalid else None

    facts: list[str] = []
    for c in invalid or checks:
        facts.extend(c.facts)

    correction = None
    if invalid:
        replacements = [r for c in invalid for r in c.replacements]
        fixed = apply_replacements(claim_row.text, replacements)
        correction = fixed if fixed != claim_row.text else None

    # Invalid checks' evidence first, then supporting evidence (e.g. sourced operands of a
    # recomputed figure) so every finding links back to the sources it relies on.
    evidence_ids: list[int] = []
    quotes: list[str] = []
    ordered = invalid + [c for c in checks if c.verdict == Verdict.valid]
    for c in ordered or checks:
        for e in c.evidence:
            if e.passage_id not in evidence_ids:
                evidence_ids.append(e.passage_id)
                quotes.append(e.quote)

    rationale = " ".join(facts)
    if llm_rationale and facts:
        try:
            rationale = llm.write_rationale(claim_row.text, verdict.value, facts) or rationale
        except Exception:
            log.exception("rationale generation failed; using verifier facts")

    return Finding(
        audit_id=claim_row.audit_id,
        claim_id=claim_row.id,
        verdict=verdict,
        error_type=primary.error_type if primary else None,
        evidence_passage_ids=evidence_ids,
        evidence_quotes=quotes,
        rationale=rationale,
        correction=correction,
        confidence=round(confidence, 3),
        verifier="+".join(dict.fromkeys(c.kind for c in checks)),
        checks=[c.to_dict() for c in checks],
    )
