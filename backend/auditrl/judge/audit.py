"""Use the judge model in place of the deterministic verifiers for one claim."""

from auditrl.judge.client import JudgeClient
from auditrl.judge.evidence import retrieve
from auditrl.judge.grounding import ground
from auditrl.models import Claim, ErrorType, Finding, Verdict
from auditrl.retrieval import EvidenceIndex
from auditrl.verify.types import CheckResult, Evidence


def judge_finding(
    claim_row: Claim,
    index: EvidenceIndex,
    judge: JudgeClient,
    *,
    top_k: int = 8,
    rel_tol: float = 1e-6,
) -> Finding:
    evidence = retrieve(index, claim_row.text, k=top_k)
    proposed, raw = judge.judge(claim_row.text, claim_row.cited_source, evidence)
    if proposed is None:
        out, notes = None, ["judge output could not be parsed"]
    else:
        out, notes = ground(proposed, claim_row.text, evidence, rel_tol)

    by_eid = {e.eid: e for e in evidence}
    cited = [by_eid[eid] for eid in (out.evidence if out else []) if eid in by_eid]
    verdict = Verdict(out.verdict) if out else Verdict.uncertain
    error_type = ErrorType(out.error_type) if out and out.error_type else None
    rationale = (out.rationale if out else "") or "; ".join(notes)
    if (
        notes
        and out
        and out.verdict == "uncertain"
        and proposed
        and proposed.verdict != "uncertain"
    ):
        rationale = f"{rationale} (Downgraded: {'; '.join(notes)})".strip()
    check = CheckResult(
        kind="judge",
        verdict=verdict,
        error_type=error_type,
        facts=notes,
        evidence=[Evidence(e.passage_id or 0, e.text) for e in cited],
        confidence=0.6 if verdict != Verdict.uncertain else 0.3,
        details={
            "model": judge.name,
            "raw": raw,
            "proposed": proposed.model_dump() if proposed else None,
            "retrieved": [e.passage_id for e in evidence],
        },
    )
    return Finding(
        audit_id=claim_row.audit_id,
        claim_id=claim_row.id,
        verdict=verdict,
        error_type=error_type,
        evidence_passage_ids=[e.passage_id for e in cited if e.passage_id is not None],
        evidence_quotes=[e.text for e in cited],
        rationale=rationale,
        correction=out.correction if out else None,
        confidence=check.confidence,
        verifier="judge",
        checks=[check.to_dict()],
    )
