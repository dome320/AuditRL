import json
from pathlib import Path

from sqlmodel import Session, select

from auditrl.config import get_settings
from auditrl.models import AuditRun, Claim, Finding, Review, ReviewAction, Verdict


class ReviewError(ValueError):
    pass


def submit_review(
    session: Session,
    finding_id: int,
    action: ReviewAction,
    *,
    edited_verdict: Verdict | None = None,
    edited_correction: str | None = None,
    notes: str | None = None,
    reviewer: str | None = None,
    feedback_log: Path | None = None,
) -> Review:
    finding = session.get(Finding, finding_id)
    if finding is None:
        raise ReviewError(f"finding {finding_id} not found")
    if action != ReviewAction.edit and (edited_verdict or edited_correction):
        raise ReviewError("edited_verdict/edited_correction require action='edit'")
    review = Review(
        finding_id=finding_id,
        action=action,
        edited_verdict=edited_verdict,
        edited_correction=edited_correction,
        notes=notes,
        reviewer=reviewer or get_settings().reviewer,
    )
    session.add(review)
    session.commit()
    session.refresh(review)
    _log_feedback(session, finding, review, feedback_log)
    return review


def latest_reviews(session: Session, finding_ids: list[int]) -> dict[int, Review]:
    if not finding_ids:
        return {}
    rows = session.exec(
        select(Review)
        .where(Review.finding_id.in_(finding_ids))  # type: ignore[union-attr]
        .order_by(Review.created_at, Review.id)
    ).all()
    return {r.finding_id: r for r in rows}


def _log_feedback(session: Session, finding: Finding, review: Review, path: Path | None) -> None:
    """Append-only feedback log: each line is a candidate training/eval example (spec T05)."""
    claim = session.get(Claim, finding.claim_id)
    run = session.get(AuditRun, finding.audit_id)
    path = path or get_settings().data_dir / "feedback.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "audit_id": finding.audit_id,
        "model": run.model if run else None,
        "prompt_version": run.prompt_version if run else None,
        "claim": {
            "id": claim.id,
            "text": claim.text,
            "claim_type": claim.claim_type,
            "span": [claim.span_start, claim.span_end],
            "cited_source": claim.cited_source,
        }
        if claim
        else None,
        "finding": {
            "id": finding.id,
            "verdict": finding.verdict,
            "error_type": finding.error_type,
            "correction": finding.correction,
            "evidence_passage_ids": finding.evidence_passage_ids,
            "evidence_quotes": finding.evidence_quotes,
            "rationale": finding.rationale,
            "verifier": finding.verifier,
        },
        "review": review.model_dump(mode="json"),
    }
    with path.open("a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
