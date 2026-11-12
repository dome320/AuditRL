import logging
import time
from datetime import UTC, datetime

from sqlalchemy import Engine
from sqlmodel import Session, select

from auditrl.config import Settings, get_settings
from auditrl.extraction import extract_claims
from auditrl.findings import build_finding, check_claim
from auditrl.intake import ParsedDocument, store_document
from auditrl.llm import PROMPT_VERSION, LLMClient, get_client
from auditrl.models import (
    AuditRun,
    AuditStatus,
    Claim,
    ClaimType,
    Document,
    DocumentKind,
    Passage,
    get_engine,
)
from auditrl.retrieval import EvidenceIndex

log = logging.getLogger(__name__)


def create_audit(
    session: Session,
    report: ParsedDocument,
    sources: list[ParsedDocument],
    *,
    model: str | None = None,
    origins: dict[str, str] | None = None,
    settings: Settings | None = None,
) -> AuditRun:
    settings = settings or get_settings()
    origins = origins or {}
    run = AuditRun(
        model=model or settings.model,
        prompt_version=PROMPT_VERSION,
        config_hash=settings.config_hash(),
    )
    session.add(run)
    session.flush()
    store_document(session, run.id, report, DocumentKind.report, origins.get(report.name, "upload"))
    for src in sources:
        store_document(session, run.id, src, DocumentKind.source, origins.get(src.name, "upload"))
    session.commit()
    session.refresh(run)
    return run


def run_audit(
    audit_id: int,
    *,
    llm: LLMClient | None = None,
    engine: Engine | None = None,
    settings: Settings | None = None,
) -> AuditRun:
    settings = settings or get_settings()
    engine = engine or get_engine()
    with Session(engine) as session:
        run = session.get(AuditRun, audit_id)
        if run is None:
            raise ValueError(f"audit {audit_id} not found")
        llm = llm or get_client(run.model, settings)
        run.status = AuditStatus.running
        session.add(run)
        session.commit()
        started = time.perf_counter()
        try:
            _execute(session, run, llm, settings)
            run.status = AuditStatus.completed
        except Exception as exc:
            log.exception("audit %s failed", audit_id)
            session.rollback()
            run = session.get(AuditRun, audit_id)
            run.status = AuditStatus.failed
            run.error = f"{type(exc).__name__}: {exc}"
        run.latency_s = round(time.perf_counter() - started, 3)
        run.completed_at = datetime.now(UTC)
        run.prompt_tokens = llm.usage.prompt_tokens
        run.completion_tokens = llm.usage.completion_tokens
        run.cost_usd = round(llm.usage.cost_usd, 6)
        session.add(run)
        session.commit()
        session.refresh(run)
        return run


def _execute(session: Session, run: AuditRun, llm: LLMClient, settings: Settings) -> None:
    docs = session.exec(select(Document).where(Document.audit_id == run.id)).all()
    report = next(d for d in docs if d.kind == DocumentKind.report)
    sources = [d for d in docs if d.kind == DocumentKind.source]
    passages = session.exec(
        select(Passage).where(Passage.document_id.in_([d.id for d in sources]))  # type: ignore[union-attr]
    ).all()
    index = EvidenceIndex(sources, passages)

    aligned = extract_claims(report.text, llm, settings.extraction_chunk_chars)
    for i, a in enumerate(aligned):
        c = a.claim
        claim_row = Claim(
            audit_id=run.id,
            document_id=report.id,
            ordinal=i,
            span_start=a.start,
            span_end=a.end,
            text=report.text[a.start : a.end],
            claim_type=ClaimType(c.claim_type),
            checkable=c.checkable,
            cited_source=c.cited_source,
            figures=[f.model_dump() for f in c.figures],
            alignment_score=a.score,
        )
        session.add(claim_row)
        session.flush()
        if not c.checkable:
            continue
        checks = check_claim(
            c.model_copy(update={"text": claim_row.text}),
            index,
            llm,
            rel_tol=settings.relative_tolerance,
            top_k=settings.retrieval_top_k,
        )
        session.add(build_finding(claim_row, checks, llm, llm_rationale=settings.llm_rationale))
    session.commit()
