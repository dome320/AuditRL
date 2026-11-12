from collections.abc import Callable, Iterator
from typing import Annotated

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import Engine, func
from sqlmodel import Session, select

from auditrl.config import Settings, get_settings
from auditrl.export import build_export, to_html, to_json, to_jsonl, to_markdown
from auditrl.intake import UnsupportedFormat, parse_bytes, parse_text
from auditrl.llm import LLMClient, get_client
from auditrl.models import (
    AuditRun,
    Claim,
    Document,
    DocumentKind,
    Finding,
    Passage,
    ReviewAction,
    Verdict,
    get_engine,
    init_db,
)
from auditrl.pipeline import create_audit, run_audit
from auditrl.review import submit_review
from auditrl.review.service import ReviewError

LLMFactory = Callable[[str], LLMClient]


class ReviewIn(BaseModel):
    action: ReviewAction
    edited_verdict: Verdict | None = None
    edited_correction: str | None = None
    notes: str | None = None
    reviewer: str | None = None


def create_app(
    engine: Engine | None = None,
    llm_factory: LLMFactory | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    engine = init_db(engine or get_engine())
    llm_factory = llm_factory or (lambda model: get_client(model, settings))

    app = FastAPI(title="AuditRL", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def session_dep() -> Iterator[Session]:
        with Session(engine) as session:
            yield session

    SessionDep = Annotated[Session, Depends(session_dep)]

    def _run(audit_id: int, model: str) -> None:
        run_audit(audit_id, llm=llm_factory(model), engine=engine, settings=settings)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/config")
    def config() -> dict[str, object]:
        return {"default_model": settings.model}

    @app.post("/audits", status_code=202)
    async def create(
        session: SessionDep,
        background: BackgroundTasks,
        report: Annotated[UploadFile | None, File()] = None,
        report_text: Annotated[str | None, Form()] = None,
        report_name: Annotated[str, Form()] = "pasted_report.md",
        sources: Annotated[list[UploadFile] | None, File()] = None,
        model: Annotated[str | None, Form()] = None,
    ) -> dict[str, object]:
        try:
            if report is not None and report.filename:
                parsed_report = parse_bytes(report.filename, await report.read())
            elif report_text and report_text.strip():
                parsed_report = parse_text(report_name, report_text)
            else:
                raise HTTPException(422, "provide a report file or report_text")
            parsed_sources = [
                parse_bytes(s.filename or f"source_{i}", await s.read())
                for i, s in enumerate(sources or [])
                if s.filename
            ]
        except UnsupportedFormat as exc:
            raise HTTPException(415, str(exc)) from exc
        if not parsed_sources:
            raise HTTPException(422, "attach at least one source document")
        run = create_audit(
            session, parsed_report, parsed_sources, model=model or settings.model, settings=settings
        )
        background.add_task(_run, run.id, run.model)
        return {"id": run.id, "status": run.status}

    @app.get("/audits")
    def list_audits(session: SessionDep) -> list[dict[str, object]]:
        runs = session.exec(select(AuditRun).order_by(AuditRun.id.desc()).limit(100)).all()  # type: ignore[union-attr]
        names = {
            d.audit_id: d.name
            for d in session.exec(select(Document).where(Document.kind == DocumentKind.report))
        }
        return [{**r.model_dump(mode="json"), "report_name": names.get(r.id)} for r in runs]

    @app.get("/audits/{audit_id}")
    def get_audit(audit_id: int, session: SessionDep) -> dict[str, object]:
        run = session.get(AuditRun, audit_id)
        if run is None:
            raise HTTPException(404, "audit not found")
        docs = session.exec(select(Document).where(Document.audit_id == audit_id)).all()
        report = next(d for d in docs if d.kind == DocumentKind.report)
        counts = dict(
            session.exec(
                select(Passage.document_id, func.count())
                .where(Passage.document_id.in_([d.id for d in docs]))  # type: ignore[union-attr]
                .group_by(Passage.document_id)
            ).all()
        )
        claims = session.exec(
            select(Claim).where(Claim.audit_id == audit_id).order_by(Claim.ordinal)
        ).all()
        verdicts = {
            f.claim_id: (f.id, f.verdict)
            for f in session.exec(select(Finding).where(Finding.audit_id == audit_id))
        }
        return {
            "audit": run.model_dump(mode="json"),
            "report": {"id": report.id, "name": report.name, "text": report.text},
            "sources": [
                {
                    "id": d.id,
                    "name": d.name,
                    "content_type": d.content_type,
                    "passages": counts.get(d.id, 0),
                }
                for d in docs
                if d.kind == DocumentKind.source
            ],
            "claims": [
                {
                    "id": c.id,
                    "text": c.text,
                    "span": [c.span_start, c.span_end],
                    "claim_type": c.claim_type,
                    "checkable": c.checkable,
                    "finding_id": verdicts.get(c.id, (None, None))[0],
                    "verdict": verdicts.get(c.id, (None, None))[1],
                }
                for c in claims
            ],
        }

    @app.get("/audits/{audit_id}/findings")
    def get_findings(audit_id: int, session: SessionDep) -> list[dict[str, object]]:
        try:
            return build_export(session, audit_id)["findings"]
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/findings/{finding_id}/review")
    def review(finding_id: int, body: ReviewIn, session: SessionDep) -> dict[str, object]:
        try:
            r = submit_review(
                session,
                finding_id,
                body.action,
                edited_verdict=body.edited_verdict,
                edited_correction=body.edited_correction,
                notes=body.notes,
                reviewer=body.reviewer,
                feedback_log=settings.data_dir / "feedback.jsonl",
            )
        except ReviewError as exc:
            status = 404 if "not found" in str(exc) else 422
            raise HTTPException(status, str(exc)) from exc
        return r.model_dump(mode="json")

    @app.get("/audits/{audit_id}/export")
    def export(
        audit_id: int,
        session: SessionDep,
        format: Annotated[str, Query(pattern="^(md|html|json|jsonl)$")] = "md",
    ) -> Response:
        try:
            data = build_export(session, audit_id)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        render, media = {
            "md": (to_markdown, "text/markdown"),
            "html": (to_html, "text/html"),
            "json": (to_json, "application/json"),
            "jsonl": (to_jsonl, "application/x-ndjson"),
        }[format]
        return Response(
            render(data),
            media_type=media,
            headers={"Content-Disposition": f'attachment; filename="audit_{audit_id}.{format}"'},
        )

    @app.get("/documents/{document_id}")
    def get_document(document_id: int, session: SessionDep) -> dict[str, object]:
        doc = session.get(Document, document_id)
        if doc is None:
            raise HTTPException(404, "document not found")
        return {"id": doc.id, "name": doc.name, "kind": doc.kind, "text": doc.text}

    return app


def __getattr__(name: str) -> FastAPI:
    # Lazily build the default app so importing this module has no side effects.
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
