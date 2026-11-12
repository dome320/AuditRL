from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(UTC)


class AuditStatus(StrEnum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class DocumentKind(StrEnum):
    report = "report"
    source = "source"


class ClaimType(StrEnum):
    arithmetic = "arithmetic"
    figure = "figure"
    citation = "citation"
    qualitative = "qualitative"


class Verdict(StrEnum):
    valid = "valid"
    invalid = "invalid"
    uncertain = "uncertain"


class ErrorType(StrEnum):
    arithmetic_error = "arithmetic_error"
    misstated_figure = "misstated_figure"
    wrong_period = "wrong_period"
    unit_error = "unit_error"
    citation_mismatch = "citation_mismatch"
    missing_source = "missing_source"


class ReviewAction(StrEnum):
    accept = "accept"
    reject = "reject"
    edit = "edit"


class AuditRun(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    status: AuditStatus = AuditStatus.pending
    error: str | None = None
    model: str
    prompt_version: str
    config_hash: str
    created_at: datetime = Field(default_factory=utcnow)
    completed_at: datetime | None = None
    latency_s: float | None = None
    cost_usd: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Document(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    audit_id: int = Field(foreign_key="auditrun.id", index=True)
    name: str
    kind: DocumentKind
    origin: str
    sha256: str
    content_type: str
    text: str


class Passage(SQLModel, table=True):
    """A locatable unit of a document: a paragraph, or one row of a table."""

    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id", index=True)
    ordinal: int
    page: int | None = None
    char_start: int
    char_end: int
    text: str
    table_id: int | None = None
    row_index: int | None = None
    row_label: str | None = None
    headers: list[str] | None = Field(default=None, sa_column=Column(JSON))
    cells: list[str] | None = Field(default=None, sa_column=Column(JSON))
    unit_scale: float = 1.0
    caption: str | None = None


class Claim(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    audit_id: int = Field(foreign_key="auditrun.id", index=True)
    document_id: int = Field(foreign_key="document.id")
    ordinal: int
    span_start: int
    span_end: int
    text: str
    claim_type: ClaimType
    checkable: bool
    cited_source: str | None = None
    figures: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    alignment_score: float = 100.0


class Finding(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    audit_id: int = Field(foreign_key="auditrun.id", index=True)
    claim_id: int = Field(foreign_key="claim.id", index=True)
    verdict: Verdict
    error_type: ErrorType | None = None
    evidence_passage_ids: list[int] = Field(default_factory=list, sa_column=Column(JSON))
    evidence_quotes: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    rationale: str
    correction: str | None = None
    confidence: float
    verifier: str
    checks: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow)


class Review(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    finding_id: int = Field(foreign_key="finding.id", index=True)
    action: ReviewAction
    edited_verdict: Verdict | None = None
    edited_correction: str | None = None
    notes: str | None = None
    reviewer: str = "anonymous"
    created_at: datetime = Field(default_factory=utcnow)
