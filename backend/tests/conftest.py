from pathlib import Path

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine, select

from auditrl.config import Settings
from auditrl.intake import parse_text, store_document
from auditrl.llm import HeuristicClient
from auditrl.models import AuditRun, DocumentKind, Passage, init_db
from auditrl.retrieval import EvidenceIndex

REPO = Path(__file__).resolve().parents[2]

APPLE_10K = """# Apple Inc. Form 10-K Fiscal 2023

Consolidated Statements of Operations (in millions)

| | Fiscal 2023 | Fiscal 2022 |
|---|---|---|
| Services net sales | 85,200 | 78,129 |
| Total net sales | $ 383,285 | $ 394,328 |
| Operating income | 114,301 | 119,437 |
| Net income (loss) | $ 96,995 | $ (2,722) |
"""

APPLE_10K_2021 = """# Apple Inc. Form 10-K Fiscal 2021

Consolidated Statements of Operations (in millions)

| | Fiscal 2021 | Fiscal 2020 |
|---|---|---|
| Total net sales | $ 365,817 | $ 274,515 |
"""


@pytest.fixture
def engine():
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    init_db(eng)
    return eng


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(model="heuristic", data_dir=tmp_path, llm_rationale=True)


@pytest.fixture
def llm() -> HeuristicClient:
    return HeuristicClient()


@pytest.fixture
def index(engine) -> EvidenceIndex:
    with Session(engine) as session:
        run = AuditRun(model="heuristic", prompt_version="v1", config_hash="x")
        session.add(run)
        session.flush()
        docs = [
            store_document(
                session,
                run.id,
                parse_text("Apple_10-K_FY2023.md", APPLE_10K),
                DocumentKind.source,
                "t",
            ),
            store_document(
                session,
                run.id,
                parse_text("Apple_10-K_FY2021.md", APPLE_10K_2021),
                DocumentKind.source,
                "t",
            ),
        ]
        session.commit()
        passages = session.exec(select(Passage)).all()
        for d in docs:
            session.refresh(d)
        session.expunge_all()
    return EvidenceIndex(docs, passages)
