"""Retrieve judge evidence with the auditor's own index, inside or outside the database."""

import re

from auditrl.intake import ParsedDocument
from auditrl.judge.format import EvidenceItem
from auditrl.models import Document, DocumentKind, Passage
from auditrl.retrieval import EvidenceIndex


def document_title(doc: Document) -> str:
    first = doc.text.strip().splitlines()[0] if doc.text.strip() else ""
    return first.lstrip("# ").strip() or doc.name


def index_from_parsed(docs: list[ParsedDocument]) -> EvidenceIndex:
    """Build an in-memory index with stable ids (documents 1..n, passages numbered in order)."""
    documents, passages = [], []
    pid = 0
    for i, parsed in enumerate(docs, start=1):
        documents.append(
            Document(
                id=i,
                audit_id=0,
                name=parsed.name,
                kind=DocumentKind.source,
                origin="memory",
                sha256=parsed.sha256,
                content_type=parsed.content_type,
                text=parsed.text,
            )
        )
        for j, p in enumerate(parsed.passages):
            pid += 1
            passages.append(
                Passage(
                    id=pid,
                    document_id=i,
                    ordinal=j,
                    page=p.page,
                    char_start=p.char_start,
                    char_end=p.char_end,
                    text=p.text,
                    table_id=p.table_id,
                    row_index=p.row_index,
                    row_label=p.row_label,
                    headers=p.headers,
                    cells=p.cells,
                    unit_scale=p.unit_scale,
                    caption=p.caption,
                )
            )
    return EvidenceIndex(documents, passages)


# Ratios are stated by name but computed from rows with other names.
_EXPANSIONS = [
    (re.compile(r"\bgross (profit )?margin", re.I), "gross profit total revenue"),
    (re.compile(r"\boperating margin", re.I), "operating income total revenue"),
    (re.compile(r"\b(net|profit|net profit) margin", re.I), "net income total revenue"),
    (re.compile(r"\bfree cash flow", re.I), "operating activities capital expenditures"),
]


def expand_query(claim: str) -> str:
    extra = [terms for pattern, terms in _EXPANSIONS if pattern.search(claim)]
    return " ".join([claim, *extra])


def retrieve(index: EvidenceIndex, claim: str, k: int = 8) -> list[EvidenceItem]:
    items = []
    ranked = index.search(expand_query(claim), k=k + 4)
    # Bare headings carry no evidence; the document title is already shown with each passage.
    ranked = [p for p in ranked if not p.text.lstrip().startswith("#")][:k]
    for n, p in enumerate(ranked, start=1):
        items.append(
            EvidenceItem(
                eid=f"E{n}",
                document=document_title(index.documents[p.document_id]),
                text=p.text,
                row_label=p.row_label,
                headers=p.headers,
                cells=p.cells,
                unit_scale=p.unit_scale,
                passage_id=p.id,
            )
        )
    return items
