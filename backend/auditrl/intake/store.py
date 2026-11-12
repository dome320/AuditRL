from sqlmodel import Session

from auditrl.intake.parsers import ParsedDocument
from auditrl.models import Document, DocumentKind, Passage


def store_document(
    session: Session,
    audit_id: int,
    parsed: ParsedDocument,
    kind: DocumentKind,
    origin: str,
) -> Document:
    doc = Document(
        audit_id=audit_id,
        name=parsed.name,
        kind=kind,
        origin=origin,
        sha256=parsed.sha256,
        content_type=parsed.content_type,
        text=parsed.text,
    )
    session.add(doc)
    session.flush()
    for i, p in enumerate(parsed.passages):
        session.add(
            Passage(
                document_id=doc.id,
                ordinal=i,
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
    session.flush()
    return doc
