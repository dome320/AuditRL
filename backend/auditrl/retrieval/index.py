from collections.abc import Iterable

from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz

from auditrl.llm import EvidenceCandidate
from auditrl.models import Document, Passage
from auditrl.verify.normalize import extract_periods, tokens


class EvidenceIndex:
    """BM25 over source passages, boosted by table row-label and period matches."""

    def __init__(self, documents: Iterable[Document], passages: Iterable[Passage]) -> None:
        self.documents = {d.id: d for d in documents}
        self.passages = [p for p in passages if p.document_id in self.documents]
        self.by_id = {p.id: p for p in self.passages}
        corpus = [tokens(self._index_text(p)) or ["_"] for p in self.passages]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def _index_text(self, p: Passage) -> str:
        doc = self.documents[p.document_id]
        return f"{doc.name} {p.caption or ''} {p.text}"

    def search(
        self, query: str, k: int = 5, doc_ids: set[int] | None = None, label: str | None = None
    ) -> list[Passage]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokens(query))
        q_periods = extract_periods(query)
        max_score = max(scores) if len(scores) and max(scores) > 0 else 1.0
        ranked: list[tuple[float, Passage]] = []
        for score, p in zip(scores, self.passages, strict=True):
            if doc_ids is not None and p.document_id not in doc_ids:
                continue
            s = score / max_score
            if label and p.row_label:
                s += fuzz.token_set_ratio(label.lower(), p.row_label.lower()) / 100
            if q_periods and p.headers and q_periods & extract_periods(" ".join(p.headers)):
                s += 0.25
            if s > 0:
                ranked.append((s, p))
        ranked.sort(key=lambda t: t[0], reverse=True)
        return [p for _, p in ranked[:k]]

    def candidate(self, p: Passage) -> EvidenceCandidate:
        return EvidenceCandidate(
            passage_id=p.id,
            document_name=self.documents[p.document_id].name,
            text=p.text,
            row_label=p.row_label,
            headers=p.headers,
            cells=p.cells,
            unit_scale=p.unit_scale,
        )

    def candidates(self, passages: Iterable[Passage]) -> list[EvidenceCandidate]:
        return [self.candidate(p) for p in passages]

    def doc_name(self, passage_id: int) -> str:
        return self.documents[self.by_id[passage_id].document_id].name
