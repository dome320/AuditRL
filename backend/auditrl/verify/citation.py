import re
from pathlib import Path

from rapidfuzz import fuzz

from auditrl.llm import ExtractedFigure, LLMClient
from auditrl.models import Document, ErrorType, Verdict
from auditrl.retrieval import EvidenceIndex
from auditrl.verify.figure import verify_figure
from auditrl.verify.types import CheckResult, uncertain

MIN_SOURCE_MATCH = 70


def display_name(doc: Document) -> str:
    return re.sub(r"_+", " ", Path(doc.name).stem).strip()


def _doc_title(doc: Document) -> str:
    first = doc.text.strip().splitlines()[0] if doc.text.strip() else ""
    return first.lstrip("# ").strip()


def match_source(cited: str, documents: list[Document]) -> Document | None:
    best: tuple[float, Document] | None = None
    for doc in documents:
        score = max(
            fuzz.token_set_ratio(cited.lower(), display_name(doc).lower()),
            fuzz.token_set_ratio(cited.lower(), _doc_title(doc).lower()),
        )
        if best is None or score > best[0]:
            best = (score, doc)
    return best[1] if best and best[0] >= MIN_SOURCE_MATCH else None


def verify_citation(
    claim_text: str,
    cited_source: str,
    figures: list[ExtractedFigure],
    index: EvidenceIndex,
    llm: LLMClient,
    *,
    rel_tol: float = 1e-6,
    top_k: int = 5,
) -> list[CheckResult]:
    """Check figures against the cited document; on a miss, look for the true source."""
    kind = "citation"
    sources = [d for d in index.documents.values() if d.kind == "source"]
    cited_doc = match_source(cited_source, sources)
    if cited_doc is None:
        return [
            uncertain(
                kind,
                f"The cited source '{cited_source}' was not among the provided documents, "
                "so the citation cannot be checked.",
                cited_source=cited_source,
            )
        ]

    results: list[CheckResult] = []
    for fig in figures:
        if fig.role == "result":
            continue
        in_cited = verify_figure(
            claim_text, fig, index, llm, doc_ids={cited_doc.id}, rel_tol=rel_tol, top_k=top_k
        )
        if in_cited.verdict != Verdict.uncertain:
            in_cited.kind = "citation+figure"
            in_cited.facts.append(
                f"The figure is checked against the cited {display_name(cited_doc)}."
            )
            results.append(in_cited)
            continue
        others = {d.id for d in sources if d.id != cited_doc.id}
        elsewhere = (
            verify_figure(claim_text, fig, index, llm, doc_ids=others, rel_tol=rel_tol, top_k=top_k)
            if others
            else None
        )
        if elsewhere is not None and elsewhere.verdict == Verdict.valid:
            true_doc = index.documents[index.by_id[elsewhere.evidence[0].passage_id].document_id]
            results.append(
                CheckResult(
                    kind,
                    Verdict.invalid,
                    error_type=ErrorType.citation_mismatch,
                    facts=[
                        f"{display_name(cited_doc)} does not report {fig.metric}"
                        + (f" for {fig.period}" if fig.period else "")
                        + f"; the figure '{fig.value_text}' comes from {display_name(true_doc)}."
                    ],
                    evidence=elsewhere.evidence,
                    replacements=[(cited_source, display_name(true_doc))],
                    confidence=0.8,
                    details={"cited_document_id": cited_doc.id, "true_document_id": true_doc.id},
                )
            )
        elif elsewhere is not None and elsewhere.verdict == Verdict.invalid:
            results.append(elsewhere)
        else:
            results.append(
                uncertain(
                    kind,
                    f"Neither the cited {display_name(cited_doc)} nor other sources report "
                    f"{fig.metric}" + (f" for {fig.period}." if fig.period else "."),
                    decisive=fig.role == "standalone",
                )
            )
    if not results:
        results.append(uncertain(kind, "The claim cites a source but states no checkable figure."))
    return results
