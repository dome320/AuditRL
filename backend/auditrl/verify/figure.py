import re
from dataclasses import replace

from auditrl.llm import ExtractedFigure, LLMClient
from auditrl.models import ErrorType, Passage, Verdict
from auditrl.retrieval import EvidenceIndex
from auditrl.verify.normalize import (
    Quantity,
    extract_periods,
    format_like,
    label_score,
    parse_quantity,
    unit_error_factor,
    values_match,
)
from auditrl.verify.types import CheckResult, Evidence, uncertain

_NEGATIVE_WORDS = re.compile(r"\b(loss|losses|deficit|outflows?|negative)\b", re.I)


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def contains(haystack: str, needle: str) -> bool:
    return bool(needle) and squash(needle) in squash(haystack)


def parse_evidence_value(value_text: str, passage: Passage) -> Quantity | None:
    """Table cells inherit the table's stated scale ('in millions'); prose carries its own."""
    return parse_quantity(value_text, default_scale=passage.unit_scale)


def _describe(q: Quantity, template: Quantity) -> str:
    return format_like(template, q.value) if q.scale != template.scale else q.raw


def verify_figure(
    claim_text: str,
    figure: ExtractedFigure,
    index: EvidenceIndex,
    llm: LLMClient,
    *,
    doc_ids: set[int] | None = None,
    rel_tol: float = 1e-6,
    top_k: int = 5,
) -> CheckResult:
    kind = "figure"
    standalone = figure.role == "standalone"
    if not contains(claim_text, figure.value_text):
        return uncertain(kind, f"'{figure.value_text}' does not appear in the claim.", standalone)
    claimed = parse_quantity(figure.value_text)
    if claimed is None:
        return uncertain(kind, f"Could not parse '{figure.value_text}'.", standalone)

    query = " ".join(filter(None, [figure.entity, figure.metric, figure.period]))
    passages = index.search(query, k=top_k, doc_ids=doc_ids, label=figure.metric)
    if not passages:
        return uncertain(kind, f"No source passage found for {figure.metric}.", standalone)
    allowed = {p.id for p in passages}
    sel = llm.locate_evidence(claim_text, figure, index.candidates(passages))

    if not sel.found or sel.passage_id not in allowed or not sel.value_text:
        return uncertain(
            kind,
            f"No source reports {figure.metric}"
            + (f" for {figure.period}" if figure.period else "")
            + ".",
            standalone,
            figure=figure.model_dump(),
        )
    passage = index.by_id[sel.passage_id]
    quote = sel.quote if contains(passage.text, sel.quote) else passage.text
    if not contains(passage.text, sel.value_text) or not contains(quote, sel.value_text):
        return uncertain(
            kind, "Selected evidence could not be confirmed in the source.", standalone
        )
    actual = parse_evidence_value(sel.value_text, passage)
    if actual is None:
        return uncertain(kind, f"Could not parse source value '{sel.value_text}'.", standalone)

    doc = index.doc_name(passage.id)
    evidence = [Evidence(passage.id, quote)]
    details = {
        "figure": figure.model_dump(),
        "claimed": claimed.value,
        "actual": actual.value,
        "source_value_text": sel.value_text,
        "selection_reason": sel.reason,
    }
    actual_desc = _describe(actual, claimed)
    # "a net loss of $2.7 billion" states the magnitude of a value the filing shows as (2,722).
    if claimed.value > 0 > actual.value and _NEGATIVE_WORDS.search(claim_text):
        actual = replace(actual, value=-actual.value, magnitude=-actual.magnitude)
    if values_match(claimed, actual, rel_tol):
        return CheckResult(
            kind,
            Verdict.valid,
            facts=[
                f"{doc} reports {figure.metric} as {actual_desc}, matching '{figure.value_text}'."
            ],
            evidence=evidence,
            confidence=0.9 if passage.cells else 0.8,
            details=details,
        )

    alt = _alternate_row_match(claimed, figure, passages, passage.id, rel_tol)
    if alt is not None:
        alt_passage, alt_cell = alt
        return CheckResult(
            kind,
            Verdict.valid,
            facts=[
                f"{index.doc_name(alt_passage.id)} reports {alt_passage.row_label} as {alt_cell}, "
                f"matching '{figure.value_text}' (the metric label is ambiguous)."
            ],
            evidence=[Evidence(alt_passage.id, alt_passage.text)],
            confidence=0.6,
            details={**details, "ambiguous_metric": True},
        )

    correction = format_like(claimed, actual.value)
    replacements = [(figure.value_text, correction)]

    other = _other_period_match(claimed, passage, sel.value_text, rel_tol)
    if other is not None:
        header, cell = other
        return CheckResult(
            kind,
            Verdict.invalid,
            error_type=ErrorType.wrong_period,
            facts=[
                f"'{figure.value_text}' matches {figure.metric} for {header} ({cell}) in {doc}, "
                f"not {figure.period or 'the stated period'}, which is {actual_desc}."
            ],
            evidence=evidence,
            replacements=replacements,
            confidence=0.85,
            details={**details, "matched_period": header},
        )
    if unit_error_factor(claimed, actual, rel_tol) is not None:
        fixed = format_like(claimed, actual.value, sig_digits=4)
        replacements = [(figure.value_text, fixed)]
        return CheckResult(
            kind,
            Verdict.invalid,
            error_type=ErrorType.unit_error,
            facts=[
                f"{doc} reports {figure.metric} as {fixed}; '{figure.value_text}' is off by "
                "a unit scale (e.g. thousands vs millions)."
            ],
            evidence=evidence,
            replacements=replacements,
            confidence=0.85,
            details=details,
        )
    return CheckResult(
        kind,
        Verdict.invalid,
        error_type=ErrorType.misstated_figure,
        facts=[f"{doc} reports {figure.metric} as {actual_desc}, not '{figure.value_text}'."],
        evidence=evidence,
        replacements=replacements,
        confidence=0.85 if passage.cells else 0.75,
        details=details,
    )


def _alternate_row_match(
    claimed: Quantity,
    figure: ExtractedFigure,
    passages: list[Passage],
    chosen_id: int,
    rel_tol: float,
) -> tuple[Passage, str] | None:
    """Guard against false accusations: another plausible row for the same period may match."""
    periods = extract_periods(figure.period or "")
    for p in passages:
        if p.id == chosen_id or not (p.cells and p.headers and p.row_label):
            continue
        if label_score(figure.metric, p.row_label) is None:
            continue
        for header, cell in zip(p.headers, p.cells, strict=False):
            if periods and not (extract_periods(header) & periods):
                continue
            q = parse_evidence_value(cell, p)
            if q is not None and values_match(claimed, q, rel_tol):
                return p, cell
    return None


def _other_period_match(
    claimed: Quantity, passage: Passage, chosen_cell: str, rel_tol: float
) -> tuple[str, str] | None:
    if not passage.cells or not passage.headers:
        return None
    for header, cell in zip(passage.headers, passage.cells, strict=False):
        if squash(cell) == squash(chosen_cell) or squash(chosen_cell) in squash(cell):
            continue
        q = parse_evidence_value(cell, passage)
        if q is not None and values_match(claimed, q, rel_tol):
            return header, cell
    return None
