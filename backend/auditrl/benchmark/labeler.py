"""Deterministic ground-truth labeler for agent-written reports.

Labels come from the filings' XBRL facts, keyed by (metric, fiscal year), and never from the
auditor under evaluation: no LLM, no retrieval, no auditor verdict logic. Only the number
parser is shared. A sentence is labeled only when every figure in it can be mapped to a fact
with confidence; everything else is reported as unlabeled and excluded from scoring.
"""

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from rapidfuzz import fuzz, utils

from auditrl.benchmark.edgar import Filing
from auditrl.benchmark.metrics import BY_KEY, MARGINS, METRICS
from auditrl.verify.normalize import Quantity, find_quantities, format_like

LABELER_VERSION = "labeler-v1"
BORDERLINE_PP = 0.15  # percentage points beyond stated rounding treated as "too close to call"

_ABBREV = re.compile(r"\b(Inc|Corp|Co|Ltd|Mr|Ms|Dr|vs|No|U\.S)\.")
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'$])")
_CITATION = re.compile(r"\((?:sources?|see)\s*:\s*([^)]+)\)", re.I)
_YEAR = re.compile(r"(?:fiscal(?:\s+year)?\s+|FY\s?'?)?((?:19|20)\d{2})\b", re.I)
_UP = re.compile(
    r"\b(gr[eo]w\w*|increas\w*|rose|rise\w*|up|jump\w*|climb\w*|expand\w*|gain\w*)\b", re.I
)
_DOWN = re.compile(
    r"\b(declin\w*|decreas\w*|fell|fall\w*|down|drop\w*|contract\w*|shr[ai]nk\w*)\b", re.I
)
_LOSS = re.compile(r"\b(loss|losses|deficit|outflows?)\b", re.I)
# Text between two figures that ties the second to the first's metric: "... from $X", "to $Y".
_CONNECTIVE = re.compile(r"^\s*(from|to|versus|vs\.?|compared (?:with|to)|against)\s*$", re.I)

_PHRASES: list[tuple[str, str]] = sorted(
    ((p, m.key) for m in METRICS for p in m.phrases), key=lambda t: -len(t[0])
)
_MARGIN_PHRASES = sorted(MARGINS, key=len, reverse=True)


@dataclass
class FigureLabel:
    value_text: str
    status: str  # correct | error | unlabeled
    reason: str
    metric: str | None = None
    fiscal_year: int | None = None
    kind: str | None = None  # figure | margin | growth
    error_type: str | None = None
    true_value: float | None = None
    correction: str | None = None
    provenance: list[str] = field(default_factory=list)
    position: int = 0


@dataclass
class SentenceLabel:
    quote: str
    start: int
    end: int
    verdict: str  # valid | invalid | unlabeled
    figures: list[FigureLabel]
    error_type: str | None = None
    correction: str | None = None
    correction_contains: str | None = None


class FactBook:
    """All values the task's filings report, per (metric, fiscal year), with provenance."""

    def __init__(self, filings: list[Filing], titles: dict[str, int]) -> None:
        self.values: dict[tuple[str, int], list[tuple[float, str]]] = {}
        for f in filings:
            for metric, by_year in f.facts.items():
                for year, fact in by_year.items():
                    self.values.setdefault((metric, year), []).append(
                        (fact.value, f"{fact.concept} FY{year} ({fact.accession})")
                    )
        self.titles = titles  # document title -> fiscal year of that 10-K
        self.years = sorted({y for _, y in self.values})

    def get(self, metric: str, year: int) -> list[tuple[float, str]]:
        return self.values.get((metric, year), [])


def split_sentences(text: str) -> list[tuple[int, int]]:
    masked = _ABBREV.sub(lambda m: m.group(0).replace(".", "\x00"), text)
    spans: list[tuple[int, int]] = []
    for para in re.finditer(r"[^\n]+", masked):
        line = para.group(0)
        if line.lstrip().startswith(("#", "|")):
            continue
        pos = 0
        for m in _SENT_END.finditer(line):
            spans.append((para.start() + pos, para.start() + m.start()))
            pos = m.end()
        spans.append((para.start() + pos, para.start() + len(line.rstrip())))
    out = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        if e - s > 0:
            out.append((s, e))
    return out


def _find_metric(text: str) -> tuple[str, int] | None:
    """Last metric phrase in `text` (closest to the number), as (metric key, position)."""
    low = text.lower()
    hits = [
        (m.start(), m.end(), key)
        for phrase, key in _PHRASES
        for m in re.finditer(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", low)
    ]
    # Drop matches nested in a longer one ("revenue" inside "cost of revenue").
    hits = [h for h in hits if not any(o != h and o[0] <= h[0] and h[1] <= o[1] for o in hits)]
    if not hits:
        return None
    start, _, key = max(hits, key=lambda h: h[1])
    return key, start


def _find_margin(text: str) -> str | None:
    low = text.lower()
    hits = [(low.rfind(p), p) for p in _MARGIN_PHRASES if p in low]
    return max(hits)[1] if hits else None


def _years(text: str) -> list[int]:
    return [int(m.group(1)) for m in _YEAR.finditer(text)]


def _match(
    q: Quantity, values: list[tuple[float, str]], loss_context: bool
) -> tuple[float, str] | None:
    lo, hi = q.interval
    for v, prov in values:
        target = -v if loss_context and v < 0 < q.value else v
        if lo - 1e-9 * abs(target) <= target <= hi + 1e-9 * abs(target):
            return v, prov
    return None


class Labeler:
    def __init__(self, book: FactBook) -> None:
        self.book = book

    def label(self, report: str) -> list[SentenceLabel]:
        out = []
        for s, e in split_sentences(report):
            sent = report[s:e]
            label = self._label_sentence(sent, s, e)
            if label is not None:
                out.append(label)
        return out

    def _label_sentence(self, sent: str, start: int, end: int) -> SentenceLabel | None:
        citations = [(m.start(), m.end(), m.group(1)) for m in _CITATION.finditer(sent)]
        masked = sent
        for cs, ce, _ in citations:
            masked = masked[:cs] + " " * (ce - cs) + masked[ce:]
        qs = [q for q in find_quantities(masked) if q.raw.strip()]
        if not qs:
            return None
        sentence_years = _years(masked)
        figures: list[FigureLabel] = []
        last_metric: str | None = None
        for i, q in enumerate(qs):
            prev_end = qs[i - 1].end if i else 0
            nxt_start = qs[i + 1].start if i + 1 < len(qs) else len(masked)
            before, after = masked[prev_end : q.start], masked[q.end : nxt_start]
            inherited = last_metric if _CONNECTIVE.match(before) else None
            fig = self._label_figure(q, masked, before, after, sentence_years, inherited)
            fig.position = q.start
            if fig.metric:
                last_metric = fig.metric
            figures.append(fig)

        self._check_citations(sent, citations, figures)

        errors = [f for f in figures if f.status == "error"]
        if errors:
            corrected = sent
            for f in errors:
                if f.correction and f.value_text in corrected:
                    corrected = corrected.replace(f.value_text, f.correction, 1)
            first = errors[0]
            return SentenceLabel(
                sent,
                start,
                end,
                "invalid",
                figures,
                error_type=first.error_type,
                correction=corrected if corrected != sent else None,
                correction_contains=first.correction,
            )
        verdict = "valid" if all(f.status == "correct" for f in figures) else "unlabeled"
        return SentenceLabel(sent, start, end, verdict, figures)

    def _label_figure(
        self,
        q: Quantity,
        sent: str,
        before: str,
        after: str,
        sentence_years: list[int],
        inherited_metric: str | None = None,
    ) -> FigureLabel:
        def unlabeled(reason: str, **kw: Any) -> FigureLabel:
            return FigureLabel(q.raw, "unlabeled", reason, **kw)

        if q.unit == "percent":
            return self._label_percent(q, sent, before, after, sentence_years)
        if q.unit not in ("currency", "number"):
            return unlabeled(f"unit {q.unit} not covered")

        hit = _find_metric(before) or _find_metric(after[:60])
        metric = hit[0] if hit else inherited_metric
        if metric is None:
            return unlabeled("no metric phrase near the figure")
        year = self._year_for(before, after, sentence_years)
        if year is None:
            return unlabeled("fiscal year not stated or ambiguous", metric=metric)
        if BY_KEY[metric].per_share != (q.scale == 1 and abs(q.value) < 1000):
            return unlabeled(
                "figure magnitude does not fit the metric", metric=metric, fiscal_year=year
            )
        values = self.book.get(metric, year)
        if not values:
            return unlabeled(
                "no filed value for this metric and year", metric=metric, fiscal_year=year
            )

        loss = bool(_LOSS.search(before[-40:] + after[:20]))
        matched = _match(q, values, loss)
        if matched:
            return FigureLabel(
                q.raw, "correct", "matches filing", metric, year, "figure", provenance=[matched[1]]
            )

        true_value, prov = values[0]
        correction = format_like(q, abs(true_value) if loss and true_value < 0 else true_value)
        for other in self.book.years:
            if other != year and _match(q, self.book.get(metric, other), loss):
                return FigureLabel(
                    q.raw,
                    "error",
                    f"matches fiscal {other}, not {year}",
                    metric,
                    year,
                    "figure",
                    "wrong_period",
                    true_value,
                    correction,
                    [prov],
                )
        for factor in (1e3, 1e-3, 1e6, 1e-6):
            if _match(q, [(v * factor, p) for v, p in values], loss):
                correction = format_like(q, true_value, sig_digits=4)
                return FigureLabel(
                    q.raw,
                    "error",
                    "off by a unit scale",
                    metric,
                    year,
                    "figure",
                    "unit_error",
                    true_value,
                    correction,
                    [prov],
                )
        return FigureLabel(
            q.raw,
            "error",
            "differs from filing",
            metric,
            year,
            "figure",
            "misstated_figure",
            true_value,
            correction,
            [prov],
        )

    def _year_for(self, before: str, after: str, sentence_years: list[int]) -> int | None:
        nearby = _years(after.split(",")[0][:40])
        if nearby:
            return nearby[0]
        prior = _years(before[-40:])
        if prior:
            return prior[-1]
        unique = sorted(set(sentence_years))
        return unique[0] if len(unique) == 1 else None

    def _label_percent(
        self, q: Quantity, sent: str, before: str, after: str, sentence_years: list[int]
    ) -> FigureLabel:
        def unlabeled(reason: str, **kw: Any) -> FigureLabel:
            return FigureLabel(q.raw, "unlabeled", reason, **kw)

        margin = _find_margin(before[-50:] + " " + after[:30])
        year = self._year_for(before, after, sentence_years)
        if margin:
            num_key, den_key = MARGINS[margin]
            if year is None:
                return unlabeled("fiscal year not stated or ambiguous", kind="margin")
            num, den = self.book.get(num_key, year), self.book.get(den_key, year)
            if not num or not den or den[0][0] == 0:
                return unlabeled("margin inputs not filed", kind="margin", fiscal_year=year)
            true = num[0][0] / den[0][0] * 100
            return self._compare_percent(
                q, true, "margin", f"{margin} FY{year}", year, [num[0][1], den[0][1]]
            )

        if _UP.search(sent) or _DOWN.search(sent):
            hit = _find_metric(before) or _find_metric(after[:60]) or _find_metric(sent)
            if hit is None:
                return unlabeled("growth without a metric", kind="growth")
            years = sorted(set(sentence_years))
            target = years[-1] if years else None
            if target is None:
                return unlabeled("growth period not stated", kind="growth", metric=hit[0])
            cur, prev = self.book.get(hit[0], target), self.book.get(hit[0], target - 1)
            if not cur or not prev or prev[0][0] == 0:
                return unlabeled(
                    "growth inputs not filed", kind="growth", metric=hit[0], fiscal_year=target
                )
            true = (cur[0][0] - prev[0][0]) / abs(prev[0][0]) * 100
            stated_negative_as_magnitude = bool(_DOWN.search(sent)) and q.value > 0
            compare = -true if stated_negative_as_magnitude else true
            label = self._compare_percent(
                q,
                compare,
                "growth",
                f"{hit[0]} growth FY{target - 1}->FY{target}",
                target,
                [cur[0][1], prev[0][1]],
                metric=hit[0],
            )
            if label.status == "correct" and (
                (_UP.search(sent) and not _DOWN.search(sent) and true < 0)
                or (_DOWN.search(sent) and not _UP.search(sent) and true > 0)
            ):
                label.status, label.error_type = "error", "arithmetic_error"
                label.reason = "stated direction contradicts the filings"
            return label
        return unlabeled("percentage of an unrecognized kind")

    @staticmethod
    def _compare_percent(
        q: Quantity,
        true: float,
        kind: str,
        what: str,
        year: int,
        prov: list[str],
        metric: str | None = None,
    ) -> FigureLabel:
        diff = abs(q.value - true)
        if diff <= q.half_ulp + 1e-9:
            return FigureLabel(
                q.raw, "correct", f"{what} = {true:.3f}%", metric, year, kind, provenance=prov
            )
        if diff <= q.half_ulp + BORDERLINE_PP:
            return FigureLabel(
                q.raw,
                "unlabeled",
                f"{what} = {true:.3f}%: within rounding noise",
                metric,
                year,
                kind,
            )
        return FigureLabel(
            q.raw,
            "error",
            f"{what} = {true:.3f}%",
            metric,
            year,
            kind,
            "arithmetic_error",
            true,
            format_like(q, abs(true) if q.value > 0 > true else true),
            prov,
        )

    def _check_citations(
        self, sent: str, citations: list[tuple[int, int, str]], figures: list[FigureLabel]
    ) -> None:
        prev_citation_end = 0
        for c_start, c_end, cited in citations:
            # A citation covers the figures between the previous citation and itself.
            covered = [f for f in figures if prev_citation_end <= f.position < c_start]
            prev_citation_end = c_end
            for title in (t.strip() for t in cited.split(";")):
                doc_year = self._resolve_title(title)
                for f in covered:
                    if f.status != "correct" or f.fiscal_year is None:
                        continue
                    if doc_year is None:
                        f.status = "unlabeled"
                        f.reason = f"cites {title!r}, which cannot be matched to a task filing"
                        continue
                    if f.fiscal_year not in (doc_year, doc_year - 1):
                        right = next(
                            (t for t, y in self.book.titles.items() if y == f.fiscal_year), None
                        )
                        if right is None:
                            continue
                        f.status, f.error_type = "error", "citation_mismatch"
                        f.reason = f"FY{f.fiscal_year} figure cited to the FY{doc_year} filing"
                        f.value_text, f.correction = title, right

    def _resolve_title(self, title: str) -> int | None:
        filed_years = set(self.book.titles.values())
        years = set(_years(title)) | {
            2000 + int(m) for m in re.findall(r"\bFY\s?'?(\d{2})\b", title)
        }
        if len(years) == 1 and re.search(r"10-?K|annual report|filing", title, re.I):
            (year,) = years
            return year if year in filed_years else None
        best = None
        for t, year in self.book.titles.items():
            score = fuzz.ratio(utils.default_process(title), utils.default_process(t))
            if score >= 90 and (best is None or score > best[0]):
                best = (score, year)
        return best[1] if best else None


def labels_to_expected(case_id: str, labels: list[SentenceLabel]) -> dict[str, Any]:
    """Convert to the scoring format shared with the curated dataset."""
    scored = []
    for lab in labels:
        if lab.verdict == "valid":
            scored.append({"quote": lab.quote, "verdict": "valid"})
        elif lab.verdict == "invalid":
            entry = {"quote": lab.quote, "verdict": "invalid", "error_type": lab.error_type}
            if lab.correction_contains:
                entry["correction_contains"] = lab.correction_contains
            scored.append(entry)
    figures = [f for lab in labels for f in lab.figures]
    return {
        "case_id": case_id,
        "labeler_version": LABELER_VERSION,
        "labels": scored,
        "unlabeled": [lab.quote for lab in labels if lab.verdict == "unlabeled"],
        "coverage": {
            "sentences_with_figures": len(labels),
            "labeled_sentences": len(scored),
            "figures": len(figures),
            "labeled_figures": sum(1 for f in figures if f.status != "unlabeled"),
            "errors": sum(1 for lab in labels if lab.verdict == "invalid"),
        },
        "detail": [asdict(lab) for lab in labels],
    }
