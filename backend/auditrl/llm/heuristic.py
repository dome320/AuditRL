"""Offline, rule-based stand-in for the LLM.

Lets the pipeline run without API keys (demos, CI, tests). It handles common sentence
patterns in financial write-ups; it is a fallback, not a substitute for a real model.
"""

import re

from rapidfuzz import fuzz

from auditrl.llm.base import (
    ComputationProposal,
    EvidenceCandidate,
    EvidenceSelection,
    ExtractedClaim,
    ExtractedFigure,
    ExtractionResult,
    LLMClient,
    Operand,
)
from auditrl.verify.normalize import Quantity, extract_periods, find_quantities, label_score

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")
_CITATION_RES = [
    re.compile(r"\((?:source|sources|per|see)\s*:?\s*([^)]+)\)", re.I),
    re.compile(
        r"\b(?:according to|as reported in|as disclosed in|per|citing)\s+(?:the\s+|its\s+)?"
        r"([A-Z][\w&.'\- ]*?(?:10-K|10-Q|8-K|20-F|filing|report|statement|release|deck)\b"
        r"(?:\s+(?:for\s+)?(?:fiscal\s+)?(?:FY\s?)?\d{2,4})?)",
        re.I,
    ),
]
_FORECAST = re.compile(
    r"\b(expect|expects|expected|will|forecast|anticipate|likely|should|could|believe|"
    r"estimate|guidance|outlook|we think|may)\b",
    re.I,
)
_GROWTH_UP = re.compile(
    r"\b(grew|grow|increased?|rose|up|growth|jumped|climbed|expanded|gain)\b", re.I
)
_GROWTH_DOWN = re.compile(r"\b(declined?|decreased?|fell|down|dropped|contracted|shrank)\b", re.I)
_RATIO_WORDS = re.compile(
    r"(\bmargin\b|\bshare of\b|\bas a percentage\b|\bpercent of\b|% of\b|"
    r"\bof (?:total )?(?:net )?(?:operating )?(?:revenues?|sales)\b|"
    r"\brepresent(?:ed|s|ing)?\b|\baccounted for\b)",
    re.I,
)
_METRIC_NOUN = re.compile(
    r"\b(revenues?|sales|income|margin|expenses?|profit|earnings|cash|flow|eps|assets|"
    r"liabilities|debt|dividends?|capex|spending|costs?|loss|losses)\b",
    re.I,
)
_SUM_WORDS = re.compile(r"\b(total(?:ed|led|ing)?|combined|together|sum|aggregate)\b", re.I)
_DENOMINATOR = re.compile(r"\b(revenue|revenues|sales|net sales|total)\b", re.I)
_STOP = set(
    "a an the of in on at to from for by and or was were is are be been its their his her "
    "with as up down grew rose fell increased decreased declined reached totaled totalled "
    "reported posted had has have generated delivered year fiscal fy compared versus vs "
    "while which that this these those than over during approximately about roughly nearly "
    "billion million thousand trillion percent margin based".split()
)
_KEEP = {"margin"}


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z&'\-]*", text)


def _metric_phrase(text: str) -> str:
    ws = [w for w in _words(text) if w.lower() not in _STOP or w.lower() in _KEEP]
    return " ".join(ws[-6:])


def _subject(sentence: str, quantities: list[Quantity]) -> str:
    head = sentence[: quantities[0].start] if quantities else sentence
    head = re.split(
        r"\b(grew|rose|fell|increased|decreased|declined|was|were|reached|totaled|totalled|of)\b",
        head,
        maxsplit=1,
        flags=re.I,
    )[0]
    return _metric_phrase(head)


def _period_near(sentence: str, q: Quantity, nxt: Quantity | None) -> str | None:
    after = sentence[q.end : nxt.start if nxt else len(sentence)]
    m = re.search(
        r"((?:fiscal(?: year)?|FY|Q[1-4]|calendar)\s*'?\d{2,4}|(?:19|20)\d{2})", after, re.I
    )
    if m:
        return m.group(1)
    before = sentence[: q.start]
    found = list(
        re.finditer(r"((?:fiscal(?: year)?|FY|Q[1-4])\s*'?\d{2,4}|(?:19|20)\d{2})", before, re.I)
    )
    return found[-1].group(1) if found else None


def _cited_source(sentence: str) -> str | None:
    for rx in _CITATION_RES:
        m = rx.search(sentence)
        if m:
            return m.group(1).strip().rstrip(".")
    return None


def _split_sentences(chunk: str) -> list[str]:
    out: list[str] = []
    for line in chunk.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "|", "---")):
            continue
        line = re.sub(r"^[-*]\s+", "", line)
        out.extend(s.strip() for s in _SENTENCE_RE.split(line) if s.strip())
    return out


def _classify(sentence: str, qs: list[Quantity]) -> str:
    money = [q for q in qs if q.unit in ("currency", "number")]
    pct = [q for q in qs if q.unit in ("percent", "bps")]
    if (
        pct
        and money
        and (
            _GROWTH_UP.search(sentence)
            or _GROWTH_DOWN.search(sentence)
            or _RATIO_WORDS.search(sentence)
        )
    ):
        return "arithmetic"
    if len(money) >= 3 and _SUM_WORDS.search(sentence):
        return "arithmetic"
    return "figure"


def _sum_result_index(sentence: str, money: list[Quantity]) -> int:
    for i, q in enumerate(money):
        if _SUM_WORDS.search(sentence[max(0, q.start - 40) : q.start]):
            return i
    return max(range(len(money)), key=lambda i: money[i].value)


class HeuristicClient(LLMClient):
    name = "heuristic"

    def extract_claims(self, chunk: str) -> ExtractionResult:
        claims: list[ExtractedClaim] = []
        for sentence in _split_sentences(chunk):
            if len(sentence) < 15:
                continue
            cited = _cited_source(sentence)
            source_span = None
            if cited:
                idx = sentence.find(cited)
                source_span = (idx, idx + len(cited))
            qs = [
                q
                for q in find_quantities(sentence)
                if not (source_span and source_span[0] <= q.start < source_span[1])
            ]
            if not qs or _FORECAST.search(sentence):
                claims.append(
                    ExtractedClaim(text=sentence, claim_type="qualitative", checkable=False)
                )
                continue
            ctype = _classify(sentence, qs)
            subject = _subject(sentence, qs)
            money = [q for q in qs if q.unit in ("currency", "number")]
            sum_result = (
                money[_sum_result_index(sentence, money)]
                if ctype == "arithmetic" and not any(q.unit == "percent" for q in qs)
                else None
            )
            figures: list[ExtractedFigure] = []
            for i, q in enumerate(qs):
                nxt = qs[i + 1] if i + 1 < len(qs) else None
                prev_end = qs[i - 1].end if i else 0
                local = _metric_phrase(sentence[prev_end : q.start])
                has_noun = bool(_METRIC_NOUN.search(local))
                metric = local if has_noun else f"{subject} {local}".strip()
                if ctype == "arithmetic":
                    is_result = q is sum_result or (sum_result is None and q.unit == "percent")
                    role = "result" if is_result else "input"
                else:
                    role = "standalone"
                figures.append(
                    ExtractedFigure(
                        value_text=q.raw,
                        metric=metric or subject,
                        period=_period_near(sentence, q, nxt),
                        role=role,
                    )
                )
            claim_type = "citation" if cited and ctype == "figure" else ctype
            claims.append(
                ExtractedClaim(
                    text=sentence,
                    claim_type=claim_type,
                    checkable=True,
                    cited_source=cited,
                    figures=figures,
                )
            )
        return ExtractionResult(claims=claims)

    def locate_evidence(
        self, claim_text: str, figure: ExtractedFigure, candidates: list[EvidenceCandidate]
    ) -> EvidenceSelection:
        periods = extract_periods(figure.period or "")
        best: tuple[float, EvidenceSelection] | None = None
        for c in candidates:
            if c.cells and c.headers and c.row_label:
                score = label_score(figure.metric, c.row_label)
                if score is None:
                    continue
                col = self._pick_column(c.headers, c.cells, periods)
                if col is None:
                    continue
                sel = EvidenceSelection(
                    found=True,
                    passage_id=c.passage_id,
                    quote=c.text,
                    value_text=c.cells[col],
                    reason=f"row '{c.row_label}', column '{c.headers[col]}'",
                )
            else:
                sel_score = self._match_prose(figure, periods, c)
                if sel_score is None:
                    continue
                score, sel = sel_score
            if best is None or score > best[0]:
                best = (score, sel)
        return best[1] if best else EvidenceSelection(found=False, reason="no matching row/period")

    @staticmethod
    def _pick_column(headers: list[str], cells: list[str], periods: set[str]) -> int | None:
        numeric = [i for i, cell in enumerate(cells) if find_quantities(cell)]
        if not numeric:
            return None
        if periods:
            hits = [
                i for i in numeric if i < len(headers) and extract_periods(headers[i]) & periods
            ]
            return hits[0] if hits else None
        return numeric[0] if len(numeric) == 1 else None

    @staticmethod
    def _match_prose(
        figure: ExtractedFigure, periods: set[str], c: EvidenceCandidate
    ) -> tuple[float, EvidenceSelection] | None:
        best: tuple[float, EvidenceSelection] | None = None
        for sentence in _split_sentences(c.text):
            if periods and not (extract_periods(sentence) & periods):
                continue
            qs = find_quantities(sentence)
            for i, q in enumerate(qs):
                prev_end = qs[i - 1].end if i else 0
                ctx = sentence[max(prev_end, q.start - 80) : q.start]
                score = fuzz.token_set_ratio(figure.metric.lower(), ctx.lower())
                if score >= 75 and (best is None or score > best[0]):
                    best = (
                        score,
                        EvidenceSelection(
                            found=True,
                            passage_id=c.passage_id,
                            quote=sentence,
                            value_text=q.raw,
                            reason="prose match",
                        ),
                    )
        return best

    def propose_computation(
        self, claim_text: str, figures: list[ExtractedFigure], candidates: list[EvidenceCandidate]
    ) -> ComputationProposal | None:
        results = [f for f in figures if f.role == "result"]
        inputs = [f for f in figures if f.role == "input"]
        if not results:
            return None
        result = results[0]
        res_q = find_quantities(result.value_text)
        if not res_q:
            return None
        names = "abcdefgh"

        if res_q[0].unit not in ("percent", "bps"):
            if len(inputs) < 2:
                return None
            ops = [
                Operand(name=names[i], value_text=f.value_text, source="claim")
                for i, f in enumerate(inputs[:8])
            ]
            return ComputationProposal(
                operation="sum",
                expression=" + ".join(o.name for o in ops),
                operands=ops,
                result_value_text=result.value_text,
            )

        is_growth = bool(_GROWTH_UP.search(claim_text) or _GROWTH_DOWN.search(claim_text))
        is_ratio = bool(_RATIO_WORDS.search(claim_text))
        if is_ratio and len(inputs) >= 2:
            den_idx = next(
                (i for i, f in enumerate(inputs[:2]) if _DENOMINATOR.search(f.metric)),
                max(range(2), key=lambda i: find_quantities(inputs[i].value_text)[0].value),
            )
            num, den = inputs[1 - den_idx], inputs[den_idx]
            return ComputationProposal(
                operation="margin",
                expression="a / b * 100",
                operands=[
                    Operand(name="a", value_text=num.value_text, source="claim"),
                    Operand(name="b", value_text=den.value_text, source="claim"),
                ],
                result_value_text=result.value_text,
            )
        if not is_growth:
            return None
        decline = bool(_GROWTH_DOWN.search(claim_text)) and res_q[0].value > 0
        expr = "(a - b) / a * 100" if decline else "(b - a) / a * 100"
        if len(inputs) >= 2:
            old, new = self._order_by_time(claim_text, inputs[0], inputs[1])
            return ComputationProposal(
                operation="growth",
                expression=expr,
                operands=[
                    Operand(name="a", value_text=old.value_text, source="claim"),
                    Operand(name="b", value_text=new.value_text, source="claim"),
                ],
                result_value_text=result.value_text,
            )
        if len(inputs) == 1:
            prior = self._prior_period_value(inputs[0], candidates)
            if prior is None:
                return None
            return ComputationProposal(
                operation="growth",
                expression=expr,
                operands=[
                    prior,
                    Operand(name="b", value_text=inputs[0].value_text, source="claim"),
                ],
                result_value_text=result.value_text,
            )
        return None

    @staticmethod
    def _order_by_time(
        claim_text: str, f1: ExtractedFigure, f2: ExtractedFigure
    ) -> tuple[ExtractedFigure, ExtractedFigure]:
        y1 = max(extract_periods(f1.period or "") or {"0"})
        y2 = max(extract_periods(f2.period or "") or {"0"})
        if y1 != y2 and "0" not in (y1, y2):
            return (f1, f2) if y1 < y2 else (f2, f1)
        i1, i2 = claim_text.find(f1.value_text), claim_text.find(f2.value_text)
        from_first = re.search(r"\bfrom\b", claim_text[: max(i1, i2)], re.I)
        if from_first or i1 < i2:
            return (f1, f2) if i1 < i2 else (f2, f1)
        return f2, f1

    @staticmethod
    def _prior_period_value(
        current: ExtractedFigure, candidates: list[EvidenceCandidate]
    ) -> Operand | None:
        years = sorted(p for p in extract_periods(current.period or "") if p.isdigit())
        if not years:
            return None
        prior = str(int(years[-1]) - 1)
        best: tuple[float, Operand] | None = None
        for c in candidates:
            if not (c.cells and c.headers and c.row_label):
                continue
            score = label_score(current.metric, c.row_label)
            if score is None:
                continue
            for i, h in enumerate(c.headers):
                if i < len(c.cells) and prior in extract_periods(h) and find_quantities(c.cells[i]):
                    if best is None or score > best[0]:
                        op = Operand(
                            name="a",
                            value_text=c.cells[i],
                            source="evidence",
                            passage_id=c.passage_id,
                        )
                        best = (score, op)
        return best[1] if best else None

    def write_rationale(self, claim_text: str, verdict: str, facts: list[str]) -> str:
        return " ".join(facts)
