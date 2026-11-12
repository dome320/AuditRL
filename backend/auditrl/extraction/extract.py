import logging
import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from auditrl.llm import ExtractedClaim, LLMClient

log = logging.getLogger(__name__)

MIN_ALIGNMENT_SCORE = 85.0


@dataclass
class AlignedClaim:
    claim: ExtractedClaim
    start: int
    end: int
    score: float


def chunk_text(text: str, max_chars: int) -> list[tuple[int, str]]:
    """Split on paragraph boundaries into (offset, chunk) pairs of at most ~max_chars."""
    paragraphs = [
        (m.start(), m.end()) for m in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\s*$)", text, re.S)
    ]
    chunks: list[tuple[int, str]] = []
    cur: tuple[int, int] | None = None
    for p_start, p_end in paragraphs:
        if cur and p_end - cur[0] > max_chars:
            chunks.append((cur[0], text[cur[0] : cur[1]]))
            cur = None
        cur = (cur[0] if cur else p_start, p_end)
    if cur:
        chunks.append((cur[0], text[cur[0] : cur[1]]))
    return chunks


def align_span(claim_text: str, haystack: str, cursor: int = 0) -> tuple[int, int, float] | None:
    """Locate `claim_text` in `haystack`, tolerating whitespace drift and small paraphrase."""
    needle = claim_text.strip()
    if not needle:
        return None
    for start_at in (cursor, 0):
        idx = haystack.find(needle, start_at)
        if idx != -1:
            return idx, idx + len(needle), 100.0
    pattern = r"\s+".join(re.escape(t) for t in needle.split())
    m = re.search(pattern, haystack)
    if m:
        return m.start(), m.end(), 99.0
    res = fuzz.partial_ratio_alignment(needle, haystack, score_cutoff=MIN_ALIGNMENT_SCORE)
    if res is None:
        return None
    return res.dest_start, res.dest_end, float(res.score)


def extract_claims(report_text: str, llm: LLMClient, chunk_chars: int = 6000) -> list[AlignedClaim]:
    aligned: list[AlignedClaim] = []
    seen: set[tuple[int, int]] = set()
    for offset, chunk in chunk_text(report_text, chunk_chars):
        result = llm.extract_claims(chunk)
        cursor = 0
        for claim in result.claims:
            span = align_span(claim.text, chunk, cursor)
            if span is None:
                log.warning("dropping unaligned claim: %r", claim.text[:80])
                continue
            start, end, score = span
            cursor = end
            key = (offset + start, offset + end)
            if key in seen:
                continue
            seen.add(key)
            if score < 100.0:
                claim = claim.model_copy(update={"text": chunk[start:end]})
            if claim.claim_type == "qualitative":
                claim = claim.model_copy(update={"checkable": False})
            aligned.append(AlignedClaim(claim, offset + start, offset + end, score))
    aligned.sort(key=lambda a: a.start)
    return aligned
