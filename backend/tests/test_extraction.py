from auditrl.extraction import align_span, chunk_text, extract_claims
from auditrl.llm import HeuristicClient


def test_align_exact_whitespace_and_fuzzy():
    hay = "Intro.  Revenue rose 5%\nto $10 billion. End."
    assert align_span("Intro.", hay) == (0, 6, 100.0)
    start, end, score = align_span("Revenue rose 5% to $10 billion.", hay)
    assert hay[start:end] == "Revenue rose 5%\nto $10 billion." and score == 99.0
    fuzzy = align_span("Revenue rose 5% to $10 bilion.", hay)
    assert fuzzy is not None and fuzzy[2] >= 85
    assert align_span("Completely unrelated sentence here.", hay) is None


def test_chunk_text_respects_paragraphs():
    text = "a" * 50 + "\n\n" + "b" * 50 + "\n\n" + "c" * 50
    chunks = chunk_text(text, 110)
    assert [c for _, c in chunks] == ["a" * 50 + "\n\n" + "b" * 50, "c" * 50]
    for offset, chunk in chunks:
        assert text[offset : offset + len(chunk)] == chunk


def test_extract_claims_types_and_spans():
    report = (
        "# Title\n\nRevenue was $10.0 billion in 2023 (Source: Acme 10-K 2023). "
        "Sales grew 10.0% from $9.0 billion in 2022 to $9.9 billion in 2023.\n\n"
        "We expect a strong year ahead."
    )
    claims = extract_claims(report, HeuristicClient())
    types = [a.claim.claim_type for a in claims]
    assert types == ["citation", "arithmetic", "qualitative"]
    assert claims[0].claim.cited_source == "Acme 10-K 2023"
    assert not claims[2].claim.checkable
    for a in claims:
        assert report[a.start : a.end] == a.claim.text
    roles = [f.role for f in claims[1].claim.figures]
    assert roles == ["result", "input", "input"]
