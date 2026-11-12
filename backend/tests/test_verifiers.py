import pytest

from auditrl.findings import apply_replacements
from auditrl.findings.aggregate import check_claim, merge_verdict
from auditrl.llm import ExtractedClaim, ExtractedFigure
from auditrl.models import ErrorType, Verdict
from auditrl.verify.arithmetic import verify_arithmetic
from auditrl.verify.citation import verify_citation
from auditrl.verify.figure import verify_figure


def fig(value: str, metric: str, period: str | None, role: str = "standalone") -> ExtractedFigure:
    return ExtractedFigure(value_text=value, metric=metric, period=period, role=role)


@pytest.mark.parametrize(
    ("claim", "figure", "verdict", "error_type", "fixed"),
    [
        (
            "Total net sales were $383.3 billion in fiscal 2023.",
            fig("$383.3 billion", "Total net sales", "fiscal 2023"),
            Verdict.valid,
            None,
            None,
        ),
        (
            "Total net sales were $388.3 billion in fiscal 2023.",
            fig("$388.3 billion", "Total net sales", "fiscal 2023"),
            Verdict.invalid,
            ErrorType.misstated_figure,
            "$383.3 billion",
        ),
        (
            "Total net sales were $394.3 billion in fiscal 2023.",
            fig("$394.3 billion", "Total net sales", "fiscal 2023"),
            Verdict.invalid,
            ErrorType.wrong_period,
            "$383.3 billion",
        ),
        (
            "Operating income was $114.3 million in fiscal 2023.",
            fig("$114.3 million", "Operating income", "fiscal 2023"),
            Verdict.invalid,
            ErrorType.unit_error,
            "$114,301 million",
        ),
        (
            "The company had a net loss of $2.7 billion in fiscal 2022.",
            fig("$2.7 billion", "net loss", "fiscal 2022"),
            Verdict.valid,
            None,
            None,
        ),
        (
            "Free cash flow was $99.6 billion in fiscal 2023.",
            fig("$99.6 billion", "Free cash flow", "fiscal 2023"),
            Verdict.uncertain,
            None,
            None,
        ),
    ],
)
def test_verify_figure(index, llm, claim, figure, verdict, error_type, fixed):
    result = verify_figure(claim, figure, index, llm)
    assert result.verdict == verdict, result.facts
    assert result.error_type == error_type
    if fixed:
        assert result.replacements[0][1] == fixed
        assert result.evidence, "errors must link evidence"


def test_figure_prefers_closest_row_label(index, llm):
    # "net sales" must match "Total net sales", not "Services net sales".
    result = verify_figure(
        "Net sales were $383.3 billion in fiscal 2023.",
        fig("$383.3 billion", "net sales", "fiscal 2023"),
        index,
        llm,
    )
    assert result.verdict == Verdict.valid


def test_figure_value_must_appear_in_claim(index, llm):
    result = verify_figure("Sales were strong.", fig("$1 billion", "sales", "2023"), index, llm)
    assert result.verdict == Verdict.uncertain


@pytest.mark.parametrize(
    ("claim", "figures", "verdict", "fixed"),
    [
        (
            "Services grew 9.1% from $78.1 billion in fiscal 2022 to $85.2 billion in fiscal 2023.",
            [
                fig("9.1%", "Services", None, "result"),
                fig("$78.1 billion", "Services net sales", "fiscal 2022", "input"),
                fig("$85.2 billion", "Services net sales", "fiscal 2023", "input"),
            ],
            Verdict.valid,
            None,
        ),
        (
            "Services grew 12.0% from $78.1 billion in fiscal 2022 to $85.2 billion in fiscal 2023.",
            [
                fig("12.0%", "Services", None, "result"),
                fig("$78.1 billion", "Services net sales", "fiscal 2022", "input"),
                fig("$85.2 billion", "Services net sales", "fiscal 2023", "input"),
            ],
            Verdict.invalid,
            "9.1%",
        ),
        (
            "Total net sales declined 2.8% from $394.3 billion in fiscal 2022 to $383.3 billion in fiscal 2023.",
            [
                fig("2.8%", "Total net sales", None, "result"),
                fig("$394.3 billion", "Total net sales", "fiscal 2022", "input"),
                fig("$383.3 billion", "Total net sales", "fiscal 2023", "input"),
            ],
            Verdict.valid,
            None,
        ),
        (
            "Operating margin was 31.0% in fiscal 2023, with operating income of $114.3 billion on net sales of $383.3 billion.",
            [
                fig("31.0%", "Operating margin", "fiscal 2023", "result"),
                fig("$114.3 billion", "operating income", "fiscal 2023", "input"),
                fig("$383.3 billion", "net sales", "fiscal 2023", "input"),
            ],
            Verdict.invalid,
            "29.8%",
        ),
    ],
)
def test_verify_arithmetic(index, llm, claim, figures, verdict, fixed):
    result = verify_arithmetic(claim, figures, index, llm)
    assert result.verdict == verdict, result.facts
    if fixed:
        assert result.error_type == ErrorType.arithmetic_error
        assert result.replacements[0][1] == fixed


def test_arithmetic_with_prior_period_from_evidence(index, llm):
    claim = "Total net sales declined 2.8% to $383.3 billion in fiscal 2023."
    result = verify_arithmetic(
        claim,
        [
            fig("2.8%", "Total net sales", None, "result"),
            fig("$383.3 billion", "Total net sales", "fiscal 2023", "input"),
        ],
        index,
        llm,
    )
    assert result.verdict == Verdict.valid
    assert result.evidence, "operand drawn from the filing must be cited"


def test_citation_valid_and_mismatch(index, llm):
    ok = verify_citation(
        "Total net sales were $383.3 billion in fiscal 2023 (Source: Apple 10-K FY2023).",
        "Apple 10-K FY2023",
        [fig("$383.3 billion", "Total net sales", "fiscal 2023")],
        index,
        llm,
    )
    assert [r.verdict for r in ok] == [Verdict.valid]

    claim = "Total net sales were $365.8 billion in fiscal 2021 (Source: Apple 10-K FY2023)."
    bad = verify_citation(
        claim,
        "Apple 10-K FY2023",
        [fig("$365.8 billion", "Total net sales", "fiscal 2021")],
        index,
        llm,
    )
    assert bad[0].verdict == Verdict.invalid
    assert bad[0].error_type == ErrorType.citation_mismatch
    assert "Apple 10-K FY2021" in apply_replacements(claim, bad[0].replacements)


def test_citation_to_missing_source_is_uncertain(index, llm):
    out = verify_citation(
        "Sales were $383.3 billion (Source: Goldman note).",
        "Goldman note",
        [fig("$383.3 billion", "sales", None)],
        index,
        llm,
    )
    assert out[0].verdict == Verdict.uncertain


def test_merge_verdict_rules(index, llm):
    claim = ExtractedClaim(
        text="Revenue grew 9.1% from $78.1 billion in 2022 to $85.2 billion in 2023.",
        claim_type="arithmetic",
        checkable=True,
        figures=[
            fig("9.1%", "Revenue", None, "result"),
            fig("$78.1 billion", "Unknown metric", "2022", "input"),
            fig("$85.2 billion", "Unknown metric", "2023", "input"),
        ],
    )
    checks = check_claim(claim, index, llm)
    # Inputs without evidence are informational; the verified computation decides.
    assert merge_verdict(checks)[0] == Verdict.valid
