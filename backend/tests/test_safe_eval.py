import pytest

from auditrl.verify.safe_eval import UnsafeExpression, evaluate, evaluate_interval


def test_evaluate_growth():
    assert evaluate("(b - a) / a * 100", {"a": 100.0, "b": 109.1}) == pytest.approx(9.1)


def test_caret_is_power():
    assert evaluate("a ^ 2", {"a": 3.0}) == 9.0


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('true')",
        "a.__class__",
        "[a for a in range(3)]",
        "open('x')",
        "a if a else b",
        "lambda: 1",
        "c + 1",
    ],
)
def test_rejects_unsafe_or_unknown(expr):
    with pytest.raises((UnsafeExpression, SyntaxError)):
        evaluate(expr, {"a": 1.0, "b": 2.0})


def test_rejects_huge_exponent():
    with pytest.raises(UnsafeExpression):
        evaluate("a ** 1000", {"a": 2.0})


def test_interval_propagates_rounding():
    lo, hi, point = evaluate_interval(
        "(b - a) / a * 100", {"a": (78.05, 78.15), "b": (85.15, 85.25)}
    )
    assert lo < point < hi
    assert lo == pytest.approx((85.15 - 78.15) / 78.15 * 100)
    assert hi == pytest.approx((85.25 - 78.05) / 78.05 * 100)
