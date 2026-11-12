import pytest

from auditrl.verify.normalize import (
    detect_scale,
    extract_periods,
    find_quantities,
    format_like,
    parse_quantity,
    unit_error_factor,
    values_match,
)


@pytest.mark.parametrize(
    ("text", "value", "unit", "decimals"),
    [
        ("$383.3 billion", 383.3e9, "currency", 1),
        ("$1.2B", 1.2e9, "currency", 1),
        ("USD 5bn", 5e9, "currency", 0),
        ("$1,200 million", 1.2e9, "currency", 0),
        ("€45m", 45e6, "currency", 0),
        ("(2,722)", -2722, "number", 0),
        ("-5.5%", -5.5, "percent", 1),
        ("29.8 percent", 29.8, "percent", 1),
        ("150 bps", 150, "bps", 0),
        ("2.1x", 2.1, "ratio", 1),
        ("12,345", 12345, "number", 0),
    ],
)
def test_parse_quantity(text, value, unit, decimals):
    q = parse_quantity(text)
    assert q is not None
    assert q.value == pytest.approx(value)
    assert q.unit == unit
    assert q.decimals == decimals


def test_years_and_period_labels_are_not_quantities():
    qs = find_quantities("In fiscal 2023 and Q3 FY2024 revenue was $10 million.")
    assert [q.raw for q in qs] == ["$10 million"]


def test_spaced_single_letter_is_not_a_scale():
    q = parse_quantity("$5 m")
    assert q is not None and q.value == 5


def test_default_scale_applies_to_bare_table_values():
    assert parse_quantity("383,285", default_scale=1e6).value == pytest.approx(383.285e9)
    assert parse_quantity("$1.2 billion", default_scale=1e6).value == pytest.approx(1.2e9)


@pytest.mark.parametrize(
    ("claimed", "actual", "match"),
    [
        ("$383.3 billion", "383,285", True),  # rounding of the stated figure
        ("$383.2 billion", "383,285", False),
        ("$385.0 billion", "383,285", False),
        ("$97.0 billion", "96,995", True),
        ("$99.8 billion", "96,995", False),
    ],
)
def test_values_match_uses_stated_precision(claimed, actual, match):
    assert values_match(parse_quantity(claimed), parse_quantity(actual, 1e6)) is match


def test_percent_and_currency_are_not_comparable():
    assert not values_match(parse_quantity("5%"), parse_quantity("$5"))


def test_unit_error_factor():
    claimed = parse_quantity("$29.8 million")
    actual = parse_quantity("29,760", 1e6)
    assert unit_error_factor(claimed, actual) == pytest.approx(1e-3)
    assert unit_error_factor(parse_quantity("$23.8 billion"), actual) is None


@pytest.mark.parametrize(
    ("template", "value", "expected", "sig"),
    [
        ("$78.1 billion", 85.2e9, "$85.2 billion", None),
        ("9.1%", 12.345, "12.3%", None),
        ("USD 5bn", 7e9, "USD 7bn", None),
        ("$1,200 million", 1.35e9, "$1,350 million", None),
        ("$114,301 billion", 114.301e9, "$114.3 billion", 4),
        ("-2.8%", -3.04, "-3.0%", None),
    ],
)
def test_format_like(template, value, expected, sig):
    assert format_like(parse_quantity(template), value, sig_digits=sig) == expected


def test_extract_periods():
    assert extract_periods("fiscal 2023") == {"2023"}
    assert extract_periods("FY23") == {"2023"}
    assert "Q3-2024" in extract_periods("third quarter of 2024")
    assert extract_periods("Year Ended September 30, 2023") == {"2023"}


def test_detect_scale():
    assert detect_scale("(in millions, except per share)") == 1e6
    assert detect_scale("Amounts in thousands") == 1e3
    assert detect_scale("no hint") is None
