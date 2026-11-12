"""Parsing, comparison and formatting of financial quantities and fiscal periods."""

import re
from dataclasses import dataclass, replace
from typing import Literal

from rapidfuzz import fuzz, utils

Unit = Literal["currency", "percent", "bps", "ratio", "number"]

SCALE_WORDS: dict[str, float] = {
    "thousand": 1e3,
    "thousands": 1e3,
    "k": 1e3,
    "million": 1e6,
    "millions": 1e6,
    "mn": 1e6,
    "mm": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "billions": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "trillion": 1e12,
    "trillions": 1e12,
    "tn": 1e12,
    "t": 1e12,
}

_CURRENCY = r"(?P<cur>US\$|USD\s?|\$|€|£|¥)?"
_NUMBER = r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
_SCALE = (
    r"(?:(?P<sep>\s?)(?P<scale>thousands?|millions?|billions?|trillions?|mn|mm|bn|tn|k|m|b|t)\b)?"
)
_UNIT = r"(?P<unit>\s?%|\s?percent(?:age points?)?|\s?pct\b|\s?bps\b|\s?basis points?|x\b)?"
QUANTITY_RE = re.compile(
    r"(?<![\w.])(?P<neg>[-−]|\()?" + _CURRENCY + _NUMBER + r"(?P<close>\))?" + _SCALE + _UNIT,
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: Unit
    magnitude: float
    scale: float
    decimals: int
    raw: str
    start: int = 0
    end: int = 0
    currency: str | None = None
    scale_token: str = ""
    scale_sep: str = ""
    unit_token: str = ""

    @property
    def half_ulp(self) -> float:
        """Half a unit in the last stated digit, in base units: the rounding slack of the text."""
        return 0.5 * 10 ** (-self.decimals) * self.scale

    @property
    def interval(self) -> tuple[float, float]:
        return (self.value - self.half_ulp, self.value + self.half_ulp)


def _is_year(num: str, has_cur: bool, has_unit: bool, has_scale: bool) -> bool:
    if has_cur or has_unit or has_scale or "," in num or "." in num:
        return False
    return len(num) == 4 and 1900 <= int(num) <= 2100


def find_quantities(text: str, default_scale: float = 1.0) -> list[Quantity]:
    out: list[Quantity] = []
    for m in QUANTITY_RE.finditer(text):
        num, cur = m.group("num"), m.group("cur")
        scale_tok, unit_tok = m.group("scale") or "", (m.group("unit") or "").strip()
        # A bare single-letter scale ("m", "b") must be glued to the number to count.
        if scale_tok and len(scale_tok) == 1 and m.group("sep"):
            scale_tok = ""
        if _is_year(num, bool(cur), bool(unit_tok), bool(scale_tok)):
            continue
        neg, close = m.group("neg"), m.group("close")
        start = m.start()
        if neg == "(" and not close:
            neg, start = None, m.start() + 1
        magnitude = float(num.replace(",", ""))
        decimals = len(num.split(".")[1]) if "." in num else 0
        sign = -1.0 if neg else 1.0
        lower_unit = unit_tok.lower()
        unit: Unit
        if lower_unit in ("%", "pct") or lower_unit.startswith("percent"):
            unit = "percent"
        elif lower_unit in ("bps",) or lower_unit.startswith("basis"):
            unit = "bps"
        elif lower_unit == "x":
            unit = "ratio"
        elif cur:
            unit = "currency"
        else:
            unit = "number"
        if unit in ("percent", "bps", "ratio"):
            scale = 1.0
            scale_tok = ""
        else:
            scale = SCALE_WORDS[scale_tok.lower()] if scale_tok else default_scale
        if unit_tok:
            end = m.end("unit")
        elif scale_tok:
            end = m.end("scale")
        else:
            end = m.end("close") if close else m.end("num")
        raw = text[start:end].strip()
        out.append(
            Quantity(
                value=sign * magnitude * scale,
                unit=unit,
                magnitude=sign * magnitude,
                scale=scale,
                decimals=decimals,
                raw=raw,
                start=start,
                end=end,
                currency=cur or None,
                scale_token=scale_tok,
                scale_sep=(m.group("sep") or "") if scale_tok else "",
                unit_token=m.group("unit") or "",
            )
        )
    return out


def parse_quantity(text: str, default_scale: float = 1.0) -> Quantity | None:
    found = find_quantities(text, default_scale)
    return found[0] if found else None


def comparable(a: Quantity, b: Quantity) -> bool:
    if a.unit == b.unit:
        return True
    loose = {"currency", "number"}
    return a.unit in loose and b.unit in loose


def intervals_overlap(a: tuple[float, float], b: tuple[float, float], slack: float = 0.0) -> bool:
    return a[0] - slack <= b[1] and b[0] - slack <= a[1]


def values_match(claimed: Quantity, actual: Quantity, rel_tol: float = 1e-6) -> bool:
    if not comparable(claimed, actual):
        return False
    slack = rel_tol * max(abs(claimed.value), abs(actual.value))
    return intervals_overlap(claimed.interval, actual.interval, slack)


def value_matches_interval(
    claimed: Quantity, computed: tuple[float, float], rel_tol: float = 1e-6
) -> bool:
    slack = rel_tol * max(abs(claimed.value), abs(computed[0]), abs(computed[1]))
    return intervals_overlap(claimed.interval, computed, slack)


def unit_error_factor(claimed: Quantity, actual: Quantity, rel_tol: float = 1e-6) -> float | None:
    """Return the factor (e.g. 1e3) if the claim matches the actual value up to a scale slip."""
    if not comparable(claimed, actual) or actual.value == 0:
        return None
    for factor in (1e3, 1e-3, 1e6, 1e-6):
        shifted = replace(actual, value=actual.value * factor, scale=actual.scale * factor)
        if values_match(claimed, shifted, rel_tol):
            return factor
    return None


def format_like(template: Quantity, value: float, sig_digits: int | None = None) -> str:
    """Render `value` (base units) in the same style as `template` as written in a report."""
    if template.unit in ("percent", "bps", "ratio"):
        magnitude = value
    else:
        magnitude = value / template.scale
    neg = magnitude < 0
    decimals = template.decimals
    if sig_digits is not None and magnitude:
        int_digits = len(str(int(abs(magnitude)))) if abs(magnitude) >= 1 else 0
        decimals = max(decimals, sig_digits - int_digits) if int_digits < sig_digits else 0
    body = f"{abs(magnitude):,.{decimals}f}"
    if "," not in template.raw and abs(template.magnitude) >= 1000:
        body = body.replace(",", "")
    out = f"{template.currency or ''}{body}"
    if template.scale_token:
        out += f"{template.scale_sep}{template.scale_token}"
    out += template.unit_token
    return f"-{out}" if neg else out


_SCALE_HINT_RE = re.compile(
    r"in\s+(thousands|millions|billions)|\((?:\$|usd)?\s*(?:in\s+)?(thousands|millions|billions)",
    re.IGNORECASE,
)


def detect_scale(text: str) -> float | None:
    m = _SCALE_HINT_RE.search(text)
    if not m:
        return None
    word = (m.group(1) or m.group(2)).lower()
    return SCALE_WORDS[word]


_QUARTER_RE = re.compile(r"\b(?:Q([1-4])|([1-4])Q|(first|second|third|fourth) quarter)\b", re.I)
_YEAR_RE = re.compile(r"(?:\bFY\s?'?|\bfiscal(?: year)?\s+)?(?<!\d)((?:19|20)\d{2})(?!\d)", re.I)
_SHORT_FY_RE = re.compile(r"\bFY\s?'?(\d{2})\b", re.I)
_ORDINALS = {"first": "1", "second": "2", "third": "3", "fourth": "4"}


def extract_periods(text: str) -> set[str]:
    """Normalized period keys like {'2023', 'Q3-2023'}."""
    years = {m.group(1) for m in _YEAR_RE.finditer(text)}
    years |= {f"20{m.group(1)}" for m in _SHORT_FY_RE.finditer(text)}
    keys = set(years)
    for qm in _QUARTER_RE.finditer(text):
        q = qm.group(1) or qm.group(2) or _ORDINALS[qm.group(3).lower()]
        keys |= {f"Q{q}-{y}" for y in years}
    return keys


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def label_score(metric: str, label: str, threshold: float = 70) -> float | None:
    """Row-label match score; token_sort breaks ties toward the closest (least extra) label."""
    m, lbl = utils.default_process(metric), utils.default_process(label)
    base = fuzz.token_set_ratio(m, lbl)
    if base < threshold:
        return None
    return (base + fuzz.token_sort_ratio(m, lbl)) / 2
