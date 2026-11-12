"""SEC EDGAR XBRL company facts: fetching (cached, rate limited) and annual fact extraction."""

import json
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from auditrl.benchmark.metrics import METRICS

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/"
MIN_INTERVAL_S = 0.15  # SEC asks for at most 10 requests per second

Fetcher = Callable[[str, str], bytes]


def _http_get(url: str, user_agent: str) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": user_agent, "Accept-Encoding": "identity"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


class EdgarClient:
    def __init__(self, cache_dir: Path, user_agent: str, fetch: Fetcher | None = None) -> None:
        if "@" not in user_agent:
            raise ValueError(
                "SEC requires a User-Agent with contact info, e.g. "
                "'AuditRL research you@example.com' (set AUDITRL_SEC_USER_AGENT)"
            )
        self.cache_dir = cache_dir
        self.user_agent = user_agent
        self.fetch = fetch or _http_get
        self._last_request = 0.0

    def _get_json(self, url: str, cache_name: str) -> Any:
        path = self.cache_dir / cache_name
        if path.exists():
            return json.loads(path.read_text())
        wait = MIN_INTERVAL_S - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        data = self.fetch(url, self.user_agent)
        self._last_request = time.monotonic()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return json.loads(data)

    def tickers(self) -> dict[str, tuple[int, str]]:
        raw = self._get_json(TICKERS_URL, "company_tickers.json")
        return {v["ticker"].upper(): (int(v["cik_str"]), v["title"]) for v in raw.values()}

    def company_facts(self, cik: int) -> dict[str, Any]:
        return self._get_json(FACTS_URL.format(cik=cik), f"companyfacts/CIK{cik:010d}.json")


@dataclass(frozen=True)
class Fact:
    metric: str
    concept: str
    fiscal_year: int
    value: float
    unit: str
    start: str
    end: str
    accession: str
    filed: str


@dataclass
class Filing:
    """One 10-K as filed: its own fiscal year plus the comparative years it reports."""

    accession: str
    fiscal_year: int
    filed: str
    period_end: str
    facts: dict[str, dict[int, Fact]] = field(default_factory=dict)

    def value(self, metric: str, fiscal_year: int) -> Fact | None:
        return self.facts.get(metric, {}).get(fiscal_year)

    def url(self, cik: int) -> str:
        return FILING_URL.format(cik=cik, accn=self.accession.replace("-", ""))


def _annual(row: dict[str, Any]) -> bool:
    if "start" not in row or row.get("form") != "10-K":
        return False
    days = (date.fromisoformat(row["end"]) - date.fromisoformat(row["start"])).days
    return 340 <= days <= 380


def _fiscal_year(end: str, current_end: str, filing_fy: int) -> int:
    years_back = round((date.fromisoformat(current_end) - date.fromisoformat(end)).days / 365.25)
    return filing_fy - years_back


def extract_filings(companyfacts: dict[str, Any]) -> list[Filing]:
    """Group annual 10-K facts by filing; each fact is labeled with the fiscal year it covers.

    The `fy` field on a fact is the filing's fiscal year, not the period's, so comparative
    years are derived from period end dates relative to the filing's latest period.
    """
    gaap = companyfacts.get("facts", {}).get("us-gaap", {})
    rows_by_accn: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
    for metric in METRICS:
        for concept in metric.concepts:
            units = gaap.get(concept, {}).get("units", {})
            for unit, rows in units.items():
                for row in rows:
                    if _annual(row) and row.get("fp") == "FY":
                        rows_by_accn.setdefault(row["accn"], []).append(
                            (metric.key, concept, unit, row)
                        )  # type: ignore[arg-type]

    # Keep the original 10-K per period end (amendments and re-filings repeat periods).
    originals: dict[str, tuple[str, str, list[Any]]] = {}
    for accn, items in rows_by_accn.items():
        current_end = max(r["end"] for *_, r in items)
        filed = max(r["filed"] for *_, r in items)
        if current_end not in originals or filed < originals[current_end][1]:
            originals[current_end] = (accn, filed, items)

    filings: list[Filing] = []
    prev_fy: int | None = None
    for current_end in sorted(originals):
        accn, filed, items = originals[current_end]
        filing_fy = max(int(r["fy"]) for *_, r in items if r["end"] == current_end)
        if prev_fy is not None and filing_fy <= prev_fy:
            filing_fy = prev_fy + 1  # filer mis-tagged `fy` (seen in practice); periods are annual
        prev_fy = filing_fy
        filing = Filing(accession=accn, fiscal_year=filing_fy, filed=filed, period_end=current_end)
        for metric in METRICS:
            for concept in metric.concepts:  # first concept with data in this filing wins
                hits = [(c, u, r) for m, c, u, r in items if m == metric.key and c == concept]
                if not hits:
                    continue
                by_year = filing.facts.setdefault(metric.key, {})
                for c, u, r in hits:
                    fy = _fiscal_year(r["end"], current_end, filing_fy)
                    by_year.setdefault(
                        fy,
                        Fact(
                            metric.key,
                            c,
                            fy,
                            float(r["val"]),
                            u,
                            r["start"],
                            r["end"],
                            accn,
                            r["filed"],
                        ),
                    )
                break
        filings.append(filing)
    return filings
