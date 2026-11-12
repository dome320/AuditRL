"""Render a filing's XBRL facts as a source document the auditor and agent can both read."""

from auditrl.benchmark.edgar import Filing
from auditrl.benchmark.metrics import METRICS


def doc_name(ticker: str, fiscal_year: int) -> str:
    return f"{ticker}_10-K_FY{fiscal_year}.md"


def doc_title(company: str, fiscal_year: int) -> str:
    return f"{company} Form 10-K Fiscal {fiscal_year}"


def _millions(value: float) -> str:
    m = round(value / 1e6)
    return f"({abs(m):,})" if m < 0 else f"{m:,}"


def _per_share(value: float) -> str:
    return f"$ ({abs(value):.2f})" if value < 0 else f"$ {value:.2f}"


def render_filing(company: str, ticker: str, cik: int, filing: Filing) -> str:
    fy = filing.fiscal_year
    years = [fy, fy - 1]
    head = f"| | {' | '.join(f'Fiscal {y}' for y in years)} |\n|---|{'---|' * len(years)}\n"

    money_rows, share_rows = [], []
    for metric in METRICS:
        cells = []
        for y in years:
            fact = filing.value(metric.key, y)
            if fact is None:
                cells.append("—")
            else:
                cells.append(_per_share(fact.value) if metric.per_share else _millions(fact.value))
        if all(c == "—" for c in cells):
            continue
        row = f"| {metric.label} | {' | '.join(cells)} |\n"
        (share_rows if metric.per_share else money_rows).append(row)

    parts = [
        f"# {doc_title(company, fy)}",
        f"Source: SEC EDGAR XBRL financial data for {company} ({ticker}, CIK {cik}), "
        f"Form 10-K accession {filing.accession}, filed {filing.filed}, for the fiscal year "
        f"ended {filing.period_end}. Filing index: {filing.url(cik)}",
        "Consolidated financial data (in millions)",
        head + "".join(money_rows),
    ]
    if share_rows:
        parts += ["Per share data (in dollars)", head + "".join(share_rows)]
    return "\n\n".join(p.rstrip("\n") for p in parts) + "\n"
