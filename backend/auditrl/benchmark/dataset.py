"""Benchmark layout on disk.

<bench>/tasks.jsonl                      one research task per (company, fiscal year)
<bench>/sources/<TICKER>/facts.json      XBRL facts snapshot (labeler input, provenance)
<bench>/sources/<TICKER>/<doc>.md        rendered filings shared by agent and auditor
<bench>/cases/<task_id>/report.md        agent-written report
<bench>/cases/<task_id>/trace.json       full agent trace
<bench>/cases/<task_id>/sources/         the exact documents the agent saw
<bench>/cases/<task_id>/expected.json    ground-truth labels
<bench>/splits.json                      frozen split manifest
<bench>/runs/*.json                      evaluation run records
"""

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from auditrl.benchmark.agent import ResearchAgent, Task, Trace
from auditrl.benchmark.edgar import EdgarClient, Fact, Filing, extract_filings
from auditrl.benchmark.labeler import FactBook, Labeler, labels_to_expected
from auditrl.benchmark.sources import doc_name, doc_title, render_filing


def save_filings(path: Path, filings: list[Filing]) -> None:
    payload = [
        {
            **{k: v for k, v in asdict(f).items() if k != "facts"},
            "facts": [asdict(fact) for by_year in f.facts.values() for fact in by_year.values()],
        }
        for f in filings
    ]
    path.write_text(json.dumps(payload, indent=1) + "\n")


def load_filings(path: Path) -> list[Filing]:
    filings = []
    for raw in json.loads(path.read_text()):
        f = Filing(raw["accession"], raw["fiscal_year"], raw["filed"], raw["period_end"])
        for fact in raw["facts"]:
            f.facts.setdefault(fact["metric"], {})[fact["fiscal_year"]] = Fact(**fact)
        filings.append(f)
    return filings


def build_tasks(
    bench: Path, client: EdgarClient, tickers: list[str], years_per_company: int = 2
) -> list[Task]:
    """Fetch filings and write sources for the latest fiscal years of each company.

    Each task gets the 10-K for its fiscal year and the prior year's 10-K, so citing the
    wrong filing is possible.
    """
    known = client.tickers()
    tasks: list[Task] = []
    for ticker in tickers:
        ticker = ticker.upper()
        if ticker not in known:
            raise ValueError(f"unknown ticker {ticker}")
        cik, title = known[ticker]
        company = _company_name(client.company_facts(cik).get("entityName") or title)
        filings = {f.fiscal_year: f for f in extract_filings(client.company_facts(cik))}
        years = [
            y
            for y in sorted(filings, reverse=True)
            if y - 1 in filings and filings[y].value("revenue", y)
        ]
        years = years[:years_per_company]
        if not years:
            continue
        src_dir = bench / "sources" / ticker
        src_dir.mkdir(parents=True, exist_ok=True)
        used = sorted({y for fy in years for y in (fy, fy - 1)})
        save_filings(src_dir / "facts.json", [filings[y] for y in used])
        for y in used:
            (src_dir / doc_name(ticker, y)).write_text(
                render_filing(company, ticker, cik, filings[y])
            )
        for fy in years:
            tasks.append(
                Task(
                    task_id=f"{ticker.lower()}_fy{fy}",
                    ticker=ticker,
                    cik=cik,
                    company=company,
                    fiscal_year=fy,
                    sources={doc_name(ticker, y): doc_title(company, y) for y in (fy, fy - 1)},
                )
            )
    existing = {t.task_id: t for t in load_tasks(bench)}
    existing.update({t.task_id: t for t in tasks})
    (bench / "tasks.jsonl").write_text(
        "".join(
            json.dumps(asdict(t)) + "\n" for t in sorted(existing.values(), key=lambda t: t.task_id)
        )
    )
    return tasks


def _company_name(raw: str) -> str:
    name = raw.strip()
    return name.title() if name.isupper() else name


def load_tasks(bench: Path) -> list[Task]:
    path = bench / "tasks.jsonl"
    if not path.exists():
        return []
    return [
        Task.from_dict(json.loads(line)) for line in path.read_text().splitlines() if line.strip()
    ]


def task_documents(bench: Path, task: Task) -> dict[str, str]:
    """Document title -> text for the agent."""
    return {
        title: (bench / "sources" / task.ticker / name).read_text()
        for name, title in task.sources.items()
    }


def generate_case(
    bench: Path, task: Task, agent: ResearchAgent, force: bool = False
) -> Trace | None:
    case_dir = bench / "cases" / task.task_id
    if (case_dir / "report.md").exists() and not force:
        return None
    trace = agent.run(task, task_documents(bench, task))
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "trace.json").write_text(json.dumps(trace.to_dict(), indent=1) + "\n")
    if trace.error:
        return trace
    (case_dir / "sources").mkdir(exist_ok=True)
    for name in task.sources:
        shutil.copyfile(bench / "sources" / task.ticker / name, case_dir / "sources" / name)
    (case_dir / "report.md").write_text(trace.report + "\n")
    (case_dir / "task.json").write_text(json.dumps(asdict(task), indent=1) + "\n")
    return trace


def label_case(bench: Path, case_dir: Path) -> dict[str, Any]:
    task = Task.from_dict(json.loads((case_dir / "task.json").read_text()))
    filings = [
        f
        for f in load_filings(bench / "sources" / task.ticker / "facts.json")
        if f.fiscal_year in (task.fiscal_year, task.fiscal_year - 1)
    ]
    titles = {doc_title(task.company, f.fiscal_year): f.fiscal_year for f in filings}
    labels = Labeler(FactBook(filings, titles)).label((case_dir / "report.md").read_text())
    expected = labels_to_expected(task.task_id, labels)
    (case_dir / "expected.json").write_text(json.dumps(expected, indent=1) + "\n")
    return expected


def case_dirs(bench: Path) -> list[Path]:
    root = bench / "cases"
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if (p / "report.md").exists())
