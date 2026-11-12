"""Phase 2 benchmark: EDGAR extraction, labeler, agent, dataset/splits, harness, feedback."""

import json
from pathlib import Path

import pytest

from auditrl.benchmark.agent import AssistantTurn, ResearchAgent, Task
from auditrl.benchmark.dataset import (
    build_tasks,
    case_dirs,
    generate_case,
    label_case,
    load_filings,
    load_tasks,
    save_filings,
)
from auditrl.benchmark.edgar import EdgarClient, extract_filings
from auditrl.benchmark.harness import compare, load_runs, run_eval
from auditrl.benchmark.labeler import FactBook, Labeler, labels_to_expected, split_sentences
from auditrl.benchmark.sources import doc_title, render_filing
from auditrl.benchmark.splits import IntegrityError, assign_split, freeze, load_manifest
from auditrl.evaluation import score_case
from auditrl.intake import parse_text
from auditrl.review.feedback import export_feedback

CIK = 1234
COMPANY = "Acme Corp"
TITLE_25, TITLE_24 = doc_title(COMPANY, 2025), doc_title(COMPANY, 2024)

# fiscal year -> metric concept -> value (USD)
VALUES = {
    2023: {"Revenues": 90e9, "GrossProfit": 34e9, "OperatingIncomeLoss": 18e9, "NetIncomeLoss": 15e9,
           "EarningsPerShareDiluted": 3.0, "ResearchAndDevelopmentExpense": 8e9,
           "PaymentsToAcquirePropertyPlantAndEquipment": 4e9},
    2024: {"Revenues": 100e9, "GrossProfit": 40e9, "OperatingIncomeLoss": 22e9, "NetIncomeLoss": 18e9,
           "EarningsPerShareDiluted": 3.6, "ResearchAndDevelopmentExpense": 9e9,
           "PaymentsToAcquirePropertyPlantAndEquipment": 5e9},
    2025: {"Revenues": 120e9, "GrossProfit": 50e9, "OperatingIncomeLoss": 30e9, "NetIncomeLoss": 24e9,
           "EarningsPerShareDiluted": 4.8, "ResearchAndDevelopmentExpense": 10e9,
           "PaymentsToAcquirePropertyPlantAndEquipment": 6e9},
}  # fmt: skip
PERIOD = {2023: ("2022-10-01", "2023-09-30"), 2024: ("2023-10-01", "2024-09-30"),
          2025: ("2024-10-01", "2025-09-30")}  # fmt: skip


def _companyfacts() -> dict:
    """10-Ks for FY2024 and FY2025 (the latter mis-tagged fy=2024), plus a later amendment."""
    filings = [
        ("0001-24-000001", "2024-11-01", 2024, [2024, 2023], 1.0),
        ("0001-25-000001", "2025-11-01", 2024, [2025, 2024], 1.0),
        ("0001-25-000099", "2025-12-15", 2025, [2025, 2024], 1.1),  # 10-K/A style re-file
    ]
    gaap: dict = {}
    for accn, filed, fy_tag, years, scale in filings:
        for year in years:
            start, end = PERIOD[year]
            for concept, val in VALUES[year].items():
                unit = "USD/shares" if concept.startswith("Earnings") else "USD"
                gaap.setdefault(concept, {"units": {}})["units"].setdefault(unit, []).append(
                    {
                        "start": start,
                        "end": end,
                        "val": val * scale,
                        "accn": accn,
                        "fy": fy_tag,
                        "fp": "FY",
                        "form": "10-K",
                        "filed": filed,
                    }  # fmt: skip
                )
    # A quarterly row must be ignored.
    gaap["Revenues"]["units"]["USD"].append(
        {
            "start": "2025-07-01",
            "end": "2025-09-30",
            "val": 1.0,
            "accn": "q",
            "fy": 2025,
            "fp": "Q4",
            "form": "10-Q",
            "filed": "2025-11-01",
        }  # fmt: skip
    )
    return {"entityName": "ACME CORP", "facts": {"us-gaap": gaap}}


@pytest.fixture
def filings():
    return {f.fiscal_year: f for f in extract_filings(_companyfacts())}


@pytest.fixture
def book(filings):
    return FactBook([filings[2025], filings[2024]], {TITLE_25: 2025, TITLE_24: 2024})


# --- EDGAR ------------------------------------------------------------------


def test_extract_filings_fixes_fiscal_years_and_ignores_amendments(filings):
    assert sorted(filings) == [2024, 2025]
    f25 = filings[2025]
    assert f25.accession == "0001-25-000001"
    assert f25.value("revenue", 2025).value == 120e9
    assert f25.value("revenue", 2024).value == 100e9
    assert f25.value("eps_diluted", 2025).value == 4.8
    assert filings[2024].value("net_income", 2023).value == 15e9


def test_edgar_client_requires_contact_and_caches(tmp_path):
    with pytest.raises(ValueError):
        EdgarClient(tmp_path, "no contact")
    calls = []

    def fetch(url, ua):
        calls.append(url)
        if url.endswith("company_tickers.json"):
            return json.dumps(
                {"0": {"cik_str": CIK, "ticker": "acme", "title": "ACME CORP"}}
            ).encode()
        return json.dumps(_companyfacts()).encode()

    client = EdgarClient(tmp_path, "test you@example.com", fetch=fetch)
    assert client.tickers() == {"ACME": (CIK, "ACME CORP")}
    client.company_facts(CIK)
    client.company_facts(CIK)
    assert len(calls) == 2
    assert (tmp_path / "companyfacts" / f"CIK{CIK:010d}.json").exists()


def test_render_filing_round_trips_through_parser(filings):
    text = render_filing(COMPANY, "ACME", CIK, filings[2025])
    doc = parse_text("ACME_10-K_FY2025.md", text)
    rows = {p.row_label: p for p in doc.passages if p.row_label}
    assert rows["Total revenue"].cells == ["120,000", "100,000"]
    assert rows["Total revenue"].unit_scale == 1e6
    assert rows["Diluted earnings per share"].cells == ["$ 4.80", "$ 3.60"]
    assert "0001-25-000001" in text


def test_filings_snapshot_round_trip(tmp_path, filings):
    save_filings(tmp_path / "facts.json", list(filings.values()))
    loaded = {f.fiscal_year: f for f in load_filings(tmp_path / "facts.json")}
    assert loaded[2025].value("capex", 2024) == filings[2025].value("capex", 2024)


# --- Labeler ----------------------------------------------------------------

REPORT = (
    f"Acme Corp grew revenue to $120.0 billion in fiscal 2025, up 20.0% from $100.0 billion "
    f"in fiscal 2024 (Source: {TITLE_25}). Gross margin was 41.7% in fiscal 2025 and operating "
    f"margin was 27.0%. Net income was $24.0 billion in fiscal 2025 (Source: {TITLE_24}), and "
    f"diluted EPS was $4.80. Research and development expense was $9.0 billion in fiscal 2025. "
    f"Capital expenditures were $6.0 million in fiscal 2025. Net income was $21.0 billion in "
    f"fiscal 2025. We expect demand to remain healthy. Backlog reached $7.5 billion in fiscal 2025."
)


def _by_quote(labels, needle):
    return next(lbl for lbl in labels if needle in lbl.quote)


def test_labeler_labels_errors_and_abstains(book):
    labels = Labeler(book).label(REPORT)
    assert _by_quote(labels, "grew revenue").verdict == "valid"
    margin = _by_quote(labels, "Gross margin")
    assert (margin.verdict, margin.error_type, margin.correction_contains) == (
        "invalid",
        "arithmetic_error",
        "25.0%",
    )
    cited = _by_quote(labels, "Net income was $24.0 billion")
    assert (cited.verdict, cited.error_type, cited.correction_contains) == (
        "invalid",
        "citation_mismatch",
        TITLE_25,
    )
    assert next(f for f in cited.figures if f.metric == "eps_diluted").status == "correct"
    rnd = _by_quote(labels, "Research and development")
    assert (rnd.error_type, rnd.correction_contains) == ("wrong_period", "$10.0 billion")
    assert _by_quote(labels, "Capital expenditures").error_type == "unit_error"
    misstated = _by_quote(labels, "$21.0 billion")
    assert (misstated.error_type, misstated.correction_contains) == (
        "misstated_figure",
        "$24.0 billion",
    )
    assert _by_quote(labels, "Backlog").verdict == "unlabeled"
    assert not any("expect demand" in lbl.quote for lbl in labels)


def test_labels_to_expected_excludes_unlabeled(book):
    expected = labels_to_expected("acme", Labeler(book).label(REPORT))
    quotes = [lbl["quote"] for lbl in expected["labels"]]
    assert not any("Backlog" in q for q in quotes)
    assert any("Backlog" in q for q in expected["unlabeled"])
    assert expected["coverage"]["errors"] == 5


def test_split_sentences_keeps_abbreviations():
    text = "Revenue rose 5.0% vs. the prior year. Results improved. Done."
    sentences = [text[s:e] for s, e in split_sentences(text)]
    assert sentences == ["Revenue rose 5.0% vs. the prior year.", "Results improved.", "Done."]


def test_labeler_borderline_percent_is_unlabeled(book):
    # True gross margin is 41.67%; 41.8% is within the borderline band.
    (label,) = Labeler(book).label("Gross margin was 41.8% in fiscal 2025.")
    assert label.verdict == "unlabeled"


# --- Agent ------------------------------------------------------------------


def _task() -> Task:
    return Task("acme_fy2025", "ACME", CIK, COMPANY, 2025,
                {"ACME_10-K_FY2025.md": TITLE_25, "ACME_10-K_FY2024.md": TITLE_24})  # fmt: skip


class ScriptedModel:
    """Fake completion: list docs + calculate, read a filing, then write the report."""

    def __init__(self, report: str) -> None:
        self.report = report
        self.calls: list[list[dict]] = []

    def __call__(self, messages, tools):
        self.calls.append(list(messages))
        step = sum(1 for m in messages if m["role"] == "assistant")
        if step == 0 and tools:
            return AssistantTurn(None, [
                _call("1", "list_documents", {}),
                _call("2", "calculate", {"expression": "(120000 - 100000) / 100000 * 100"}),
            ], 100, 20, 0.001)  # fmt: skip
        if step == 1 and tools:
            return AssistantTurn(
                None, [_call("3", "read_document", {"title": TITLE_25})], 50, 10, 0.001
            )
        return AssistantTurn(self.report, [], 200, 300, 0.002)


def _call(cid, name, args):
    return {
        "id": cid,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def test_agent_runs_tools_and_records_trace():
    model = ScriptedModel(REPORT)
    trace = ResearchAgent("fake", complete=model).run(
        _task(), {TITLE_25: "doc25", TITLE_24: "doc24"}
    )
    assert trace.error is None and trace.report == REPORT
    results = {t["name"]: t["result"] for t in trace.tool_calls}
    assert results["calculate"] == "20.0"
    assert results["read_document"] == "doc25"
    assert TITLE_24 in results["list_documents"]
    assert trace.prompt_tokens == 350 and trace.cost_usd == pytest.approx(0.004)
    assert "fiscal 2025" in model.calls[0][1]["content"]


def test_agent_records_provider_errors():
    def boom(messages, tools):
        raise ConnectionError("no network")

    trace = ResearchAgent("fake", complete=boom).run(_task(), {TITLE_25: "x"})
    assert trace.report == "" and "ConnectionError" in trace.error


def test_agent_tool_errors_are_returned_to_model():
    assert ResearchAgent._run_tool("calculate", '{"expression": "__import__(1)"}', {}).startswith(
        "error"
    )
    assert ResearchAgent._run_tool("read_document", '{"title": "nope"}', {"A": "x"}).startswith(
        "error"
    )
    assert (
        ResearchAgent._run_tool("read_document", '{"title": "acme"}', {"Acme Corp 10-K": "x"})
        == "x"
    )


# --- Dataset, splits, harness -----------------------------------------------


@pytest.fixture
def bench(tmp_path):
    def fetch(url, ua):
        if url.endswith("company_tickers.json"):
            return json.dumps(
                {"0": {"cik_str": CIK, "ticker": "ACME", "title": "ACME CORP"}}
            ).encode()
        return json.dumps(_companyfacts()).encode()

    root = tmp_path / "bench"
    client = EdgarClient(tmp_path / "cache", "test you@example.com", fetch=fetch)
    tasks = build_tasks(root, client, ["acme"], years_per_company=1)
    assert [t.task_id for t in tasks] == ["acme_fy2025"]
    assert load_tasks(root)[0].sources == _task().sources
    agent = ResearchAgent("fake", complete=ScriptedModel(REPORT))
    generate_case(root, tasks[0], agent)
    label_case(root, root / "cases" / "acme_fy2025")
    return root


def test_dataset_case_layout(bench):
    case = bench / "cases" / "acme_fy2025"
    assert sorted(p.name for p in (case / "sources").iterdir()) == [
        "ACME_10-K_FY2024.md", "ACME_10-K_FY2025.md"]  # fmt: skip
    expected = json.loads((case / "expected.json").read_text())
    assert expected["coverage"]["errors"] == 5
    assert json.loads((case / "trace.json").read_text())["model"] == "fake"
    assert generate_case(bench, _task(), ResearchAgent("fake", complete=ScriptedModel("x"))) is None


def test_splits_group_by_company_and_freeze_detects_tampering(bench):
    assert assign_split("ACME") == assign_split("acme")
    splits = {assign_split(f"T{i}") for i in range(200)}
    assert splits == {"train", "dev", "test"}

    manifest = freeze(bench, case_dirs(bench))
    entry = manifest["cases"]["acme_fy2025"]
    assert entry["split"] == assign_split("ACME")
    # Force the case into test, then tamper with its labels.
    manifest["cases"]["acme_fy2025"]["split"] = "test"
    (bench / "splits.json").write_text(json.dumps(manifest))
    expected_path = bench / "cases" / "acme_fy2025" / "expected.json"
    expected_path.write_text(expected_path.read_text().replace("invalid", "valid"))
    with pytest.raises(IntegrityError):
        run_eval(bench, model="heuristic", split="test")
    with pytest.raises(IntegrityError):
        freeze(bench, case_dirs(bench))


def test_run_eval_writes_record_and_compare(bench, settings):
    freeze(bench, case_dirs(bench))
    split = load_manifest(bench)["cases"]["acme_fy2025"]["split"]
    record, path = run_eval(bench, split=split, settings=settings)
    assert path.exists()
    for key in ("model", "auditor_prompt_version", "config_hash", "manifest_hash",
                "labeler_version", "git_commit", "summary"):  # fmt: skip
        assert key in record
    assert record["model"] == "heuristic" and record["cases"] == 1
    assert record["summary"]["errors"] == 5
    table = compare(load_runs(bench))
    assert "heuristic" in table and "F1" in table
    assert "failed" in compare([{"name": "small", "error": "no key"}])


# --- Scoring ----------------------------------------------------------------


def _finding(fid, text, verdict, error_type=None, correction=None):
    return {"finding_id": fid, "claim_text": text, "verdict": verdict, "error_type": error_type,
            "correction": correction, "evidence": [{"quote": "q"}] if verdict != "uncertain" else []}  # fmt: skip


def test_score_case_aggregates_split_claims_and_unscored():
    sentence = "Revenue was $120.0 billion and net income was $21.0 billion in fiscal 2025."
    expected = {
        "labels": [
            {
                "quote": sentence,
                "verdict": "invalid",
                "error_type": "misstated_figure",
                "correction_contains": "$24.0 billion",
            }
        ],  # fmt: skip
        "unlabeled": ["Backlog reached $7.5 billion in fiscal 2025."],
    }
    export = {
        "audit": {"latency_s": 1.0, "cost_usd": 0.0},
        "findings": [
            _finding(1, "Revenue was $120.0 billion", "valid"),
            _finding(
                2,
                "net income was $21.0 billion in fiscal 2025",
                "invalid",
                "misstated_figure",
                "net income was $24.0 billion",
            ),  # fmt: skip
            _finding(3, "Backlog reached $7.5 billion", "invalid", "misstated_figure"),
            _finding(4, "Margins expanded sharply", "invalid", "missing_source"),
        ],
    }
    scored = score_case(export, expected)
    (row,) = scored["rows"]
    assert row["predicted"] == "invalid" and row["correction_ok"]
    assert row["predicted_error_types"] == ["misstated_figure"]
    assert scored["unscored_invalid"] == ["Backlog reached $7.5 billion"]
    assert scored["unlabeled_invalid"] == ["Margins expanded sharply"]


# --- Feedback ---------------------------------------------------------------


def test_feedback_export(tmp_path):
    def record(fid, action, **review):
        return {
            "audit_id": 1, "model": "m", "prompt_version": "v1",
            "claim": {"id": fid, "text": f"claim {fid}", "claim_type": "numeric", "span": [0, 5],
                      "cited_source": None},
            "finding": {"id": fid, "verdict": "invalid", "error_type": "misstated_figure",
                        "correction": "fixed", "evidence_passage_ids": [1],
                        "evidence_quotes": ["row"], "rationale": "r"},
            "review": {"action": action, "reviewer": "ann", "created_at": "t", **review},
        }  # fmt: skip

    log = tmp_path / "feedback.jsonl"
    lines = [
        record(1, "reject"),
        record(1, "accept"),  # later review of the same finding wins
        record(2, "edit", edited_verdict="valid", notes="figure is right"),
        record(3, "reject"),
    ]
    log.write_text("".join(json.dumps(r) + "\n" for r in lines))
    counts = export_feedback(log, tmp_path / "out.jsonl")
    assert counts == {"total": 3, "accepted": 1, "edited": 1, "rejected": 1}
    ex = {
        e["provenance"]["finding_id"]: e for e in map(json.loads, (tmp_path / "out.jsonl").open())
    }
    assert all(e["status"] == "candidate" for e in ex.values())
    assert ex[1]["target"]["verdict"] == "invalid" and ex[1]["rejected"] is None
    assert ex[2]["target"] == {"verdict": "valid", "error_type": None, "correction": None,
                               "rationale": "figure is right"}  # fmt: skip
    assert ex[2]["rejected"]["verdict"] == "invalid"
    assert ex[3]["target"] is None and ex[3]["rejected"]["correction"] == "fixed"


def test_bench_cli_registered():
    from typer.testing import CliRunner

    from auditrl.cli import app

    result = CliRunner().invoke(app, ["bench", "--help"])
    assert result.exit_code == 0
    for cmd in ("fetch", "generate", "label", "freeze", "eval", "compare", "baselines"):
        assert cmd in result.output


def test_universe_and_baselines_files_exist():
    root = Path(__file__).resolve().parents[2] / "data" / "benchmark"
    assert len([line for line in (root / "universe.txt").read_text().splitlines()
                if line and not line.startswith("#")]) >= 40  # fmt: skip
    assert (root / "baselines.yaml").exists()
