"""Phase 3: judge format, grounding, pipeline integration, training data, rewards, gate."""

import json
from pathlib import Path

import pytest
from sqlmodel import Session, select

from auditrl.benchmark.agent import ResearchAgent
from auditrl.benchmark.dataset import build_tasks, generate_case, label_case
from auditrl.benchmark.edgar import EdgarClient
from auditrl.benchmark.splits import assign_split
from auditrl.intake import parse_text
from auditrl.judge import EvidenceItem, JudgeClient, JudgeOutput, build_messages, ground
from auditrl.judge.format import parse_output, render_target
from auditrl.models import Finding
from auditrl.pipeline import create_audit, run_audit
from auditrl.training.data import BuildConfig, build_dataset, load_rows
from auditrl.training.evaluate import compare_evals, eval_judge
from auditrl.training.gate import rl_gate
from auditrl.training.rewards import judge_reward, score
from auditrl.training.trainer import TrainConfig
from tests.conftest import APPLE_10K
from tests.test_benchmark import CIK, REPORT, ScriptedModel, _companyfacts

REPO = Path(__file__).resolve().parents[2]

REV = EvidenceItem(
    eid="E1",
    document="Acme Corp Form 10-K Fiscal 2025",
    text="Total revenue | Fiscal 2025: 120,000 | Fiscal 2024: 100,000",
    row_label="Total revenue",
    headers=["Fiscal 2025", "Fiscal 2024"],
    cells=["120,000", "100,000"],
    unit_scale=1e6,
)
GP = EvidenceItem(
    eid="E2",
    document="Acme Corp Form 10-K Fiscal 2025",
    text="Gross profit | Fiscal 2025: 50,000 | Fiscal 2024: 40,000",
    row_label="Gross profit",
    headers=["Fiscal 2025", "Fiscal 2024"],
    cells=["50,000", "40,000"],
    unit_scale=1e6,
)


def _out(**kw) -> str:
    return render_target(JudgeOutput(**kw))


# --- format and grounding -----------------------------------------------------


def test_prompt_renders_table_rows_with_scale_and_columns():
    user = build_messages("Revenue was $120.0 billion.", None, [REV])[1]["content"]
    assert (
        "[E1] (Acme Corp Form 10-K Fiscal 2025) Total revenue (in millions) | Fiscal 2025: 120,000"
        in user
    )
    assert "Cited source: none" in user


def test_parse_output_handles_fences_and_normalizes():
    out = parse_output(
        '```json\n{"verdict": "valid", "error_type": "unit_error", "correction": "x"}\n```'
    )
    assert out.verdict == "valid" and out.error_type is None and out.correction is None
    with pytest.raises(ValueError):
        parse_output("I think it is fine")
    with pytest.raises(ValueError):
        parse_output('{"verdict": "maybe"}')


def test_grounding_downgrades_unsupported_or_unevidenced_findings():
    claim = "Revenue was $110.0 billion in fiscal 2025."
    good = JudgeOutput(verdict="invalid", error_type="misstated_figure", evidence=["E1"],
                       correction="Revenue was $120.0 billion in fiscal 2025.")  # fmt: skip
    assert ground(good, claim, [REV])[0].verdict == "invalid"
    made_up = good.model_copy(update={"correction": "Revenue was $125.0 billion in fiscal 2025."})
    out, notes = ground(made_up, claim, [REV])
    assert out.verdict == "uncertain" and out.correction is None and "not supported" in notes[0]
    no_evidence = good.model_copy(update={"evidence": ["E9"]})
    out, notes = ground(no_evidence, claim, [REV])
    assert out.verdict == "uncertain" and any("unknown evidence" in n for n in notes)


def test_grounding_accepts_derived_percentages():
    margin = JudgeOutput(verdict="invalid", error_type="arithmetic_error", evidence=["E1", "E2"],
                         correction="Gross margin was 41.7% in fiscal 2025.")  # fmt: skip
    assert (
        ground(margin, "Gross margin was 45.0% in fiscal 2025.", [REV, GP])[0].verdict == "invalid"
    )
    decline = JudgeOutput(verdict="invalid", error_type="arithmetic_error", evidence=["E1"],
                          correction="Revenue fell 16.7% in fiscal 2025.")  # fmt: skip
    assert ground(decline, "Revenue fell 10.0% in fiscal 2025.", [REV])[0].verdict == "invalid"


# --- judge client and pipeline --------------------------------------------------


class FakeTransport:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.messages: list = []

    def __call__(self, name, messages):
        self.messages.append(messages)
        return (
            self.responses.pop(0) if len(self.responses) > 1 else self.responses[0],
            100,
            20,
            0.001,
        )


def test_judge_client_retries_invalid_json():
    transport = FakeTransport("not json", _out(verdict="valid", evidence=["E1"], rationale="ok"))
    client = JudgeClient("fake", transport=transport)
    out, _ = client.judge("Revenue was $120.0 billion.", None, [REV])
    assert out.verdict == "valid"
    assert client.usage.calls == 2 and "Invalid JSON" in transport.messages[1][-1]["content"]


def test_pipeline_uses_judge_and_guards_corrections(engine, settings, llm):
    report = parse_text(
        "report.md",
        "Apple's total net sales were $390.0 billion in fiscal 2023. "
        "Operating income was $114.3 billion in fiscal 2023.",
    )
    with Session(engine) as session:
        run = create_audit(session, report, [parse_text("apple_10k_2023.md", APPLE_10K)],
                           model="heuristic", settings=settings)  # fmt: skip

    def respond(name, messages):
        user = messages[1]["content"]
        if "390.0" in user:
            eid = next(
                line.split("]")[0][1:] for line in user.splitlines() if "Total net sales" in line
            )
            return _out(verdict="invalid", error_type="misstated_figure", evidence=[eid],
                        correction="Apple's total net sales were $383.3 billion in fiscal 2023.",
                        rationale="The filing reports 383,285."), 10, 5, 0.0  # fmt: skip
        return _out(verdict="invalid", error_type="misstated_figure", evidence=["E1"],
                    correction="Operating income was $999.0 billion in fiscal 2023."), 10, 5, 0.0  # fmt: skip

    judge = JudgeClient("fake-judge", transport=respond)
    run = run_audit(run.id, llm=llm, engine=engine, settings=settings, judge=judge)
    assert run.status == "completed" and run.prompt_tokens >= 20
    with Session(engine) as session:
        findings = {f.correction: f for f in session.exec(select(Finding)).all()}
    fixed = findings["Apple's total net sales were $383.3 billion in fiscal 2023."]
    assert fixed.verdict == "invalid" and fixed.verifier == "judge" and fixed.evidence_quotes
    guarded = findings[None]
    assert guarded.verdict == "uncertain" and "Downgraded" in guarded.rationale


# --- training data ---------------------------------------------------------------


def _tickers_by_split() -> dict[str, str]:
    found: dict[str, str] = {}
    i = 0
    while len(found) < 3:
        name = f"ZZ{i}"
        found.setdefault(assign_split(name), name)
        i += 1
    return found


@pytest.fixture
def multi_bench(tmp_path):
    tickers = _tickers_by_split()

    def fetch(url, ua):
        if url.endswith("company_tickers.json"):
            return json.dumps(
                {
                    str(i): {"cik_str": CIK, "ticker": t, "title": "ACME CORP"}
                    for i, t in enumerate(tickers.values())
                }
            ).encode()
        return json.dumps(_companyfacts()).encode()

    root = tmp_path / "bench"
    client = EdgarClient(tmp_path / "cache", "test you@example.com", fetch=fetch)
    tasks = build_tasks(root, client, list(tickers.values()), years_per_company=1)
    for task in tasks:  # one agent-written, labeled report per company
        generate_case(root, task, ResearchAgent("fake", complete=ScriptedModel(REPORT)))
        label_case(root, root / "cases" / task.task_id)
    return root, tickers


def test_build_dataset_respects_splits_and_is_deterministic(multi_bench, tmp_path):
    root, tickers = multi_bench
    cfg = BuildConfig(name="t", per_task=30)
    m1 = build_dataset(root, tmp_path / "d1", cfg)
    m2 = build_dataset(root, tmp_path / "d2", cfg)
    assert m1["dataset_hash"] == m2["dataset_hash"]
    assert m1["tickers"]["excluded_test"] == [tickers["test"]]
    train, val = (
        load_rows(tmp_path / "d1" / "train.jsonl"),
        load_rows(tmp_path / "d1" / "val.jsonl"),
    )
    assert {r["meta"].get("ticker") for r in train if r["source"] == "synthetic"} == {
        tickers["train"]
    }
    assert {r["meta"].get("ticker") for r in val if r["source"] == "synthetic"} == {tickers["dev"]}
    assert not any(tickers["test"].lower() in r["id"] for r in train + val)
    assert {r["source"] for r in train} == {"synthetic", "silver"}
    silver_invalid = [
        r for r in train if r["source"] == "silver" and r["target"]["verdict"] == "invalid"
    ]
    assert any(r["target"]["error_type"] == "citation_mismatch" for r in silver_invalid)


def test_gold_targets_are_grounded_and_score_maximum(multi_bench, tmp_path):
    root, _ = multi_bench
    build_dataset(root, tmp_path / "d", BuildConfig(name="t", per_task=60))
    rows = load_rows(tmp_path / "d" / "train.jsonl") + load_rows(tmp_path / "d" / "val.jsonl")
    synthetic = [r for r in rows if r["source"] == "synthetic"]
    assert {r["target"]["error_type"] for r in synthetic} >= {
        "misstated_figure", "wrong_period", "unit_error", "arithmetic_error", "citation_mismatch"}  # fmt: skip
    best = {"valid": 1.1, "invalid": 1.6, "uncertain": 0.5}
    for r in synthetic:
        evidence = [EvidenceItem(**e) for e in r["evidence"]]
        gold = parse_output(r["completion"][0]["content"])
        assert ground(gold, r["claim"], evidence)[0].verdict == gold.verdict, r["claim"]
        assert (
            score(r["completion"][0]["content"], r["target"], r["claim"], evidence).total
            == best[gold.verdict]
        )


def test_feedback_examples_drop_test_overlap(multi_bench, tmp_path):
    root, tickers = multi_bench
    test_report = (root / "cases" / f"{tickers['test'].lower()}_fy2025" / "report.md").read_text()
    leaked = test_report.split(". ")[0]

    def record(fid, claim):
        return {
            "audit_id": 1, "model": "m", "prompt_version": "v1",
            "claim": {"id": fid, "text": claim, "claim_type": "figure", "span": [0, 1], "cited_source": None},
            "finding": {"id": fid, "verdict": "valid", "error_type": None, "correction": None,
                        "evidence_passage_ids": [1], "evidence_quotes": ["Total revenue 120,000"],
                        "rationale": "r", "verifier": "figure"},
            "review": {"action": "accept", "reviewer": "ann", "created_at": "t"},
        }  # fmt: skip

    log = tmp_path / "feedback.jsonl"
    log.write_text(
        json.dumps(record(1, leaked))
        + "\n"
        + json.dumps(record(2, "Novel claim about revenue."))
        + "\n"
    )
    manifest = build_dataset(root, tmp_path / "d", BuildConfig(name="t", per_task=5), log)
    assert manifest["feedback_dropped_test_overlap"] == 1
    fb = [r for r in load_rows(tmp_path / "d" / "train.jsonl") if r["source"] == "feedback"]
    assert [r["claim"] for r in fb] == ["Novel claim about revenue."]


# --- rewards, gate, evaluation, config -----------------------------------------------


def test_rewards_penalize_false_accusations_and_unsupported_corrections():
    claim = "Revenue was $120.0 billion in fiscal 2025."
    valid_target = {"verdict": "valid", "error_type": None, "evidence": ["E1"], "correction": None}
    assert score(_out(verdict="valid", evidence=["E1"]), valid_target, claim, [REV]).total == 1.1
    accuse = _out(verdict="invalid", error_type="misstated_figure", evidence=["E1"],
                  correction="Revenue was $130.0 billion in fiscal 2025.")  # fmt: skip
    s = score(accuse, valid_target, claim, [REV])
    assert s.components["verdict"] == -1.5 and s.components["unsupported_correction"] == -0.5
    assert score("garbage", valid_target, claim, [REV]).total == -1.0
    unc_target = {"verdict": "uncertain", "error_type": None, "evidence": [], "correction": None}
    assert score(_out(verdict="uncertain"), unc_target, claim, [REV]).total == 0.5
    rewards = judge_reward(
        prompts=[None],
        completions=[[{"role": "assistant", "content": _out(verdict="valid", evidence=["E1"])}]],
        claim=[claim],
        evidence=[[REV.to_dict()]],
        target=[valid_target],
        trainer_state=None,
    )
    assert rewards == [1.1]


def test_rl_gate(tmp_path):
    def line(fid, action, verifier="figure", **review):
        return json.dumps({
            "audit_id": 1, "claim": None,
            "finding": {"id": fid, "verdict": "invalid", "verifier": verifier},
            "review": {"action": action, **review},
        })  # fmt: skip

    log = tmp_path / "feedback.jsonl"
    lines = [line(i, "accept") for i in range(55)] + [
        line(100, "reject"),
        line(101, "edit", edited_verdict="valid"),
    ]
    lines += [line(200 + i, "reject", verifier="judge") for i in range(20)]
    log.write_text("\n".join(lines) + "\n")
    result = rl_gate(log, tmp_path / "gate.json", min_reviews=50, threshold=0.9)
    assert result["reviews"] == 57 and result["passed"]
    assert json.loads((tmp_path / "gate.json").read_text())["passed"]
    assert not rl_gate(log, tmp_path / "gate.json", min_reviews=100)["passed"]


def test_eval_judge_and_compare(multi_bench, tmp_path):
    root, _ = multi_bench
    manifest = build_dataset(root, tmp_path / "d", BuildConfig(name="t", per_task=10))
    rows = load_rows(tmp_path / "d" / "train.jsonl")[:8]
    golds = iter(r["completion"][0]["content"] for r in rows)
    client = JudgeClient("oracle", transport=lambda name, messages: (next(golds), 1, 1, 0.0))
    record = eval_judge(rows, client, "oracle", manifest)
    assert record["metrics"]["verdict_accuracy"] == 1.0 and record["metrics"]["parse_rate"] == 1.0
    assert "oracle" in compare_evals([record])


def test_training_configs_load():
    sft = TrainConfig.from_yaml(REPO / "data" / "training" / "configs" / "sft.yaml")
    grpo = TrainConfig.from_yaml(REPO / "data" / "training" / "configs" / "grpo.yaml")
    assert sft.stage == "sft" and sft.lora.r == 16
    assert grpo.stage == "grpo" and grpo.init_adapter
    bad = Path(__file__).parent / "_bad.yaml"
    bad.write_text("stage: sft\nlearning_rat: 1\n")
    try:
        with pytest.raises(ValueError):
            TrainConfig.from_yaml(bad)
    finally:
        bad.unlink()


def test_train_cli_registered():
    from typer.testing import CliRunner

    from auditrl.cli import app

    result = CliRunner().invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    for cmd in ("build-data", "rl-gate", "eval-judge", "compare-judges"):
        assert cmd in result.output
