"""`auditrl train ...`: judge training data, RL gate, and offline judge evaluation.

GPU training itself runs on Modal: see `auditrl/training/modal_app.py`.
"""

import json
from pathlib import Path
from typing import Annotated

import typer

from auditrl.config import get_settings

train_app = typer.Typer(no_args_is_help=True, help="Phase 3 judge training tools")


def _training_dir() -> Path:
    return get_settings().training_dir.resolve()


@train_app.command("build-data")
def build_data(
    name: Annotated[
        str, typer.Option(help="Dataset name (directory under data/training)")
    ] = "judge-v1",
    per_task: Annotated[int, typer.Option(help="Synthetic examples per task")] = 40,
    seed: int = 0,
    silver: Annotated[bool, typer.Option(help="Include labeled agent reports")] = True,
    feedback: Annotated[bool, typer.Option(help="Include reviewer feedback")] = True,
) -> None:
    """Build train/val JSONL from train/dev-split companies (test split is never read)."""
    from auditrl.training.data import BuildConfig, build_dataset

    settings = get_settings()
    cfg = BuildConfig(name=name, seed=seed, per_task=per_task, top_k=settings.judge_top_k,
                      include_silver=silver, include_feedback=feedback)  # fmt: skip
    out = _training_dir() / name
    manifest = build_dataset(
        settings.bench_dir.resolve(), out, cfg, settings.data_dir / "feedback.jsonl"
    )
    typer.echo(json.dumps(manifest["counts"], indent=1))
    typer.echo(f"dataset {manifest['dataset_hash']} -> {out}")


@train_app.command("rl-gate")
def rl_gate_cmd(
    min_reviews: int = 50,
    threshold: float = 0.9,
) -> None:
    """Measure verifier/reviewer agreement; GRPO launches only if this passes."""
    from auditrl.training.gate import rl_gate

    settings = get_settings()
    result = rl_gate(
        settings.data_dir / "feedback.jsonl",
        _training_dir() / "rl_gate.json",
        min_reviews=min_reviews,
        threshold=threshold,
    )
    typer.echo(json.dumps({k: result[k] for k in ("reviews", "agreement", "passed", "reason")}))


@train_app.command("eval-judge")
def eval_judge_cmd(
    model: Annotated[
        str, typer.Option(help="LiteLLM model, e.g. openai/judge-<run> for an adapter")
    ],
    api_base: Annotated[
        str | None, typer.Option(help="OpenAI-compatible endpoint (Modal URL)/v1")
    ] = None,
    dataset: str = "judge-v1",
    split: Annotated[str, typer.Option(help="val | train")] = "val",
    limit: Annotated[int, typer.Option(help="Max examples")] = 300,
    name: Annotated[str | None, typer.Option(help="Label for this evaluation")] = None,
) -> None:
    """Score a judge on labeled examples: verdict accuracy, F1, false accusations, reward."""
    from auditrl.judge import JudgeClient
    from auditrl.training.data import load_rows
    from auditrl.training.evaluate import eval_judge, save_eval

    settings = get_settings()
    data_dir = _training_dir() / dataset
    rows = load_rows(data_dir / f"{split}.jsonl")[:limit]
    manifest = json.loads((data_dir / "manifest.json").read_text())
    client = JudgeClient(model, api_base=api_base, api_key=settings.judge_api_key)
    record = eval_judge(rows, client, name or model, manifest)
    path = save_eval(record, _training_dir() / "evals")
    typer.echo(json.dumps(record["metrics"], indent=1))
    typer.echo(f"saved {path}")


@train_app.command("compare-judges")
def compare_judges(
    names: Annotated[list[str] | None, typer.Argument(help="Substrings of eval names")] = None,
) -> None:
    """Compare saved judge evaluations side by side (untuned vs SFT vs RL)."""
    from auditrl.training.evaluate import compare_evals

    paths = sorted((_training_dir() / "evals").glob("*.json"))
    records = [json.loads(p.read_text()) for p in paths]
    if names:
        records = [r for r in records if any(n in r["name"] for n in names)]
    typer.echo(compare_evals(records) if records else "no evaluations")
