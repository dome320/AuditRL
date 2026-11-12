"""`auditrl bench ...`: build, label, freeze, and evaluate the Phase 2 benchmark."""

import json
from pathlib import Path
from typing import Annotated

import typer

from auditrl.config import get_settings

bench_app = typer.Typer(no_args_is_help=True, help="Phase 2 benchmark tools")

BenchOpt = Annotated[Path | None, typer.Option("--bench", help="Benchmark directory")]


def _bench(path: Path | None) -> Path:
    return (path or get_settings().bench_dir).resolve()


@bench_app.command("fetch")
def fetch(
    tickers: Annotated[str | None, typer.Option(help="Comma-separated tickers")] = None,
    universe: Annotated[
        Path | None,
        typer.Option(help="File with one ticker per line (default: bench/universe.txt)"),
    ] = None,
    years: Annotated[int, typer.Option(help="Fiscal years per company")] = 2,
    bench: BenchOpt = None,
) -> None:
    """Download XBRL facts from SEC EDGAR, render source filings, and write tasks.jsonl."""
    from auditrl.benchmark.dataset import build_tasks
    from auditrl.benchmark.edgar import EdgarClient

    settings = get_settings()
    root = _bench(bench)
    if tickers:
        names = [t.strip() for t in tickers.split(",") if t.strip()]
    else:
        path = universe or root / "universe.txt"
        names = [
            line.split("#")[0].strip()
            for line in path.read_text().splitlines()
            if line.split("#")[0].strip()
        ]
    if not settings.sec_user_agent:
        raise typer.BadParameter("set AUDITRL_SEC_USER_AGENT to 'Your Name you@example.com'")
    client = EdgarClient(settings.data_dir / "edgar", settings.sec_user_agent)
    tasks = build_tasks(root, client, names, years_per_company=years)
    typer.echo(f"{len(tasks)} tasks from {len(names)} tickers -> {root / 'tasks.jsonl'}")


@bench_app.command("generate")
def generate(
    model: Annotated[str | None, typer.Option(help="Agent model (LiteLLM string)")] = None,
    mode: Annotated[str, typer.Option(help="'tools' or 'context'")] = "tools",
    limit: Annotated[int | None, typer.Option(help="Max tasks to run")] = None,
    only: Annotated[str | None, typer.Option(help="Comma-separated task ids")] = None,
    force: Annotated[bool, typer.Option(help="Regenerate existing cases")] = False,
    bench: BenchOpt = None,
) -> None:
    """Have the research agent write reports for tasks (needs a provider API key)."""
    from auditrl.benchmark.agent import ResearchAgent
    from auditrl.benchmark.dataset import generate_case, label_case, load_tasks

    settings = get_settings()
    root = _bench(bench)
    tasks = load_tasks(root)
    if only:
        wanted = set(only.split(","))
        tasks = [t for t in tasks if t.task_id in wanted]
    agent = ResearchAgent(
        model or settings.agent_model,
        mode="context" if mode == "context" else "tools",
        temperature=settings.agent_temperature,
    )
    done = 0
    for task in tasks:
        if limit is not None and done >= limit:
            break
        trace = generate_case(root, task, agent, force=force)
        if trace is None:
            continue
        done += 1
        if trace.error:
            typer.secho(f"{task.task_id}: {trace.error}", fg="red", err=True)
            continue
        expected = label_case(root, root / "cases" / task.task_id)
        cov = expected["coverage"]
        typer.echo(
            f"{task.task_id}: {len(trace.report.split())} words, "
            f"{cov['labeled_figures']}/{cov['figures']} figures labeled, {cov['errors']} errors, "
            f"${trace.cost_usd:.4f}"
        )


@bench_app.command("label")
def label(bench: BenchOpt = None) -> None:
    """Re-run the deterministic labeler over every case (e.g. after a labeler change)."""
    from auditrl.benchmark.dataset import case_dirs, label_case

    root = _bench(bench)
    totals = {"figures": 0, "labeled_figures": 0, "errors": 0}
    for case_dir in case_dirs(root):
        cov = label_case(root, case_dir)["coverage"]
        for k in totals:
            totals[k] += cov[k]
    typer.echo(json.dumps(totals))


@bench_app.command("freeze")
def freeze_cmd(bench: BenchOpt = None) -> None:
    """Assign company-grouped splits and freeze case hashes in splits.json."""
    from auditrl.benchmark.dataset import case_dirs
    from auditrl.benchmark.splits import freeze

    root = _bench(bench)
    manifest = freeze(root, case_dirs(root))
    counts: dict[str, int] = {}
    for entry in manifest["cases"].values():
        counts[entry["split"]] = counts.get(entry["split"], 0) + 1
    typer.echo(f"frozen: {counts}")


@bench_app.command("eval")
def eval_cmd(
    model: Annotated[str | None, typer.Option(help="Auditor model or 'heuristic'")] = None,
    split: Annotated[str, typer.Option(help="train | dev | test | all")] = "test",
    limit: Annotated[int | None, typer.Option()] = None,
    name: Annotated[str | None, typer.Option(help="Run name (default: model)")] = None,
    bench: BenchOpt = None,
) -> None:
    """Evaluate the auditor on a split and save a run record under runs/."""
    from auditrl.benchmark.harness import run_eval
    from auditrl.evaluation import format_summary

    record, path = run_eval(_bench(bench), model=model, split=split, limit=limit, name=name)
    typer.echo(
        format_summary(
            {"model": record["model"], "summary": record["summary"], "cases": record["per_case"]}
        )
    )
    typer.echo(f"\nrun record: {path}")


@bench_app.command("compare")
def compare_cmd(
    runs: Annotated[
        list[str] | None, typer.Argument(help="Run id substrings (default: all)")
    ] = None,
    bench: BenchOpt = None,
) -> None:
    """Compare saved runs side by side."""
    from auditrl.benchmark.harness import compare, load_runs

    typer.echo(compare(load_runs(_bench(bench), runs)))


@bench_app.command("baselines")
def baselines_cmd(
    config: Annotated[Path | None, typer.Option(help="Default: bench/baselines.yaml")] = None,
    split: Annotated[str, typer.Option()] = "test",
    limit: Annotated[int | None, typer.Option()] = None,
    bench: BenchOpt = None,
) -> None:
    """Run every baseline in baselines.yaml and print a comparison."""
    from auditrl.benchmark.harness import compare, run_baselines

    root = _bench(bench)
    records = run_baselines(root, config or root / "baselines.yaml", split=split, limit=limit)
    typer.echo(compare(records))
