import json
from pathlib import Path
from typing import Annotated

import typer
from sqlmodel import Session

from auditrl.bench_cli import bench_app
from auditrl.config import get_settings
from auditrl.export import build_export, to_json, to_jsonl, to_markdown
from auditrl.intake import parse_path
from auditrl.models import get_engine, init_db
from auditrl.pipeline import create_audit, run_audit

app = typer.Typer(no_args_is_help=True, help="AuditRL command-line tools")
app.add_typer(bench_app, name="bench")

SOURCE_SUFFIXES = {".md", ".markdown", ".txt", ".pdf", ".html", ".htm", ".csv", ".xlsx", ".xlsm"}


def _source_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in SOURCE_SUFFIXES)


@app.command()
def audit(
    report: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    sources: Annotated[list[Path], typer.Option("--sources", "-s", exists=True)],
    model: Annotated[str | None, typer.Option(help="LiteLLM model string or 'heuristic'")] = None,
    out: Annotated[Path | None, typer.Option(help="Directory for audit.md/json/jsonl")] = None,
) -> None:
    """Audit REPORT against one or more source files or directories."""
    settings = get_settings()
    engine = init_db(get_engine())
    files = [f for s in sources for f in _source_files(s)]
    if not files:
        raise typer.BadParameter("no supported source files found")
    with Session(engine) as session:
        run = create_audit(
            session,
            parse_path(report),
            [parse_path(f) for f in files],
            model=model or settings.model,
            origins={report.name: str(report), **{f.name: str(f) for f in files}},
            settings=settings,
        )
    run = run_audit(run.id, engine=engine, settings=settings)
    if run.status != "completed":
        typer.secho(f"audit failed: {run.error}", fg="red", err=True)
        raise typer.Exit(1)
    with Session(engine) as session:
        data = build_export(session, run.id)
    summary = to_markdown(data)
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "audit.md").write_text(summary)
        (out / "audit.json").write_text(to_json(data))
        (out / "audit.jsonl").write_text(to_jsonl(data))
        typer.echo(f"wrote {out}/audit.(md|json|jsonl)")
    typer.echo(summary)


@app.command("eval-p1")
def eval_p1(
    dataset: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    model: Annotated[str | None, typer.Option(help="LiteLLM model string or 'heuristic'")] = None,
    out: Annotated[Path | None, typer.Option(help="Write per-case results JSON here")] = None,
) -> None:
    """Smoke evaluation on the curated Phase 1 dataset."""
    from auditrl.evaluation import evaluate_dataset, format_summary

    result = evaluate_dataset(dataset, model=model or get_settings().model)
    typer.echo(format_summary(result))
    if out:
        out.write_text(json.dumps(result, indent=2, default=str))


@app.command()
def feedback(
    out: Annotated[Path, typer.Option(help="Output JSONL of candidate examples")] = Path(
        "var/feedback_examples.jsonl"
    ),
    log: Annotated[Path | None, typer.Option(help="Default: DATA_DIR/feedback.jsonl")] = None,
) -> None:
    """Export reviewer decisions as candidate training/eval examples."""
    from auditrl.review.feedback import export_feedback

    counts = export_feedback(log or get_settings().data_dir / "feedback.jsonl", out)
    typer.echo(f"{json.dumps(counts)} -> {out}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Run the API server."""
    import uvicorn

    uvicorn.run("auditrl.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
