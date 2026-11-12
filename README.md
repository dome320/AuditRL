# AuditRL

A source-grounded auditing system for AI-generated financial research reports. Given a
report and its supporting documents (10-K excerpts, financial tables), AuditRL extracts
atomic claims, verifies figures and arithmetic against the sources, and returns
evidence-backed findings with suggested corrections. Reviewer decisions are logged as
training and evaluation signal for later phases.

**Status:** Phase 1 (audit-only MVP) is done. Phase 2 (benchmark tooling and real SEC data)
is built; generating the agent reports needs a provider API key. Phases 3-4 (post-training,
research release) are planned.

## How it works

An LLM *proposes* (which sentences are claims, which source row a figure refers to, how a
percentage was computed); deterministic Python *decides*. Every verdict comes from code
that re-parses the quoted numbers, checks that quotes actually appear in the sources, and
recomputes arithmetic with rounding-interval propagation. Claims that cannot be grounded
come back `uncertain` with no correction rather than a guessed accusation.

```
report + sources -> intake (passages with provenance) -> claim extraction (span-aligned)
  -> retrieval (BM25 + table rows) -> verifiers (figure, arithmetic, citation)
  -> findings (verdict, error type, evidence, rationale, correction) -> review -> export
```

Detected error types: `arithmetic_error`, `misstated_figure`, `wrong_period`,
`unit_error`, `citation_mismatch`. Qualitative statements and forecasts are marked not
checkable.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Node 20+.

```bash
make install
cp .env.example .env        # set AUDITRL_MODEL and the matching provider key
make dev                    # API on :8000, UI on http://localhost:5173
```

`AUDITRL_MODEL` accepts any [LiteLLM](https://docs.litellm.ai/docs/providers) model string
(`anthropic/...`, `openai/...`, `ollama/...`). Set it to `heuristic` to run fully offline
with a rule-based stand-in for the LLM. That mode handles common sentence patterns and is
meant for demos and tests, not as a substitute for a model.

### CLI

```bash
make audit REPORT=data/curated/apple_fy2023/report.md SOURCES=data/curated/apple_fy2023/sources
cd backend && uv run auditrl audit report.md -s sources/ --model heuristic --out out/
make eval-p1                # precision/recall on the curated dataset
make test                   # backend test suite
```

## Supported inputs

Markdown/text (including pipe tables), PDF (page numbers kept), SEC HTML filings (tables
kept as cells; split `$`/`)` cells merged), CSV, and XLSX. Table scale hints such as
"(in millions)" are applied to bare table values.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/audits` | multipart: `report` file or `report_text`, `sources[]`, optional `model` |
| `GET` | `/audits`, `/audits/{id}` | status, report text, claims with spans |
| `GET` | `/audits/{id}/findings` | findings with evidence, checks, latest review |
| `POST` | `/findings/{id}/review` | `accept` / `reject` / `edit` (+ verdict, correction, notes) |
| `GET` | `/audits/{id}/export?format=md\|html\|json\|jsonl` | human and machine-readable output |
| `GET` | `/documents/{id}` | source text for the evidence viewer |

Reviews are also appended to `$AUDITRL_DATA_DIR/feedback.jsonl` as candidate training and
evaluation examples. Each audit run records model, prompt version, config hash, latency,
tokens, and cost.

## Benchmark (Phase 2)

The benchmark audits reports written by an LLM research agent working from real SEC data,
so the errors are the ones agents actually make rather than planted ones.

1. **Sources.** `bench fetch` downloads XBRL company facts from SEC EDGAR for the tickers
   in `data/benchmark/universe.txt` (44 large caps) and renders each annual 10-K as a
   markdown source (current and prior year, in millions, with accession and filing URL).
   Each task is one company and fiscal year; it gets that year's 10-K plus the prior
   year's, so citing the wrong filing is a possible mistake. Set `AUDITRL_SEC_USER_AGENT`
   to a name and email first, because the SEC requires it.
2. **Reports.** `bench generate` runs the research agent (`AUDITRL_AGENT_MODEL`,
   temperature 0.7). The agent can list the filings, read them, and use a calculator. It
   writes a 200-350 word results review, and its full trace is saved next to the report.
3. **Labels.** An independent deterministic labeler checks every figure in the report
   against the XBRL facts. It covers values, margins, growth rates, fiscal years, unit
   scale, and citations. It shares no code paths with the auditor's LLM or retrieval. When
   it can't tell what a figure refers to, or a percentage falls within 0.15pp of a rounding
   boundary, it leaves the sentence unlabeled. Unlabeled sentences are excluded from
   scoring, and coverage is reported per case. These are silver labels, not human labels.
4. **Splits.** `bench freeze` assigns train/dev/test splits by hashing the ticker, so all
   years of a company land in the same split. It records a SHA-256 hash of each case's
   report, labels, and sources. Evaluation refuses to run on test cases that changed after
   the freeze.
5. **Evaluation.** `bench eval --model X --split test` writes a run record to
   `data/benchmark/runs/`. The record includes the model, auditor prompt version, config
   hash, manifest hash, labeler version, git commit, the metrics, and latency and cost per
   report. `bench baselines` runs every entry in `data/benchmark/baselines.yaml`, and
   `bench compare` prints runs side by side.

```bash
cd backend
uv run auditrl bench fetch                  # already done; data/benchmark/sources is committed
uv run auditrl bench generate --limit 5     # needs a provider key for AUDITRL_AGENT_MODEL
uv run auditrl bench freeze
uv run auditrl bench eval --model heuristic --split dev
uv run auditrl bench baselines && uv run auditrl bench compare
uv run auditrl feedback                     # reviewer decisions -> candidate examples
```

`auditrl feedback` turns the review log into candidate examples. Accepted findings become
targets. Edits become targets, with the model's original output kept as the rejected
alternative. Rejections are recorded as rejected outputs only. Everything is marked
`candidate` until it is promoted for training.

## Layout

```
backend/auditrl/
  intake/       parsers + passage provenance      extraction/  claim extraction, span alignment
  llm/          LiteLLM client, prompts, offline  retrieval/   BM25 + table-row index
  verify/       normalize, figure, arithmetic,    findings/    merge checks into findings
                citation, safe_eval               review/      reviews + feedback log
  export/       md/html/json/jsonl                api/         FastAPI app
  pipeline.py   orchestration                     evaluation.py  claim-level scoring
  benchmark/    EDGAR client, research agent, labeler, dataset, splits, harness
frontend/       React + Vite review UI
data/curated/   10 smoke-test cases (see its README for limitations)
data/benchmark/ universe, tasks, rendered SEC sources, cases, splits.json, runs/
```

## Roadmap

- **Phase 2 (remaining):** generate agent reports once an API key is configured, freeze
  the splits, and run the strong and small baselines. Human labels on a sample would
  measure how accurate the silver labels are.
- **Phase 3:** LoRA SFT of a 1-8B model on validated examples; optional RL using the Phase 1
  verifiers as rewards, if they agree well enough with human labels.
- **Phase 4:** demo, reproducible experiments, adapters, and a technical report.
