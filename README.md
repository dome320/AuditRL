# AuditRL

A source-grounded auditing system for AI-generated financial research reports. Given a
report and its supporting documents (10-K excerpts, financial tables), AuditRL extracts
atomic claims, verifies figures and arithmetic against the sources, and returns
evidence-backed findings with suggested corrections. Reviewer decisions are logged as
training and evaluation signal for later phases.

**Status:** Phase 1 (audit-only MVP). Phases 2-4 (benchmark, post-training, research
release) are planned.

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

## Layout

```
backend/auditrl/
  intake/       parsers + passage provenance      extraction/  claim extraction, span alignment
  llm/          LiteLLM client, prompts, offline  retrieval/   BM25 + table-row index
  verify/       normalize, figure, arithmetic,    findings/    merge checks into findings
                citation, safe_eval               review/      reviews + feedback log
  export/       md/html/json/jsonl                api/         FastAPI app
  pipeline.py   orchestration                     evaluation.py  curated-set scoring
frontend/       React + Vite review UI
data/curated/   10 smoke-test cases (see its README for limitations)
```

## Roadmap

- **Phase 2:** agent-generated report traces, independent human labels, frozen held-out
  split by company/filing, baselines (small untuned model and a frontier model).
- **Phase 3:** LoRA SFT of a 1-8B model on validated examples; optional RL using the Phase 1
  verifiers as rewards, if they agree well enough with human labels.
- **Phase 4:** demo, reproducible experiments, adapters, and a technical report.
