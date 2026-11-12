# AuditRL

A source-grounded auditing system for AI-generated financial research reports. Given a
report and its supporting documents (10-K excerpts, financial tables), AuditRL extracts
atomic claims, verifies figures and arithmetic against the sources, and returns
evidence-backed findings with suggested corrections. Reviewer decisions are logged as
training and evaluation signal for later phases.

**Status:** Phase 1 (audit-only MVP) is done. Phase 2 (benchmark tooling and real SEC data)
is built; generating the agent reports needs a provider API key. Phase 3 (judge model
post-training) is built and ready to run on Modal; no model has been trained yet. Phase 4
(research release) is planned.

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

## Judge training (Phase 3)

Phase 3 fine-tunes a small open model (Qwen2.5-7B-Instruct with LoRA by default) as a
*judge*. The judge receives one claim plus the passages the auditor retrieved for it,
labeled E1 to Ek. It returns a JSON object with a verdict, error type, the evidence ids it
used, a correction, and a rationale. Setting `AUDITRL_JUDGE_MODEL` makes the auditor use the
judge instead of the deterministic verifiers. A deterministic grounding check still runs on
the judge's answer: an `invalid` verdict that cites no evidence is downgraded to
`uncertain`, and so is a correction whose figures don't appear in the cited evidence or
follow from it as a ratio or growth rate.

- **Data.** `auditrl train build-data` writes `data/training/judge-v1/{train,val}.jsonl` and
  a `manifest.json` with counts and a dataset hash.
  - Most examples are synthetic. Each one states a real figure from a train-split company's
    SEC filings, either correctly or with one injected error: a misstated figure, the wrong
    fiscal year, a units slip, wrong margin or growth arithmetic, or a citation to the wrong
    filing. Some state figures the filings don't contain. Labels are exact by construction.
    The evidence comes from the same retrieval the live auditor uses, and when retrieval
    misses the rows needed to decide, the label is `uncertain`.
  - Silver-labeled agent reports and accepted or edited reviewer feedback are added
    automatically once they exist.
  - Train data comes only from train-split companies and validation data only from
    dev-split companies. Test-split companies are never read. Feedback whose claim appears
    in a test-split report is dropped.
  - The current build has 1,600 train and 1,200 validation examples from 35 companies.
- **Rewards.** `auditrl/training/rewards.py` is built on the Phase 1 normalizer and the
  grounding check.
  - A correct verdict earns +1.0, with bonuses for the right error type, a correct fix, and
    citing the right evidence.
  - A false accusation costs -1.5, and a correction whose figures the evidence doesn't
    support costs -0.5.
  - Justified abstention earns +0.5, while abstaining when the evidence was enough costs
    -0.25.
- **SFT.** Supervised fine-tuning uses TRL and PEFT on a Modal GPU and is tracked in
  Weights & Biases. Each run logs the config, dataset hash, benchmark manifest hash, git
  commit, validation metrics, and the adapter as an artifact.
- **RL.** GRPO continues from the SFT adapter using the rewards above. It only launches
  if `auditrl train rl-gate` passes, meaning at least 50 reviewed findings with 90% or
  higher agreement between the verifiers and reviewers; otherwise pass `--force-rl`.
- **Comparison (T03).** `modal deploy` serves the base model and every adapter through an
  OpenAI-compatible vLLM endpoint. `auditrl train eval-judge` scores a judge on the
  validation split, and `bench eval --judge-model ...` scores it on the frozen benchmark
  test split.

```bash
cd backend
uv sync --extra cloud && uv run modal setup          # once; then create the wandb secret:
uv run modal secret create wandb WANDB_API_KEY=...
uv run auditrl train build-data
uv run modal run auditrl/training/modal_app.py --stage sft --config ../data/training/configs/sft.yaml
uv run auditrl train rl-gate                          # GRPO needs this to pass (or --force-rl)
uv run modal run auditrl/training/modal_app.py --stage grpo --config ../data/training/configs/grpo.yaml
uv run modal deploy auditrl/training/modal_app.py     # prints the serve URL
URL=https://<workspace>--auditrl-judge-serve.modal.run/v1
uv run auditrl train eval-judge --model openai/Qwen/Qwen2.5-7B-Instruct --api-base $URL --name untuned
uv run auditrl train eval-judge --model openai/judge-<sft run> --api-base $URL --name sft
uv run auditrl train compare-judges
uv run auditrl bench eval --split test --judge-model openai/judge-<sft run> --judge-api-base $URL
```

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
  judge/        judge prompt/format, client, retrieval, grounding guard
  training/     training data, rewards, RL gate, TRL trainer, Modal app
frontend/       React + Vite review UI
data/curated/   10 smoke-test cases (see its README for limitations)
data/benchmark/ universe, tasks, rendered SEC sources, cases, splits.json, runs/
data/training/  training configs, dataset manifests, judge evals, training run records
```

## Roadmap

- **Phase 2 (remaining):** generate agent reports once an API key is configured, freeze
  the splits, and run the strong and small baselines. Human labels on a sample would
  measure how accurate the silver labels are.
- **Phase 3 (remaining):** run SFT on Modal; compare the untuned and SFT judges on
  validation and on the benchmark test split; collect reviews until the RL gate can be
  measured, then decide whether to run GRPO.
- **Phase 4:** demo, reproducible experiments, adapters, and a technical report.
