# Curated Phase 1 dataset

Ten small audit cases used as a smoke test for the Phase 1 pipeline (`make eval-p1`).

Each case directory contains:

- `sources/`: income-statement excerpts (in millions) transcribed from the company's 10-K.
- `report.md`: a short research write-up with correct claims, one or two qualitative
  sentences, and injected errors.
- `expected.json`: claim-level labels. Each label has a verbatim `quote`, a `verdict`
  (`valid` or `invalid`), and for errors an `error_type` and a `correction_contains`
  string that a correct fix must include. Every injected error is labeled, so an
  `invalid` finding without a matching label counts as a false positive.

Injected errors (18): arithmetic errors (6), wrong period (4), misstated figure (4),
unit/scale slips (2), and mismatched citations (2).

Regenerate with `python data/curated/build_dataset.py`; edit cases there, not in the
generated files.

## Limitations

- Figures were transcribed by hand from 10-K filings and should be spot-checked against
  EDGAR before being treated as ground truth.
- Reports are templated, single-company, and short. The offline `heuristic` client was
  developed alongside these sentences, so its scores here measure pipeline plumbing, not
  generalization. Phase 2 replaces this set with agent-generated reports, independent
  labels, and a held-out split.
