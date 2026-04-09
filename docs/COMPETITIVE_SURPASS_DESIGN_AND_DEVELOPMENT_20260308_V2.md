# Competitive Surpass Design And Development

## Objective

Build beyond benchmark parity by strengthening four layers together instead of
over-optimizing a single classifier:

1. Stable semantic outputs
2. Review and evidence operability
3. Retrieval and vector backend observability
4. Benchmark-grade CI summaries and operator runbooks

This document records both what is already in `main` and what is currently in
flight on stacked PRs.

## Design Axes

### 1. Semantic stability over raw model novelty

The platform should prefer:

- `hybrid` as primary decision path
- `graph2d` as weak signal / diagnostic signal
- `coarse + fine` labels as the stable external contract
- explicit `decision_source`, `decision_path`, `source_contributions`

### 2. Evidence-first review operations

Industrial usefulness depends on reviewer context, not only on classification
accuracy. The review loop therefore needs:

- actionable evidence payloads
- queue prioritization
- review-ready exports
- benchmark-visible evidence coverage

### 3. Retrieval backend operational readiness

Vector search must be measurable and governable. Readiness includes:

- Qdrant health
- indexing completeness
- migration coverage
- scan truncation visibility
- maintenance error taxonomy

### 4. CI as benchmark companion

Benchmark status should be visible in one place:

- job summary
- PR comment
- benchmark scorecard JSON/Markdown
- exported operator artifacts

## Delivered In Main

### A. Structured explainability and benchmark foundation

Already integrated before this round:

- assistant structured evidence
- benchmark scorecard generator
- benchmark scorecard CI integration
- OCR review pack export and CI wiring
- active-learning review queue export

Reference docs already in `main`:

- `docs/BENCHMARK_SURPASS_DELIVERY_20260308.md`
- `docs/BENCHMARK_ALIGNMENT_AND_BEYOND_STATUS_20260308.md`

### B. Qdrant maintenance error taxonomy

Integrated via `main@b12938e`.

Delivered:

- maintenance stats now classify Qdrant failures into typed categories
- maintenance output now exposes:
  - `error_type`
  - `error_severity`
  - `error_hint`

Validation:

- `tests/unit/test_maintenance_endpoint_coverage.py`
- `docs/QDRANT_MAINTENANCE_ERROR_TAXONOMY_VALIDATION_20260308.md`

## In-Flight Stack

### 1. Review queue evidence reporting

PR: `#173`

Delivered on branch:

- `scripts/export_active_learning_review_queue_report.py`
- summary metrics:
  - `evidence_count_total`
  - `average_evidence_count`
  - `records_with_evidence_count`
  - `records_with_evidence_ratio`
  - `top_evidence_sources`
- CSV export fields:
  - `evidence_count`
  - `evidence_sources`
  - `evidence_summary`

Validation:

- `pytest -q tests/unit/test_export_active_learning_review_queue_report.py`
- result: `2 passed`

Doc:

- `docs/REVIEW_QUEUE_EVIDENCE_REPORT_VALIDATION_20260308.md`

### 2. Benchmark scorecard review-queue evidence status

PR: `#174`

Delivered on branch:

- benchmark `review_queue` component now carries evidence richness metrics
- new benchmark status: `evidence_gap`
- benchmark recommendations now explicitly call out weak review evidence

Validation:

- `pytest -q tests/unit/test_generate_benchmark_scorecard.py`
- result: `4 passed`

Doc:

- `docs/REVIEW_QUEUE_EVIDENCE_BENCHMARK_VALIDATION_20260308.md`

### 3. Review queue evidence CI wiring

PR: `#175`

Delivered on branch:

- `evaluation-report.yml` now exports review queue evidence metrics
- job summary shows evidence totals, ratio, and top sources
- PR comment shows evidence totals and reviewer-facing evidence source summary

Validation:

- `pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
- result: `3 passed`
- `make validate-openapi`
- result: `5 passed`

Doc:

- `docs/REVIEW_QUEUE_EVIDENCE_CI_VALIDATION_20260308.md`

### 4. Benchmark review-queue evidence CI wiring

PR: `#176`

Delivered on branch:

- benchmark scorecard workflow outputs now expose:
  - `review_queue_average_evidence`
  - `review_queue_evidence_ratio`
  - `review_queue_top_evidence_sources`
- job summary now shows benchmark review queue evidence lines
- PR comment now includes a dedicated benchmark review queue evidence row

Validation:

- `pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
- result: `3 passed`
- `make validate-openapi`
- result: `5 passed`

Doc:

- `docs/REVIEW_QUEUE_EVIDENCE_BENCHMARK_CI_VALIDATION_20260308.md`

### 5. Qdrant benchmark readiness stack

Still active:

- `#170` qdrant readiness benchmark scorecard
- `#171` qdrant benchmark CI wiring

Intent:

- make vector backend readiness visible in benchmark scorecards
- keep benchmark claims tied to actual backend readiness rather than model-only metrics

## Validation Standard

For each delivery line:

1. `py_compile`
2. `flake8`
3. targeted `pytest`
4. workflow/openapi validation when routes or CI are touched
5. dedicated Markdown validation note

This is now the default operating model for benchmark-surpass work.

## Current Assessment

The codebase is already beyond simple benchmark parity on engineering
operability in these areas:

- explainability structure
- review exportability
- CI benchmark visibility
- Qdrant observability and governance

It is not yet fully beyond benchmark parity on all semantic branches. The main
remaining gaps are:

- more real-data history-sequence evidence
- broader B-Rep real-data evidence
- fully merged review-queue evidence stack
- fully merged Qdrant benchmark readiness stack

## Next Recommended Execution Order

1. Merge the review-queue evidence stack: `#173 -> #175` and scorecard branch `#174`
2. Merge the Qdrant benchmark readiness stack: `#170 -> #171`
3. Re-run integrated benchmark summary after both stacks land
4. Continue with the next low-conflict line:
   - benchmark companion summaries for review evidence
   - or retrieval-operability score rollup

## Status

Best current description:

- `benchmark-surpass on operability is in active delivery`
- `benchmark-surpass on full semantic proof is still accumulating evidence`
