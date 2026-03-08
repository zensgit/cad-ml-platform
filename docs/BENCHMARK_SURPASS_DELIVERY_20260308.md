# Benchmark Surpass Delivery

## Scope

- Baseline: `origin/main@19b3c6c4e9bebd639aef0dc38e85d26f3eea40f9`
- Date: `2026-03-08`
- Purpose: record what was actually delivered into mainline for the current benchmark/surpass batch.

## Delivered In Main

### 1. Assistant structured evidence

Merged via `#149`.

Delivered:

- structured evidence helpers under `src/core/assistant/`
- assistant API output now carries stable evidence payloads
- assistant explainability tests and validation doc

Validation evidence:

- `tests/unit/assistant/test_explainability.py`
- `tests/unit/assistant/test_api_service.py`
- `tests/unit/assistant/test_llm_api.py`

### 2. Benchmark scorecard generator

Merged via `#152`.

Delivered:

- `scripts/generate_benchmark_scorecard.py`
- unified JSON + Markdown scorecard generation
- status model across:
  - hybrid
  - graph2d
  - history_sequence
  - brep
  - migration_governance

Validation evidence:

- `tests/unit/test_generate_benchmark_scorecard.py`

### 3. Benchmark scorecard CI integration

Originally introduced in stacked `#153`, integrated into `main` during stacked merge.

Delivered:

- `evaluation-report.yml` can optionally generate benchmark scorecard artifacts
- job summary now emits benchmark component status lines
- PR comment includes benchmark scorecard summary

Validation evidence:

- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
- `docs/BENCHMARK_SCORECARD_CI_VALIDATION_20260308.md`

### 4. OCR review pack export

Merged via `#151`.

Delivered:

- `scripts/export_ocr_review_pack.py`
- flat CSV + JSON summary export
- support for:
  - review candidate filtering
  - include-ready mode
  - top-k limiting

Validation evidence:

- `tests/unit/test_export_ocr_review_pack.py`

### 5. OCR review pack CI integration

Originally introduced in stacked `#154`, integrated into `main` during stacked merge.

Delivered:

- `evaluation-report.yml` can optionally build OCR review pack artifacts
- job summary emits:
  - review candidate count
  - automation ready count
  - readiness / coverage
  - top priorities / gaps / reasons / actions
- PR comment now includes OCR review pack status and insights

Validation evidence:

- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
- `docs/OCR_REVIEW_PACK_CI_VALIDATION_20260308.md`

## Integration Validation

The stacked merge batch was validated in the isolated merge worktree before pushing `main`.

Validated groups:

- workflow regression:
  - `pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
  - result: `3 passed`
- assistant explainability:
  - `pytest -q tests/unit/assistant/test_explainability.py tests/unit/assistant/test_api_service.py tests/unit/assistant/test_llm_api.py`
  - result: `70 passed, 2 skipped`
- benchmark scorecard:
  - `pytest -q tests/unit/test_generate_benchmark_scorecard.py`
  - result: `2 passed`
- OCR review pack:
  - `pytest -q tests/unit/test_export_ocr_review_pack.py`
  - result: `2 passed`
- workflow syntax / lint:
  - YAML parse passed
  - `py_compile` passed
  - `flake8` passed for workflow regression test

## Delivery Effect

This batch materially improves benchmark-surpass readiness in four ways:

1. Evidence is now structured, not narrative-only.
2. Benchmark progress is summarized in a single scorecard instead of scattered metrics.
3. OCR review now has exportable, automatable operator artifacts.
4. CI can surface benchmark status and OCR review status as first-class signals.

## What This Does Not Yet Prove

This batch improves architecture, observability, and operator usefulness. It does not by itself prove:

- history-sequence strong real-data superiority
- broad B-Rep superiority
- standalone Graph2D superiority

Those still require more real-data evidence.

## Next Recommended Work

1. Add assistant evidence offline export and CI summary wiring
2. Feed real benchmark artifacts into scorecard generation in CI
3. Expand OCR review-pack inputs from fixtures to replayable real-data artifacts
4. Continue history `.h5` and STEP/B-Rep real-data validation

## Status

Current status should be described as:

- `benchmark-ready and partially beyond on engineering operability`
- not yet `fully benchmark-surpassing on all semantic branches`
