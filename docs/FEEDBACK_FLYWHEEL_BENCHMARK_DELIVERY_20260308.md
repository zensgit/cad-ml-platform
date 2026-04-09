# Feedback Flywheel Benchmark Delivery 20260308

## Scope

This delivery closes the benchmark gap between feedback collection, retraining evidence,
and CI-visible benchmark status.

Merged scope:

- `#178` feedback flywheel scorecard component
- `#179` feedback flywheel CI wiring
- `#180` reusable feedback flywheel benchmark bundle

## Design

### 1. Scorecard layer

`scripts/generate_benchmark_scorecard.py` now treats the feedback flywheel as a first-class
benchmark component instead of an implicit side effect of feedback exports.

The scorecard evaluates:

- feedback volume
- correction volume
- coarse correction volume
- fine-tune sample coverage
- metric-training triplet coverage
- label-distribution evidence

Status model:

- `missing`
- `passive_feedback_only`
- `feedback_collected`
- `partially_closed_loop`
- `closed_loop_ready`

### 2. CI layer

`.github/workflows/evaluation-report.yml` now accepts:

- `benchmark_scorecard_feedback_summary`
- `benchmark_scorecard_finetune_summary`
- `benchmark_scorecard_metric_train_summary`

The workflow passes these summaries into the benchmark scorecard step and surfaces
`feedback_flywheel_status` in:

- step outputs
- GitHub job summary
- PR comment benchmark table

### 3. Reusable artifact layer

`src/core/benchmark/feedback_flywheel.py` centralizes:

- status derivation
- recommendation derivation
- Markdown rendering

`scripts/export_feedback_flywheel_benchmark.py` produces a standalone JSON/Markdown artifact
for operators who want the feedback flywheel benchmark without the full scorecard.

## Files

Core files:

- `scripts/generate_benchmark_scorecard.py`
- `.github/workflows/evaluation-report.yml`
- `scripts/train_metric_model.py`
- `scripts/export_feedback_flywheel_benchmark.py`
- `src/core/benchmark/feedback_flywheel.py`

Primary tests:

- `tests/unit/test_generate_benchmark_scorecard.py`
- `tests/unit/test_train_metric_model.py`
- `tests/unit/test_feedback_flywheel_benchmark.py`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
- `tests/unit/test_finetune_from_feedback.py`
- `tests/test_feedback.py`

Validation docs:

- `docs/FEEDBACK_FLYWHEEL_BENCHMARK_VALIDATION_20260308.md`
- `docs/FEEDBACK_FLYWHEEL_BENCHMARK_CI_VALIDATION_20260308.md`
- `docs/FEEDBACK_FLYWHEEL_BUNDLE_VALIDATION_20260308.md`

## Validation

Local validation after stacked merge:

```bash
python3 -m py_compile \
  scripts/generate_benchmark_scorecard.py \
  scripts/train_metric_model.py \
  scripts/export_feedback_flywheel_benchmark.py \
  tests/unit/test_generate_benchmark_scorecard.py \
  tests/unit/test_train_metric_model.py \
  tests/unit/test_feedback_flywheel_benchmark.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

pytest -q \
  tests/unit/test_generate_benchmark_scorecard.py \
  tests/unit/test_train_metric_model.py \
  tests/unit/test_finetune_from_feedback.py \
  tests/test_feedback.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  tests/unit/test_feedback_flywheel_benchmark.py

git diff --check origin/main...HEAD
```

Result:

- `28 passed`
- `git diff --check` passed
- `py_compile` passed

## Outcome

The benchmark system now exposes whether human feedback has actually become a usable retraining
loop, not just whether feedback exists. This improves operator visibility and closes a real
competitive gap versus systems that only report extraction or classification accuracy.
