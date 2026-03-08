# Active Learning Review Queue Report Validation 2026-03-08

## Goal

Add a stable review-queue summary artifact that benchmark/governance automation can
consume without needing the live API.

## Delivered

- Added [scripts/export_active_learning_review_queue_report.py](/private/tmp/cad-ml-platform-next-benchmark-20260308/scripts/export_active_learning_review_queue_report.py).
- The script accepts either:
  - `--data-dir` pointing at `ACTIVE_LEARNING_DATA_DIR`
  - `--input-path` pointing at a review queue `csv/json/jsonl` export
- The generated summary JSON includes:
  - `status`
  - `total`
  - `high_priority_count`
  - `critical_priority_count`
  - `high_priority_ratio`
  - `automation_ready_count`
  - `automation_ready_ratio`
  - `by_sample_type`
  - `by_feedback_priority`
  - `by_decision_source`
  - `by_review_reason`
  - `top_feedback_priorities`
  - `top_decision_sources`
  - `top_review_reasons`
  - `recommended_actions`

## Workflow Integration

- `evaluation-report.yml` now supports optional review queue summary generation via:
  - `active_learning_review_queue_report_enable`
  - `active_learning_review_queue_report_input`
  - `active_learning_review_queue_report_data_dir`
- New env vars:
  - `ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_ENABLE`
  - `ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_INPUT`
  - `ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_DATA_DIR`
  - `ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_OUTPUT_JSON`
- New workflow step:
  - `Build active-learning review queue report (optional)`
- New artifact:
  - `active-learning-review-queue-report-${run_number}`

## Benchmark / Summary Wiring

- Benchmark scorecard workflow env now accepts:
  - `BENCHMARK_SCORECARD_ASSISTANT_EVIDENCE_JSON`
  - `BENCHMARK_SCORECARD_REVIEW_QUEUE_JSON`
- Job summary now surfaces:
  - benchmark assistant explainability status
  - benchmark review queue status
  - review queue input/status/counts/top priorities/recommendations
- PR comment now surfaces:
  - review queue report status
  - review queue insights

## Validation

Commands:

```bash
python3 -m py_compile \
  scripts/export_active_learning_review_queue_report.py \
  tests/unit/test_export_active_learning_review_queue_report.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  tests/unit/test_generate_benchmark_scorecard.py

flake8 \
  scripts/export_active_learning_review_queue_report.py \
  tests/unit/test_export_active_learning_review_queue_report.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  tests/unit/test_generate_benchmark_scorecard.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_export_active_learning_review_queue_report.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  tests/unit/test_generate_benchmark_scorecard.py
```

Results:

- `py_compile`: passed
- `flake8`: passed
- `pytest`: `7 passed`

## Notes

- This change does not require the live API server.
- The report is intentionally file-based so CI, benchmark scorecards, and offline
  governance tooling can all consume the same artifact shape.
