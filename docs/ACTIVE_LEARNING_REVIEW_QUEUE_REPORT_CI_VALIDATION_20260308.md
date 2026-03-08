# Active-Learning Review Queue Report CI Validation

## Goal
- Wire the existing `scripts/export_active_learning_review_queue_report.py` into
  `.github/workflows/evaluation-report.yml`.
- Publish the review-queue report as an artifact.
- Expose review-queue status in the job summary, PR comment, and benchmark
  scorecard generation path.

## Key Changes
- Added workflow-dispatch inputs:
  - `active_learning_review_queue_report_enable`
  - `active_learning_review_queue_report_input`
  - `active_learning_review_queue_report_top_k`
- Added workflow env vars:
  - `ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_ENABLE`
  - `ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_INPUT`
  - `ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_OUTPUT_JSON`
  - `ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_OUTPUT_CSV`
  - `ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_TOP_K`
- Added optional workflow step:
  - `Build active-learning review queue report (optional)`
- Added optional artifact upload:
  - `Upload active-learning review queue report`
- Extended job summary and PR comment with:
  - input path
  - total queue size
  - operational status
  - critical/high counts and ratios
  - automation-ready count and ratio
  - top sample types / priorities / decision sources / review reasons
- Fed the generated JSON summary into the benchmark scorecard step when present.

## Validation Commands
```bash
python3 -m py_compile \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

## Expected Result
- Workflow regression test covers the new dispatch inputs, env vars, report
  build step, artifact upload, summary lines, and PR comment strings.
- No changes are required to the underlying review-queue export script because
  the script already exists on `main`; this work is CI wiring only.
