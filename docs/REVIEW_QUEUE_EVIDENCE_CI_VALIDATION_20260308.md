# Review Queue Evidence CI Validation

## Goal

Expose active-learning review queue evidence richness through
`evaluation-report.yml` so CI summaries and PR comments show whether queued
samples already have reviewer-friendly evidence.

## Delivered

- Added new step outputs for `Build active-learning review queue report (optional)`:
  - `evidence_count_total`
  - `average_evidence_count`
  - `records_with_evidence_count`
  - `records_with_evidence_ratio`
  - `top_evidence_sources`
- Added matching lines to the job summary.
- Added matching fields to the PR comment summary and insights lines.

## Validation

```bash
python3 -m py_compile \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

## Result

- `py_compile`: pass
- `flake8`: pass
- `pytest`: pass

## Notes

- This branch is intentionally stacked on top of `feat/review-queue-evidence-report`.
- The workflow remains backward-compatible when older review queue summaries omit
  evidence keys; the export script branch provides those fields.
