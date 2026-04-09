# Review Queue Evidence Benchmark CI Validation

## Goal

Expose benchmark scorecard review-queue evidence metrics through
`evaluation-report.yml` so CI summaries and PR comments show the operational
quality of review evidence, not just backlog volume.

## Delivered

- Added benchmark scorecard outputs:
  - `review_queue_average_evidence`
  - `review_queue_evidence_ratio`
  - `review_queue_top_evidence_sources`
- Added matching job summary lines.
- Added matching PR comment fields:
  - benchmark summary line now includes review queue evidence ratio and average evidence
  - a dedicated `Benchmark Review Queue Evidence` row summarizes top evidence sources

## Validation

```bash
python3 -m py_compile \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

make validate-openapi
```

## Result

- `py_compile`: pass
- `flake8`: pass
- `pytest`: pass
- `validate-openapi`: pass

## Notes

- This branch intentionally stacks on top of:
  - `feat/review-queue-evidence-report`
  - `feat/review-queue-evidence-benchmark`
- The workflow remains backward-compatible because missing review queue evidence
  fields resolve to empty strings or zero values.
