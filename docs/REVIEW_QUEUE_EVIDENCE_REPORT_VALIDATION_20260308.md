# Review Queue Evidence Report Validation

## Goal

Extend `scripts/export_active_learning_review_queue_report.py` so benchmark and CI
consumers can observe evidence richness in the active-learning review queue.

## Delivered

- Added evidence fields to normalized rows:
  - `evidence_count`
  - `evidence_sources`
  - `evidence_summary`
- Added evidence rollups to summary JSON:
  - `evidence_count_total`
  - `average_evidence_count`
  - `records_with_evidence_count`
  - `records_with_evidence_ratio`
  - `top_evidence_sources`
- Added evidence columns to exported CSV:
  - `evidence_count`
  - `evidence_sources`
  - `evidence_summary`

## Validation

```bash
python3 -m py_compile \
  scripts/export_active_learning_review_queue_report.py \
  tests/unit/test_export_active_learning_review_queue_report.py

flake8 \
  scripts/export_active_learning_review_queue_report.py \
  tests/unit/test_export_active_learning_review_queue_report.py \
  --max-line-length=100

pytest -q tests/unit/test_export_active_learning_review_queue_report.py
```

## Result

- `py_compile`: pass
- `flake8`: pass
- `pytest`: `2 passed`

## Notes

- Input compatibility is preserved for both raw `samples.jsonl` and exported
  `review_queue_*.csv` / `review_queue_*.jsonl`.
- Evidence fields default safely to empty values when older exports do not
  contain them.
