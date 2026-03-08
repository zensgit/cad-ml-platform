# Active Learning Review Queue Report Export Validation 2026-03-08

## Goal

Provide a low-conflict offline script that turns active-learning review queue
records into a benchmark-consumable JSON summary.

## Supported inputs

- raw active-learning `samples.jsonl`
- exported review queue `csv`
- exported review queue `jsonl`

## Output

Stable summary fields include:

- `total`
- `by_sample_type`
- `by_feedback_priority`
- `by_decision_source`
- `by_review_reason`
- `critical_count`
- `high_count`
- `automation_ready_count`
- `critical_ratio`
- `high_ratio`
- `automation_ready_ratio`
- `operational_status`
- `top_sample_types`
- `top_feedback_priorities`
- `top_decision_sources`
- `top_review_reasons`

## Files

- `scripts/export_active_learning_review_queue_report.py`
- `tests/unit/test_export_active_learning_review_queue_report.py`

## Validation

Commands run:

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

