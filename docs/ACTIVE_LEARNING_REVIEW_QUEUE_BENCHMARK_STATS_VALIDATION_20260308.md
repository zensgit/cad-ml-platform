# Active Learning Review Queue Benchmark Stats Validation 2026-03-08

## Goal

Make the review queue stats API directly useful for benchmark and operational
readiness decisions.

## Changes

Extended `GET /api/v1/active-learning/review-queue/stats` with stable fields:

- `critical_count`
- `high_count`
- `automation_ready_count`
- `critical_ratio`
- `high_ratio`
- `automation_ready_ratio`
- `operational_status`

`operational_status` is derived as:

- `under_control`
- `critical_backlog`
- `managed_backlog`
- `routine_backlog`

## Files

- `src/core/active_learning.py`
- `src/api/v1/active_learning.py`
- `tests/test_active_learning_api.py`

## Validation

Commands run:

```bash
python3 -m py_compile src/core/active_learning.py src/api/v1/active_learning.py tests/test_active_learning_api.py
flake8 src/core/active_learning.py src/api/v1/active_learning.py tests/test_active_learning_api.py --max-line-length=100
pytest -q tests/test_active_learning_api.py -k review_queue_stats_endpoint
```

