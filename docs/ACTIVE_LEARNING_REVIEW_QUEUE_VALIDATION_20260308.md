# Active Learning Review Queue Validation

Date: 2026-03-08

## Goal

Promote active learning from a simple pending list to an operator-friendly
review queue with:

- ranking by urgency
- filtering by queue slice
- pagination
- queue summary for dashboards and review tooling

## Design

### Core ranking logic

`src/core/active_learning.py` now provides `get_review_queue()` and ranks
samples by default using:

1. `feedback_priority`
2. `sample_type`
3. lower confidence first
4. older samples first

Default urgency order:

- `critical`
- `high`
- `medium`
- `normal`

Default sample type order:

- `knowledge_conflict`
- `branch_conflict`
- `hybrid_rejection`
- `low_confidence`
- `review`

### Filtering

Supported filters:

- `status`
- `sample_type`
- `feedback_priority`

### Sorting

Supported sort modes:

- `priority`
- `confidence`
- `created_at`

### API surface

New endpoint:

- `GET /api/v1/active-learning/review-queue`

Response contains:

- `total`
- `returned`
- `limit`
- `offset`
- `has_more`
- `sort_by`
- `summary`
- `items`

`summary` includes:

- `status`
- `total`
- `by_sample_type`
- `by_feedback_priority`

## Files Changed

- `src/core/active_learning.py`
- `src/api/v1/active_learning.py`
- `tests/test_active_learning_api.py`

## Validation

### Compile

```bash
python3 -m py_compile \
  src/core/active_learning.py \
  src/api/v1/active_learning.py \
  tests/test_active_learning_api.py
```

Result: pass

### Lint

```bash
flake8 \
  src/core/active_learning.py \
  src/api/v1/active_learning.py \
  tests/test_active_learning_api.py \
  --max-line-length=100
```

Result: pass

### Targeted tests

```bash
pytest -q tests/test_active_learning_api.py
```

Result: `8 passed`

## Outcome

The active-learning surface now supports a real review-operations workflow
instead of just a raw pending sample list.

This creates a clean next step for:

- review UI / admin dashboard
- ranked reviewer assignment
- active-learning export prioritization
- governance reporting
