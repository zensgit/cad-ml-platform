# Active Learning Review Queue Stats Validation

Date: 2026-03-08
Branch: `feat/active-learning-review-queue-stats`
Status: validated locally

## Goal

Extend the active-learning review queue with queue-level observability so review
operations can see:

- counts by sample type
- counts by feedback priority
- counts by decision source
- counts by uncertainty reason
- counts by review reason

The capability is exposed both in `/api/v1/active-learning/review-queue` summary
and a dedicated `/api/v1/active-learning/review-queue/stats` endpoint.

## Scope

Updated:

- `src/core/active_learning.py`
- `src/api/v1/active_learning.py`
- `tests/test_active_learning_api.py`
- `config/openapi_schema_snapshot.json`

## Validation

### Static checks

```bash
python3 -m py_compile \
  src/core/active_learning.py \
  src/api/v1/active_learning.py \
  tests/test_active_learning_api.py
```

```bash
flake8 \
  src/core/active_learning.py \
  src/api/v1/active_learning.py \
  tests/test_active_learning_api.py \
  --max-line-length=100
```

### API tests

```bash
pytest -q tests/test_active_learning_api.py
```

Result:

- `10 passed`

### OpenAPI contract

```bash
python3 scripts/ci/generate_openapi_schema_snapshot.py \
  --output config/openapi_schema_snapshot.json
make validate-openapi
```

Result:

- snapshot refreshed
- `5 passed`

## Outcome

- Review queue now exposes benchmark-friendly operational summaries.
- Queue list and queue stats share the same filtered summary builder.
- Decision-source and reason aggregations can feed review dashboards and
  benchmark/reporting flows directly.
