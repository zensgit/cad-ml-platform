# Active Learning Review Queue Export Validation 20260308

## Goal
- Export the ranked review queue with coarse/fine labels, decision source, review reasons, and score context.
- Support benchmark curation and human review without changing model logic.

## Design
- Added `ActiveLearner.export_review_queue(...)`.
- Added `GET /api/v1/active-learning/review-queue/export`.
- Supported formats:
  - `csv`
  - `jsonl`
- Export keeps:
  - queue ordering
  - queue summary
  - decision source
  - review reasons
  - coarse/fine prediction context

## Files
- `src/core/active_learning.py`
- `src/api/v1/active_learning.py`
- `tests/test_active_learning_api.py`

## Validation
```bash
python3 -m py_compile \
  src/core/active_learning.py \
  src/api/v1/active_learning.py \
  tests/test_active_learning_api.py

flake8 \
  src/core/active_learning.py \
  src/api/v1/active_learning.py \
  tests/test_active_learning_api.py \
  --max-line-length=100

pytest -q tests/test_active_learning_api.py
python3 scripts/ci/generate_openapi_schema_snapshot.py --output config/openapi_schema_snapshot.json
make validate-openapi
```
