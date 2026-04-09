# Benchmark Scorecard Qdrant Readiness Validation 2026-03-08

## Goal

Extend `scripts/generate_benchmark_scorecard.py` so the benchmark scorecard can
consume the Qdrant readiness and observability fields that are already on
`main`, instead of treating vector readiness as an implicit side note.

The scorecard now reports at least:

- backend readiness
- indexed ratio
- unindexed vectors count
- scan truncation state

## Scope

Files changed:

- `scripts/generate_benchmark_scorecard.py`
- `tests/unit/test_generate_benchmark_scorecard.py`

## Design

### New optional input

- `--qdrant-readiness-summary`

Accepted payload shapes:

- a full `vectors_stats` style response containing `backend` plus
  `backend_health`
- a direct Qdrant backend-health object

### New scorecard component

- `components.qdrant_backend`

Reported fields include:

- `status`
- `backend`
- `readiness`
- `indexed_ratio`
- `unindexed_vectors_count`
- `scan_truncated`
- `readiness_hints`

### New scorecard behavior

- when Qdrant is `ready`, the scorecard exposes the backend summary without
  changing the existing happy-path result
- when Qdrant is `partial_scan`, `indexing`, `degraded`, `empty`, or
  `non_qdrant_backend`, the overall scorecard is downgraded to
  `benchmark_ready_with_qdrant_gap`
- recommendations now explicitly call out Qdrant scan coverage, indexing
  backfill, or health issues

## Test Coverage

Added or extended unit coverage for:

- happy path with a wrapped `backend_health` payload
- missing optional Qdrant input
- degraded benchmark result when direct Qdrant health shows partial scan and
  backfill

## Validation

```bash
python3 -m py_compile scripts/generate_benchmark_scorecard.py \
  tests/unit/test_generate_benchmark_scorecard.py

flake8 scripts/generate_benchmark_scorecard.py \
  tests/unit/test_generate_benchmark_scorecard.py \
  --max-line-length=100

pytest -q tests/unit/test_generate_benchmark_scorecard.py
```

## Validation Result

- `py_compile`: pass
- `flake8`: pass
- `pytest`: `4 passed`
- warning note: one existing `PendingDeprecationWarning` from `starlette`
  importing `multipart`; no code changes were made for that external warning

## Practical Outcome

The benchmark scorecard now reflects whether the vector backend is actually
ready for competitive claims, not just whether the classifier and governance
layers look good in isolation.
