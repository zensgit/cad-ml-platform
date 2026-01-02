# Full Test Run (Re-run) (2026-01-01)

## Scope

- Execute `make test` after resolving Prometheus metric registry duplication.

## Fix Applied

- Updated `src/core/dedup2d_metrics.py` to reuse existing Prometheus collectors when the
  same metric name is registered multiple times during test imports.

## Command

- `make test`

## Results

- OK: `3991 passed, 20 skipped, 3 warnings`.
- Warnings: SWIG-related DeprecationWarning from vision integration tests.
