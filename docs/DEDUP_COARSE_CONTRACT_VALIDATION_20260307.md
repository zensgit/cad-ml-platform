# Dedup Coarse Contract Validation 2026-03-07

## Goal

Add stable coarse/fine semantic contract fields to 2D dedup match results without
changing the existing duplicate/similar workflow.

Target surface:

- `POST /api/v1/dedup/2d/search`
- async job results that reuse `run_dedup_2d_pipeline()`

## Changes

Files:

- `src/core/dedupcad_2d_pipeline.py`
- `src/api/v1/dedup.py`
- `tests/test_dedup_2d_proxy.py`

Behavior:

- Dedup match items now expose:
  - `fine_part_type`
  - `coarse_part_type`
  - `decision_source`
  - `is_coarse_label`
- Contracts are derived from match `file_name` using the existing
  `FilenameClassifier`, then normalized through the shared coarse-label mapper.
- Enrichment is applied inside the shared dedup pipeline, so both sync and async
  search paths return the same contract.

## Validation

Commands:

```bash
python3 -m py_compile \
  src/core/dedupcad_2d_pipeline.py \
  src/api/v1/dedup.py \
  tests/test_dedup_2d_proxy.py

flake8 \
  src/core/dedupcad_2d_pipeline.py \
  src/api/v1/dedup.py \
  tests/test_dedup_2d_proxy.py \
  --max-line-length=100

pytest -q \
  tests/test_dedup_2d_proxy.py \
  tests/contract/test_openapi_schema_snapshot.py
```

Results:

- `py_compile`: pass
- `flake8`: pass
- `pytest`: `31 passed`
- OpenAPI snapshot: unchanged

## Notes

- This change is additive. Existing `duplicates` and `similar` payload fields are
  unchanged.
- Contract quality is bounded by filename quality; unmatched file names leave the
  new fields as `null`.
