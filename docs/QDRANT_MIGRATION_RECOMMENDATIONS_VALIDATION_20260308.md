# Qdrant Migration Recommendations Validation - 2026-03-08

## Goal

Add simple phased-rollout guidance to vector migration pending summary so operators can choose the next source feature version to migrate.

## Changes

Updated [src/api/v1/vectors.py](/private/tmp/cad-ml-platform-vectors-migrate-recommendations-20260308-003200/src/api/v1/vectors.py) with additional `GET /api/v1/vectors/migrate/pending/summary` fields:

- `recommended_from_versions`
- `largest_pending_from_version`
- `largest_pending_count`

Behavior:

- recommendation order sorts observed source versions by pending count descending, then version name ascending
- filtered summaries only recommend within the filtered version subset

Added coverage in:

- [test_vector_migration_pending_summary.py](/private/tmp/cad-ml-platform-vectors-migrate-recommendations-20260308-003200/tests/unit/test_vector_migration_pending_summary.py)

## Validation

Commands:

```bash
python3 -m py_compile src/api/v1/vectors.py tests/unit/test_vector_migration_pending_summary.py
flake8 src/api/v1/vectors.py tests/unit/test_vector_migration_pending_summary.py --max-line-length=100
pytest -q tests/unit/test_vector_migration_pending_summary.py
```

Result:

- `py_compile`: pass
- `flake8`: pass
- `pytest`: `4 passed`
