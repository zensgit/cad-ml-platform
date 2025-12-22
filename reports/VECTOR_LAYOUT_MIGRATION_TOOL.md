#!/usr/bin/env markdown
# Vector Layout Migration Tool Report

## Scope
- Add a safe migration script to reorder legacy vectors stored in Redis.
- Ensure layout conversion uses canonical order: base geometric + semantic + extensions.

## Changes
- Script: `scripts/migrate_vector_layout.py`
- Helper: `FeatureExtractor.reorder_legacy_vector()`
- Tests: `tests/unit/test_feature_vector_layout.py`

## Tests
- `.venv/bin/python -m pytest tests/unit/test_feature_vector_layout.py -q`
- Result: `3 passed in 4.16s`

## Verification
- Local dry-run output: `reports/VECTOR_LAYOUT_MIGRATION.md` (REDIS_URL not configured here).
- Production run should set `REDIS_URL` (and optionally `--assume-version`) then re-run.

## Usage (Prod)
```bash
REDIS_URL=redis://host:6379/0 .venv/bin/python scripts/migrate_vector_layout.py --dry-run 1
REDIS_URL=redis://host:6379/0 .venv/bin/python scripts/migrate_vector_layout.py --dry-run 0
```

## Notes
- Rebuild FAISS index after migration.
- Run similarity regression checks after migration.
