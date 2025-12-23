# Week6 Step3 - Vector Migration Batch Runner (2025-12-22)

## Scope
- Batch migration runner using `/api/v1/vectors/migrate/preview` + `/api/v1/vectors/migrate`.

## Test Run
- `python3 scripts/vector_migration_batch.py --base-url http://localhost:8000 --to-version v4`

## Report Output
- `reports/DEV_VECTOR_MIGRATION_BATCH_20251222.md`

## Result
- Dry-run completed against in-memory vectors; preview and migration summaries captured.
