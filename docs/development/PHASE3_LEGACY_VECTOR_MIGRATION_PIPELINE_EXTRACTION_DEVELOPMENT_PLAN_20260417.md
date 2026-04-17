# Phase 3 Legacy Vector Migration Pipeline Extraction Development Plan

## Goal

Extract the legacy analyze-route vector migration orchestration from
`src/api/v1/analyze.py` into a shared helper while preserving:

- `/api/v1/analyze/vectors/migrate`
- `/api/v1/analyze/vectors/migrate/status`

behavior, response shape, and migration-history bookkeeping.

## Scope

### In

- add `src/core/legacy_vector_migration_pipeline.py`
- move legacy analyze-route migrate execution logic into the helper
- move legacy analyze-route migration status assembly into the helper
- keep analyze routes as thin delegates
- add helper-level and route-level regression coverage

### Out

- `/api/v1/vectors/migrate*` primary route family
- migration summary / pending / plan endpoints
- migration algorithm changes

## Design

Create:

- `run_legacy_vector_migrate_pipeline(payload=...)`
- `run_legacy_vector_migration_status_pipeline()`

The helper keeps:

- vector existence checks against in-memory legacy store
- cache-backed CAD document reconstruction
- feature re-extraction using target feature version
- dry-run / migrated / skipped / error item assembly
- `_MIGRATION_STATUS` ring-buffer bookkeeping

`analyze.py` keeps only response-model construction.

## Risk Controls

- preserve legacy behavior instead of forcing the richer `/api/v1/vectors`
  migration semantics onto deprecated analyze routes
- preserve `_MIGRATION_STATUS` history shape
- validate helper behavior directly with stateful unit tests
- validate route-level delegation with integration monkeypatch tests

## Validation Plan

1. `python3 -m py_compile src/core/legacy_vector_migration_pipeline.py src/api/v1/analyze.py tests/unit/test_legacy_vector_migration_pipeline.py tests/integration/test_analyze_legacy_vector_migration_pipeline.py`
2. `.venv311/bin/flake8 src/core/legacy_vector_migration_pipeline.py src/api/v1/analyze.py tests/unit/test_legacy_vector_migration_pipeline.py tests/integration/test_analyze_legacy_vector_migration_pipeline.py`
3. `.venv311/bin/python -m pytest -q tests/unit/test_legacy_vector_migration_pipeline.py tests/integration/test_analyze_legacy_vector_migration_pipeline.py tests/unit/test_vector_migration_status.py tests/unit/test_vector_migrate_response_fields.py tests/unit/test_vector_migration_history.py tests/unit/test_vector_migration_counts_history.py tests/unit/test_analysis_error_handling.py tests/integration/test_analyze_error_handling.py tests/unit/test_analysis_ocr_pipeline.py tests/integration/test_analyze_ocr_pipeline.py tests/unit/test_analysis_result_envelope.py tests/integration/test_analyze_result_envelope.py tests/unit/test_analysis_preflight.py tests/unit/test_document_pipeline.py tests/test_api_integration.py`
