# Vector Backend Reload Reason Labels Design

## Overview
Add reason labels to vector backend reload metrics so failures are classified by
cause (invalid backend, auth failure, initialization error).

## Updates
- Added `reason` label to `vector_store_reload_total`.
- Recorded `auth_failed` for admin token failures on vector reload endpoint.
- Labeled invalid backend and initialization errors in reload handlers.
- Updated tests to validate new labels where metrics are available.

## Files
- `src/utils/analysis_metrics.py`
- `src/api/v1/maintenance.py`
- `src/api/v1/vectors.py`
- `tests/unit/test_backend_reload_failures.py`
- `tests/unit/test_vector_backend_reload_failure.py`
- `tests/unit/test_maintenance_api_coverage.py`
