# DEV_WEEK7_PHASE5_DOCS_CONSISTENCY_20251224

## Scope
- Fix error code table generation to handle annotated mappings.
- Regenerate error code reference docs.
- Validate metrics exports and dashboard references.

## Changes
- Update generator to parse annotated ERROR_SOURCE_MAPPING assignments.
- Regenerate `docs/ERROR_CODES.md` with mapped sources where available.

## Validation
- Command: `python3 scripts/generate_error_code_table.py`
  - Result: Updated `docs/ERROR_CODES.md` from source enum + mapping.
- Command: `python3 scripts/verify_metrics_export.py`
  - Result: `Metrics export verification passed.`
- Command: `python3 scripts/validate_dashboard_metrics.py`
  - Result: `Dashboard metrics all exported.`

## Notes
- Unmapped error codes still show `-` source; mapping can be expanded in `src/core/errors_extended.py` if needed.
