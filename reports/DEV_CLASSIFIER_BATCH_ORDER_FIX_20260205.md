# DEV_CLASSIFIER_BATCH_ORDER_FIX_20260205

## Summary
- Fixed `/classify/batch` to preserve input order even when filenames repeat.
- Added a regression test for duplicate filenames and documented the ordering guarantee.

## Changes
- `src/inference/classifier_api.py`: track batch results by index instead of filename to avoid collisions.
- `tests/unit/test_classifier_api_cache.py`: add duplicate-filename ordering test.
- `docs/cad_classifier_edge_cases.md`: clarify batch order semantics.

## Validation
- `python3 -m pytest tests/unit/test_classifier_api_cache.py -q` (pass).
- `python3 -m pytest tests/unit -q` (fails with many existing failures early in the run; stopped after confirming failures).
