# Phase 3 Analyze Drift Attachment Extraction Verification

## Summary
- The `analyze.py` drift attachment wrapper was moved into `src/core/analysis_drift_attachment.py`.
- `analyze.py` now delegates the drift cache lookup and failure-swallowing wrapper while still passing the route-local drift pipeline alias and shared drift state.

## Files
- `src/core/analysis_drift_attachment.py`
- `src/api/v1/analyze.py`
- `tests/unit/test_analysis_drift_attachment.py`

## Validation
- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile ...`
- `.venv311/bin/flake8 ...`
- `.venv311/bin/python -m pytest -q ...`

## Expected Outcome
- No route contract change
- Drift attachment remains non-fatal on cache/pipeline failures
- Existing analyze compatibility surfaces remain intact
