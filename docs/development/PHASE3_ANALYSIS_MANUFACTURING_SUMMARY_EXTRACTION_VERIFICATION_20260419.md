# Phase 3 Analyze Manufacturing Summary Extraction Verification

## Summary
- `manufacturing_decision` attachment was moved from `src/api/v1/analyze.py` into `src/core/analysis_manufacturing_summary.py`.
- `analyze.py` now delegates the post-processing step while still passing the route-local `build_manufacturing_decision_summary` alias.

## Files
- `src/core/analysis_manufacturing_summary.py`
- `src/api/v1/analyze.py`
- `tests/unit/test_analysis_manufacturing_summary.py`

## Validation
- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile ...`
- `.venv311/bin/flake8 ...`
- `.venv311/bin/python -m pytest -q ...`

## Expected Outcome
- No response contract change
- Existing integration monkeypatch coverage continues to pass
- Failure path remains warning-only and non-fatal
