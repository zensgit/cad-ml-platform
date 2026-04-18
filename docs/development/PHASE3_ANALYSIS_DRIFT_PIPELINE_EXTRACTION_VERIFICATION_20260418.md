# Phase 3 Analysis Drift Pipeline Extraction Verification

## Implemented
- added shared helper: `src/core/analysis_drift_pipeline.py`
- updated `src/api/v1/analyze.py` to delegate per-analysis drift updates to the shared helper
- added helper unit coverage: `tests/unit/test_analysis_drift_pipeline.py`

## Validation
- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile src/core/analysis_drift_pipeline.py src/api/v1/analyze.py tests/unit/test_analysis_drift_pipeline.py`
- `.venv311/bin/flake8 src/core/analysis_drift_pipeline.py src/api/v1/analyze.py tests/unit/test_analysis_drift_pipeline.py`
- `.venv311/bin/python -m pytest -q tests/unit/test_analysis_drift_pipeline.py tests/unit/test_drift_endpoint_coverage.py tests/unit/test_drift_auto_refresh.py`

## Result
- drift recording logic is centralized without changing `_DRIFT_STATE` compatibility
- analyze route behavior stays the same while dropping the inline drift block
- helper and drift state regressions passed
- targeted pytest result: `48 passed, 7 warnings`
