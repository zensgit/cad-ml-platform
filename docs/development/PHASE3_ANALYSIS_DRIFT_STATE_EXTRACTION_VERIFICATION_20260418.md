# Phase 3 Analysis Drift State Extraction Verification

## Implemented
- added shared state module: `src/core/analysis_drift_state.py`
- updated `src/api/v1/analyze.py` to re-export the shared drift state via `_DRIFT_STATE`
- added shared state smoke coverage: `tests/unit/test_analysis_drift_state.py`

## Validation
- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile src/core/analysis_drift_state.py src/api/v1/analyze.py tests/unit/test_analysis_drift_state.py tests/unit/test_analysis_drift_pipeline.py tests/unit/test_drift_auto_refresh.py tests/unit/test_drift_startup_trigger.py tests/unit/test_drift_endpoint_coverage.py`
- `.venv311/bin/flake8 src/core/analysis_drift_state.py src/api/v1/analyze.py tests/unit/test_analysis_drift_state.py tests/unit/test_analysis_drift_pipeline.py tests/unit/test_drift_auto_refresh.py tests/unit/test_drift_startup_trigger.py tests/unit/test_drift_endpoint_coverage.py`
- `.venv311/bin/python -m pytest -q tests/unit/test_analysis_drift_state.py tests/unit/test_analysis_drift_pipeline.py tests/unit/test_drift_auto_refresh.py tests/unit/test_drift_startup_trigger.py tests/unit/test_drift_endpoint_coverage.py`

## Result
- drift state ownership is centralized without changing `src.api.v1.analyze._DRIFT_STATE` compatibility
- drift startup, refresh, and endpoint regressions continue to pass against the shared object
- targeted pytest result: `42 passed, 7 warnings`
