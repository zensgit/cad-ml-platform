# Phase 3 Analyze Shadow Compat Extraction Verification

## Implemented
- added compat module: `src/api/v1/analyze_shadow_compat.py`
- updated `src/api/v1/analyze.py` to re-export the shadow helpers via the compat module
- added compat smoke coverage: `tests/unit/test_analyze_shadow_compat.py`

## Validation
- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile src/api/v1/analyze.py src/api/v1/analyze_shadow_compat.py tests/unit/test_analyze_shadow_compat.py tests/unit/test_analyze_graph2d_gate_helpers.py tests/unit/test_analyze_history_sequence_resolution.py`
- `.venv311/bin/flake8 src/api/v1/analyze.py src/api/v1/analyze_shadow_compat.py tests/unit/test_analyze_shadow_compat.py tests/unit/test_analyze_graph2d_gate_helpers.py tests/unit/test_analyze_history_sequence_resolution.py`
- `.venv311/bin/python -m pytest -q tests/unit/test_analyze_shadow_compat.py tests/unit/test_analyze_graph2d_gate_helpers.py tests/unit/test_analyze_history_sequence_resolution.py tests/unit/test_classification_shadow_pipeline.py`

## Result
- the shadow helper export surface is centralized without changing `src.api.v1.analyze` compatibility
- helper behavior remains owned by `src/core/classification/shadow_pipeline.py`
- compat smoke tests and existing analyze/shadow regressions passed
- targeted pytest result: `29 passed, 7 warnings`
