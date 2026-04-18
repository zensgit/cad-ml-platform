# Phase 3 Analyze Process Audit Compat Development Plan

## Goal
- remove the duplicate `process_rules_audit` implementation from `src/api/v1/analyze.py`
- preserve the existing `src.api.v1.analyze.process_rules_audit` import surface as a compatibility alias

## Scope
- switch `src/api/v1/analyze.py` to reuse `src.api.v1.process.process_rules_audit`
- remove analyze-only imports that existed only for the duplicated implementation
- add focused smoke coverage for the compat alias

## Risk Controls
- keep the old `/api/v1/analyze/process/rules/audit` route behavior unchanged by preserving the formal implementation in `src/api/v1/process.py`
- preserve `src.api.v1.analyze.process_rules_audit` import compatibility for any remaining patch points
- do not change response models, metrics, or process rules loading behavior

## Validation Plan
- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile ...`
- `.venv311/bin/flake8 ...`
- `.venv311/bin/python -m pytest -q` on process audit endpoint, route uniqueness, and compat smoke coverage
