# DEV_ANALYZE_PART_CLASSIFIER_PROVIDER_SHADOW_20260208

## Goal
Add a **shadow-only** wiring path so `/api/v1/analyze/` can optionally run the in-process PartClassifier via the provider framework without changing the primary decision path (Fusion/Hybrid/rules).

This is meant for safe evaluation of the "CAD part/component" classifier while keeping production behavior unchanged by default.

## Changes
- `src/api/v1/analyze.py`
  - Added an optional provider invocation for DXF/DWG when enabled.
  - Writes upload bytes to a temporary file and calls `ProviderRegistry.get("classifier", <name>).process(...)`.
  - Adds an additive field to the classification payload:
    - `classification.part_classifier_prediction`
  - Feature flags:
    - `PART_CLASSIFIER_PROVIDER_ENABLED` (default: `false`)
    - `PART_CLASSIFIER_PROVIDER_NAME` (default: `v16`)
  - Does **not** override `part_type` (shadow only).

- `tests/integration/test_analyze_dxf_fusion.py`
  - Added an integration test that registers a stub classifier provider and asserts the additive field is present when the feature flag is enabled.

- `tests/conftest.py`
  - Added the new env vars to the test env isolation list.

## Verification
Commands:
```bash
.venv/bin/python -m pytest -q tests/integration/test_analyze_dxf_fusion.py::test_analyze_dxf_adds_part_classifier_prediction_when_enabled
.venv/bin/python -m pytest -q tests/integration/test_analyze_dxf_fusion.py
.venv/bin/python -m pytest -q tests/unit/test_provider_framework.py
```

Result:
- All tests passed locally.
