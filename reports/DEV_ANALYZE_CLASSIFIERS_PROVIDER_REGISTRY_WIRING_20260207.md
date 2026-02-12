# DEV_ANALYZE_CLASSIFIERS_PROVIDER_REGISTRY_WIRING_20260207

## Summary

Wired the DXF Graph2D and Hybrid classifier calls in the `/api/v1/analyze/` pipeline to use the core provider framework (`ProviderRegistry`) instead of directly importing ML singletons. This reduces coupling and centralizes optional-dependency handling.

## Changes

- `src/api/v1/analyze.py`
  - Graph2D path now uses `ProviderRegistry.get("classifier", "graph2d" | "graph2d_ensemble")`
  - Hybrid path now uses `ProviderRegistry.get("classifier", "hybrid")`
  - Keeps existing feature flags and behavior (best-effort, failure-safe, DXF-only).

## Why

- Avoid repeated direct imports / singleton access patterns spread across API code.
- Keep `torch` and other optional dependencies behind provider adapters.
- Align classifier wiring with existing OCR/Vision ProviderRegistry usage.

## Validation

Executed:

```bash
python3 -m pytest -q \
  tests/integration/test_analyze_dxf_hybrid_override.py \
  tests/integration/test_analyze_dxf_fusion.py \
  tests/integration/test_analyze_json_fusion.py
```

Result: PASS

