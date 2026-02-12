# DEV_GRAPH2D_PREDICTION_CONTRACT_REGRESSION_20260208

## Goal
Add regression coverage for Graph2D + Hybrid behavior that does not require manual drawing review:
- API contract for `classification.graph2d_prediction` fields (allow/exclude/drawing/coarse flags).
- Golden HybridClassifier cases that inject a `graph2d_result` to validate guardrails and fallback.

## Changes
- `tests/integration/test_analyze_dxf_graph2d_prediction_contract.py`
  - Verifies API `graph2d_prediction` fields are set consistently:
    - `min_confidence` defaults to Hybrid config when `GRAPH2D_MIN_CONF` is absent
    - `is_drawing_type` and `is_coarse_label` flags
    - `excluded` behavior for `other`

- `tests/golden/golden_dxf_hybrid_cases.json`
  - Added cases that include `graph2d_result` to exercise:
    - non-matching Graph2D does not override filename label
    - Graph2D-only fallback path when filename/titleblock are absent

- `tests/unit/test_golden_dxf_hybrid_manifest.py`
  - Passes optional `graph2d_result` from golden cases into `HybridClassifier.classify(...)`.

- `scripts/eval_golden_dxf_hybrid.py`
  - Same as unit test: supports golden cases with `graph2d_result`.

## Verification
Commands:
```bash
python3 -m pytest -q tests/integration/test_analyze_dxf_graph2d_prediction_contract.py
python3 -m pytest -q tests/unit/test_golden_dxf_hybrid_manifest.py
```

Result:
- `3 passed`
- `1 passed`

