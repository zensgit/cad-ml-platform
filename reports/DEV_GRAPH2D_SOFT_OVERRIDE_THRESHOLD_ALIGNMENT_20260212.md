# DEV_GRAPH2D_SOFT_OVERRIDE_THRESHOLD_ALIGNMENT_20260212

## Goal
Prevent low-confidence / degenerate Graph2D predictions from producing misleading
`soft_override_suggestion` candidates in `/api/v1/analyze` DXF classification.

## Problem
`soft_override_suggestion.threshold` previously defaulted to a fixed `0.17`.
This could mark very low-confidence Graph2D outputs as "eligible" even when the
Graph2D pipeline itself was configured with a higher minimum confidence (e.g.
`GRAPH2D_MIN_CONF` from `config/hybrid_classifier.yaml`, default `0.5`).

## Change
- Align the default `GRAPH2D_SOFT_OVERRIDE_MIN_CONF` behavior with Graph2D's
  effective minimum confidence:
  - If env `GRAPH2D_SOFT_OVERRIDE_MIN_CONF` is set: use it (explicit override).
  - Else: default to `graph2d_prediction.min_confidence` (which itself is
    sourced from Hybrid config when env is absent).

## Implementation
- Updated `src/api/v1/analyze.py` to derive `soft_override_suggestion.threshold`
  from `graph2d_prediction.min_confidence` by default (instead of hard-coded
  `0.17`).
- Added an integration test to lock the behavior:
  `tests/integration/test_analyze_dxf_graph2d_prediction_contract.py` asserts
  the default threshold becomes `0.5` and `confidence=0.49` is ineligible.

## Validation
- `pytest tests/integration/test_analyze_dxf_graph2d_prediction_contract.py -v`
  - Result: `5 passed`
- `make validate-core-fast`
  - Result: passed (tolerance validators + openapi + service-mesh + provider tests)

## Files Changed
- `src/api/v1/analyze.py`
- `tests/integration/test_analyze_dxf_graph2d_prediction_contract.py`

