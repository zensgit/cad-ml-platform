# DEV_GRAPH2D_UNAVAILABLE_PREDICTION_ATTACHMENT_20260212

## Goal
Make `/api/v1/analyze` DXF classification output more debuggable by always
including a `graph2d_prediction` payload when Graph2D is enabled, even when the
Graph2D model is unavailable (e.g. missing `torch` / missing checkpoints).

## Change
- Always attach `results.classification.graph2d_prediction` when Graph2D returns
  a dict payload (including `status=model_unavailable`).
- Always include Graph2D gate metadata in that payload:
  - `min_confidence` (defaults from `config/hybrid_classifier.yaml` when env absent)
  - `min_margin` (defaults from `config/hybrid_classifier.yaml` when env absent)
  - `ensemble_enabled`

This keeps downstream policy outputs consistent, especially:
- `soft_override_suggestion.threshold` defaulting to the effective Graph2D
  `min_confidence` even when the model is unavailable.

## Implementation
- `src/api/v1/analyze.py`
  - Refactored the Graph2D block to compute and attach gate metadata for all
    dict results.
  - Preserved existing allow/exclude/drawing/coarse + threshold/margin checks
    for the normal (`status!=model_unavailable`) path.
- `tests/integration/test_analyze_dxf_graph2d_prediction_contract.py`
  - Added a contract test asserting `graph2d_prediction.status=model_unavailable`
    is still attached and `soft_override_suggestion` reports `graph2d_unavailable`
    with threshold `0.5` (Hybrid config default).

## Validation
- `.venv/bin/python -m pytest tests/integration/test_analyze_dxf_graph2d_prediction_contract.py -v`
  - Result: `6 passed`
- `make validate-core-fast`
  - Result: passed

## Files Changed
- `src/api/v1/analyze.py`
- `tests/integration/test_analyze_dxf_graph2d_prediction_contract.py`

