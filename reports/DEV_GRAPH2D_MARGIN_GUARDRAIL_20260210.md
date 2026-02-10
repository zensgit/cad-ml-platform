# DEV_GRAPH2D_MARGIN_GUARDRAIL_20260210

## Goal
Add a lightweight guardrail so HybridClassifier does not accept ambiguous Graph2D predictions
in anonymous DXF scenarios.

Ambiguity is measured using:
- `margin = top1_prob - top2_prob` (from Graph2D prediction payload)

This is orthogonal to `GRAPH2D_MIN_CONF`:
- `GRAPH2D_MIN_CONF` filters by absolute top-1 probability
- `GRAPH2D_MIN_MARGIN` filters by separation between top-1 and top-2

## Changes
- Added `graph2d.min_margin` to the Hybrid config model:
  - `src/ml/hybrid_config.py`
  - Env override: `GRAPH2D_MIN_MARGIN`
- HybridClassifier now filters Graph2D fine labels when margin is below the threshold:
  - `src/ml/hybrid_classifier.py`
  - Sets:
    - `graph2d_prediction.filtered=true`
    - `graph2d_prediction.filtered_reason=below_min_margin`
    - `graph2d_prediction.min_margin_effective=<value>`
- `.env.example` now includes `GRAPH2D_MIN_MARGIN`

Default behavior:
- `GRAPH2D_MIN_MARGIN=0.0` keeps existing behavior unchanged.

## Analyze Endpoint Behavior
`/api/v1/analyze` now also surfaces and applies the same guardrail fields on the
Graph2D payload:
- `graph2d_prediction.min_margin`
- `graph2d_prediction.passed_margin`

This is used to:
- block `soft_override_suggestion` eligibility when `passed_margin=false`
- prevent Graph2D from becoming a fusable L4 signal when `passed_margin=false`
  (for `GRAPH2D_FUSION_ENABLED` / FusionAnalyzer paths)

## Verification
Unit tests added:
- `tests/unit/test_hybrid_classifier_graph2d_margin_guardrail.py`
  - margin below threshold is ignored
  - margin above threshold is accepted

Integration test added (analyze contract):
- `tests/integration/test_analyze_dxf_graph2d_prediction_contract.py`
  - low-margin Graph2D blocks `soft_override_suggestion` with reason `below_margin`

Example run:
```bash
python3 -m pytest tests/unit/test_hybrid_classifier_graph2d_margin_guardrail.py -q
```

## Ops Notes
Recommended usage for many-class Graph2D checkpoints:
- Start with `GRAPH2D_MIN_MARGIN=0.01` and iterate based on a calibration set.
- Use together with temperature scaling (`GRAPH2D_TEMPERATURE` or `..._CALIBRATION_PATH`).
