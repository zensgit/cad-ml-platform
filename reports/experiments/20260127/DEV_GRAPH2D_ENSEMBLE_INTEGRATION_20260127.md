# DEV_GRAPH2D_ENSEMBLE_INTEGRATION_20260127

## Goal
Wire the new `EnsembleGraph2DClassifier` into the `/api/v1/analyze` pipeline behind a feature flag, without changing existing single-model behavior.

## Design
- New flag: `GRAPH2D_ENSEMBLE_ENABLED=true|false` (default: false)
- When enabled:
  - Use `get_ensemble_2d_classifier()` instead of `get_2d_classifier()`
  - Preserve existing Graph2D allow/exclude/drawing-type gating
  - Attach `ensemble_enabled` to the Graph2D payload for observability
- Reporting:
  - Add ensemble-related columns to `scripts/batch_analyze_dxf_local.py`
    - `graph2d_ensemble_enabled`
    - `graph2d_ensemble_size`
    - `graph2d_voting`

## Code Changes
- `src/api/v1/analyze.py`
  - Switchable single vs ensemble classifier selection via `GRAPH2D_ENSEMBLE_ENABLED`
  - Graph2D payload now includes `ensemble_enabled`
- `scripts/batch_analyze_dxf_local.py`
  - Added ensemble metadata columns to batch CSV output

## How to Use
```bash
# Enable Graph2D + ensemble
export GRAPH2D_ENABLED=true
export GRAPH2D_ENSEMBLE_ENABLED=true

# Optional: override ensemble model list
# export GRAPH2D_ENSEMBLE_MODELS="models/graph2d_edge_sage_v3.pth,models/graph2d_edge_sage_v4_best.pth,models/graph2d_edge_sage_v5_warmup.pth"

# Run a local smoke batch
DISABLE_MODEL_SOURCE_CHECK=1 \
  .venv-graph/bin/python scripts/batch_analyze_dxf_local.py \
    --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
    --max-files 5 \
    --seed 20260127 \
    --output-dir "reports/experiments/20260127/dxf_batch_analysis_ensemble_smoke_20260127_r2"
```

## Notes
- Current ensemble models (v3/v4/v5) share the same 26-label map focused on drawing types and legacy categories. This ensemble should remain a drawing-type signal until a part-type-aligned model is available.
