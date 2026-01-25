# DEV_DXF_GRAPH_FEATURES_20260123

## Summary
- Extended DXF graph feature extraction to cover ARC/TEXT/MTEXT/DIMENSION/INSERT entities.
- Added richer node attributes (layer, color, text density, closed polyline) while preserving the legacy 9-dim layout.

## Implementation
- Introduced `DXF_NODE_FEATURES` and `DXF_NODE_FEATURES_LEGACY` schemas with a new default node dimension.
- Updated `_dxf_to_graph` to compute normalized geometry, layer/color, and text signals.
- Kept border/title block hints for LINE/LWPOLYLINE heuristics.

## Files
- `src/ml/train/dataset_2d.py`
- `src/ml/vision_2d.py`

## Notes
- Legacy layout is retained when `node_dim <= len(DXF_NODE_FEATURES_LEGACY)` for checkpoint compatibility.
