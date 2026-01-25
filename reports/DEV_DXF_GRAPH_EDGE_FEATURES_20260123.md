# DEV_DXF_GRAPH_EDGE_FEATURES_20260123

## Summary
- Added edge feature extraction for DXF graphs and an edge-aware GraphSAGE classifier.

## Updates
- Dataset: `src/ml/train/dataset_2d.py` now exposes `DXF_EDGE_FEATURES` (7 dims) and optional `edge_attr` output.
- Model: `src/ml/train/model_2d.py` adds `EdgeGraphSageClassifier` with edge-conditioned message passing.
- Training/Eval: `scripts/train_2d_graph.py` and `scripts/eval_2d_graph.py` support `--model edge_sage` and edge attributes.
- Runtime: `src/ml/vision_2d.py` loads edge-aware checkpoints and passes edge attributes.

## Edge Feature Schema
- dx_norm, dy_norm, dist_norm, dir_dot, layer_diff, color_diff, same_type
