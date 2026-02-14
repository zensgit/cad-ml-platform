# DEV_GRAPH2D_NODE_DIM_EXTRA_FEATURES_20260214

## Goal

Enable safe experimentation with richer DXF node features for Graph2D without breaking existing checkpoints:

- Keep the **default node schema** unchanged (`DXF_NODE_DIM=19`).
- Allow training with larger `--node-dim` to append additional, deterministic geometry features.
- Persist feature schemas in checkpoints for easier debugging.

## Changes

### 1) Optional extra node features (appended)

Updated `src/ml/train/dataset_2d.py`:

- Added `DXF_NODE_FEATURES_EXTRA_V1` (appended when `node_dim > 19`):
  - `bbox_w_norm`: entity bbox width / drawing `max_dim`
  - `bbox_h_norm`: entity bbox height / drawing `max_dim`
  - `arc_sweep_norm`: arc sweep ratio in `[0,1]` (`delta / 2pi`), circle uses `1.0`
  - `polyline_vertex_norm`: `min(vertex_count, 64) / 64`
- Implemented a tighter ARC bbox by including cardinal points within the sweep when computing bbox width/height.

Backward compatibility:

- Existing checkpoints store `node_dim`; inference requests that exact dimensionality, so old models remain unaffected.
- When `node_dim == 19`, the extra features are computed but not appended to the final feature tensor.

### 2) Pipeline wiring for `--node-dim`

Updated `scripts/run_graph2d_pipeline_local.py`:

- Added `--node-dim` (default `19`).
- Passed through to `scripts/train_2d_graph.py --node-dim ...`.
- Recorded `node_dim` in `pipeline_summary.json`.

### 3) Persist schemas in checkpoints

Updated `scripts/train_2d_graph.py`:

- Added `node_schema` and `edge_schema` to the saved checkpoint payload for debugging:
  - `node_schema` is derived from `DXF_NODE_FEATURES` (+ extras when applicable).
  - `edge_schema` is derived from `DXF_EDGE_FEATURES` when `--model edge_sage`.

## Verification

### Unit tests

- `.venv/bin/python -m pytest tests/unit/test_dataset2d_node_extra_features.py -v`
- `.venv/bin/python -m pytest tests/unit/test_dataset2d_enhanced_keypoints.py tests/unit/test_dataset2d_eps_scale.py tests/unit/test_dataset2d_edge_augment_strategy.py tests/unit/test_dataset2d_edge_augment_knn.py -v`
- `.venv/bin/python -m pytest tests/unit/test_run_graph2d_pipeline_local_distill_wiring.py tests/unit/test_run_graph2d_pipeline_local_diagnose_strict_wiring.py -v`

### Strict-mode pipeline experiments (DXF corpus: 110 files)

Corpus:

- `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf`

Common flags:

- `--student-geometry-only`
- `--diagnose-no-text-no-filename`
- `--normalize-labels --clean-min-count 5`
- `--distill --teacher titleblock --distill-alpha 0.1`
- `--dxf-enhanced-keypoints true`
- `--dxf-edge-augment-knn-k 0`
- `--dxf-eps-scale 0.001`

Runs:

1) Baseline (node_dim=19)

- Command:
  - `.venv/bin/python scripts/run_graph2d_pipeline_local.py ... --node-dim 19`
- Strict diagnose accuracy: `0.2364`
- Artifacts: `/tmp/graph2d_pipeline_local_20260214_183427`

2) Extra features enabled (node_dim=23)

- Command:
  - `.venv/bin/python scripts/run_graph2d_pipeline_local.py ... --node-dim 23`
- Strict diagnose accuracy: `0.1455` (regressed vs baseline under this quick config)
- Artifacts: `/tmp/graph2d_pipeline_local_20260214_183608`

## Conclusion

- The repo now supports **feature-safe** Graph2D node-dimension experiments via `--node-dim`, and checkpoints record the feature schemas.
- On this corpus/config (3 epochs, titleblock distill), `node_dim=23` did **not** improve strict accuracy; keep `node_dim=19` as the default and revisit extra-feature models with longer training or tuned hyperparameters.

