# DEV_GRAPH2D_EPS_SCALE_SWEEP_20260214

## Goal

Make the epsilon-adjacency distance used by Graph2D DXF graph construction configurable, then validate whether increasing it improves strict-mode (geometry-only) performance.

## Change Summary

New env var:

- `DXF_EPS_SCALE` (float, default `0.001`)
  - Graph-build epsilon becomes: `eps = max(1e-3, max_dim * DXF_EPS_SCALE)`
  - Value is clamped to `[1e-6, 0.05]` to avoid accidental over-connection.

Wiring:

- `src/ml/train/dataset_2d.py`: uses `DXF_EPS_SCALE` for epsilon-adjacency and includes it in manifest-graph cache key.
- `scripts/audit_graph2d_strict_graph_quality.py`: uses and records `DXF_EPS_SCALE`.
- `scripts/run_graph2d_pipeline_local.py`: adds `--dxf-eps-scale` and sets `DXF_EPS_SCALE`.
- `scripts/train_2d_graph.py` / `scripts/eval_2d_graph.py`: add `--dxf-eps-scale` (optional override).

## Verification

Unit test:

```bash
./.venv/bin/python -m pytest -q tests/unit/test_dataset2d_eps_scale.py
```

## Experiments (Strict Diagnose Accuracy)

Dataset / strict mode:

- DXF corpus: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf` (110 files)
- strict diagnose: strip DXF text entities + mask filename
- weak labels from manifest (normalized + cleaned)

Common configuration:

- `--student-geometry-only`
- `--diagnose-no-text-no-filename`
- `--dxf-enhanced-keypoints true`
- `--dxf-edge-augment-knn-k 0`
- `--normalize-labels --clean-min-count 5`
- `--distill --teacher titleblock --distill-alpha 0.1`
- `--epochs 3 --batch-size 4`
- `--graph-cache disk`

### Pipeline Runs

Run A (`DXF_EPS_SCALE=0.001`):

- work dir: `/tmp/graph2d_eps_scale_001_20260214`
- strict accuracy: `0.2364`

Run B (`DXF_EPS_SCALE=0.002`):

- work dir: `/tmp/graph2d_eps_scale_002_20260214`
- strict accuracy: `0.2273`

Run C (`DXF_EPS_SCALE=0.005`):

- work dir: `/tmp/graph2d_eps_scale_005_20260214`
- strict accuracy: `0.2091`

### Graph Build Audit (Adjacency Edge Density Proxy)

Audit settings:

- `--strip-text-entities`
- `DXF_ENHANCED_KEYPOINTS=true`
- `DXF_EDGE_AUGMENT_KNN_K=0`
- `DXF_MAX_NODES=200`, `DXF_FRAME_PRIORITY_RATIO=0.1`, `DXF_LONG_LINE_RATIO=0.4`

`DXF_EPS_SCALE=0.001`:

- adj_edges p50: `264`, p90: `581.2` (ref: `/tmp/graph2d_graph_audit_enhanced_20260214`)

`DXF_EPS_SCALE=0.002`:

- adj_edges p50: `380`, p90: `781.6` (ref: `/tmp/graph2d_graph_audit_eps_002_20260214`)

`DXF_EPS_SCALE=0.005`:

- adj_edges p50: `652`, p90: `1554.0` (ref: `/tmp/graph2d_graph_audit_eps_005_20260214`)

## Conclusion

- Increasing `DXF_EPS_SCALE` increases adjacency edges substantially, but did **not** improve strict accuracy on this corpus/config.
- Recommended default for strict geometry-only runs on this corpus/config remains `DXF_EPS_SCALE=0.001`.

