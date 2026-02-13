# DEV_GRAPH2D_LOCAL_RETRAIN_COARSE_KNN_CACHE_20260213

## Goal

Validate the new DXF graph build speed/stability knobs in a real local retrain loop:

- Enable `DXFManifestDataset` in-memory graph caching during training.
- Switch empty-edge fallback from fully-connected to kNN to avoid potential edge explosions.

## Command

```bash
/usr/bin/time -p .venv/bin/python scripts/run_graph2d_pipeline_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --normalize-labels \
  --clean-min-count 2 \
  --model edge_sage \
  --loss cross_entropy \
  --class-weighting inverse \
  --sampler balanced \
  --epochs 15 \
  --diagnose-max-files 80 \
  --empty-edge-fallback knn \
  --empty-edge-knn-k 8
```

## Outputs

- work dir: `/tmp/graph2d_pipeline_local_20260213_214640`
- pipeline settings (from `pipeline_summary.json`):
  - `graph_build.empty_edge_fallback=knn`
  - `graph_build.empty_edge_knn_k=8`
  - `graph_build.cache=memory`
  - `graph_build.cache_max_items=0`

## Results

### Runtime (wall clock)

From `/usr/bin/time -p`:

- `real` 252.99s
- `user` 236.87s
- `sys` 10.03s

### Validation (eval_2d_graph.py)

Overall on validation split (`eval_metrics.csv` `__overall__` row):

- `acc`: **0.450** (9/20)
- `top2_accuracy`: **0.700**
- `macro_f1`: **0.391**
- `weighted_f1`: **0.368**

### Diagnosis (diagnose_graph2d_on_dxf_dir.py, manifest truth)

From `diagnose/summary.json`:

- sampled files: 80
- `accuracy`: **0.375**
- confidence p50/p90: **0.286 / 0.579**

## Notes

- This run uses **coarse buckets** produced by label normalization + cleaning (`--clean-min-count 2`), so metrics are relative to filename-derived weak labels in that coarse label space.
- This validates that enabling caching and kNN fallback does not change the training loop behavior or break the pipeline.

