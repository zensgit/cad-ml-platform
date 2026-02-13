# DEV_GRAPH2D_MANIFEST_EDGE_SAGE_CACHE_FIX_AND_DISK_CACHE_20260213

## Goal

Speed up local Graph2D iteration by making DXF graph caching actually work for the `edge_sage` training path, and add an optional disk cache so the local pipeline can share graph builds across its train/eval subprocess steps.

## Key Fixes / Features

### 1) Fix: Cache Works When `return_edge_attr=True` (EdgeSage)

File: `src/ml/train/dataset_2d.py`

`DXFManifestDataset` previously only cached graphs for the non-edge-attr path. Since `edge_sage` training sets `return_edge_attr=True`, the cache was effectively not used for the main local training mode.

Now:

- Both edge-attr and non-edge-attr graphs are cached consistently.

### 2) Feature: Optional Disk Cache (torch.save/torch.load)

File: `src/ml/train/dataset_2d.py`

New disk cache support so separate processes can share graph builds:

- `DXF_MANIFEST_DATASET_CACHE=disk|both`
- `DXF_MANIFEST_DATASET_CACHE_DIR=/path/to/cache`

Disk cache stores `{graph, label}` per cache key (keyed by file path + mtime + graph params + key env settings).

### 3) Local Pipeline Flags

File: `scripts/run_graph2d_pipeline_local.py`

Extended caching options:

- `--graph-cache {none,memory,disk,both}`
- `--graph-cache-dir DIR` (used for `disk|both`; default: `<work_dir>/graph_cache`)

## Validation

### Unit Tests

- `.venv/bin/python -m pytest tests/unit/test_dxf_manifest_dataset_graph_cache.py tests/unit/test_dxf_manifest_dataset_disk_cache.py -q` (passed)

### Local Pipeline Smoke (Disk Cache Write)

Command:

```bash
.venv/bin/python scripts/run_graph2d_pipeline_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --normalize-labels \
  --clean-min-count 2 \
  --model edge_sage \
  --loss cross_entropy \
  --class-weighting inverse \
  --sampler balanced \
  --epochs 1 \
  --max-samples 40 \
  --diagnose-max-files 20 \
  --graph-cache both \
  --graph-cache-dir /tmp/graph2d_manifest_graph_cache_smoke_20260213
```

Observed:

- Cache dir populated with `40` `.pt` entries:
  - `/tmp/graph2d_manifest_graph_cache_smoke_20260213/*.pt`
- Pipeline work dir:
  - `/tmp/graph2d_pipeline_local_20260213_222813`

