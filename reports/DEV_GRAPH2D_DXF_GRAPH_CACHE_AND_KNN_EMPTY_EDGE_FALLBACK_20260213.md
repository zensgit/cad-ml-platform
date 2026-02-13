# DEV_GRAPH2D_DXF_GRAPH_CACHE_AND_KNN_EMPTY_EDGE_FALLBACK_20260213

## Goal

Improve Graph2D training iteration speed and stability for large DXF datasets by:

1. Avoiding the "fully connected" edge explosion when a DXF graph has no edges.
2. Avoiding repeated DXF parsing and graph building across epochs by adding an optional in-memory graph cache.

## Changes

### 1) kNN Fallback When No Edges Are Found

File: `src/ml/train/dataset_2d.py`

When the edge builder finds no adjacency edges, the legacy behavior was to fall back to a fully-connected graph (`N*(N-1)` directed edges). This can be expensive for `N=200`.

New behavior (opt-in via env):

- `DXF_EMPTY_EDGE_FALLBACK=knn`
  - Builds a bounded kNN graph on entity centers.
- `DXF_EMPTY_EDGE_K=8`
  - Controls k.

Default remains `fully_connected` to preserve existing training/inference behavior.

### 2) DXFManifestDataset Graph Cache (Opt-In)

File: `src/ml/train/dataset_2d.py`

`DXFManifestDataset` now supports an optional in-memory cache keyed by file path + mtime + key graph params.

Environment:

- `DXF_MANIFEST_DATASET_CACHE=memory` enables caching.
- `DXF_MANIFEST_DATASET_CACHE_MAX_ITEMS=0` (0 = unlimited) bounds cache size.

### 3) Local Pipeline Support

File: `scripts/run_graph2d_pipeline_local.py`

Added flags to enable these settings for local runs:

- `--graph-cache {none,memory}` (default: `memory`)
- `--graph-cache-max-items N` (default: `0`)
- `--empty-edge-fallback {fully_connected,knn}` (default: `fully_connected`)
- `--empty-edge-knn-k K` (default: `8`)

The pipeline writes the chosen settings into `pipeline_summary.json` under `graph_build`.

## Validation

- `.venv/bin/python -m py_compile src/ml/train/dataset_2d.py scripts/run_graph2d_pipeline_local.py` (passed)
- `.venv/bin/python -m pytest tests/unit/test_dxf_graph_knn_empty_edge_fallback.py tests/unit/test_dxf_manifest_dataset_graph_cache.py -q` (passed)
- `.venv/bin/python -m flake8 src/ml/train/dataset_2d.py scripts/run_graph2d_pipeline_local.py tests/unit/test_dxf_graph_knn_empty_edge_fallback.py tests/unit/test_dxf_manifest_dataset_graph_cache.py` (passed)

