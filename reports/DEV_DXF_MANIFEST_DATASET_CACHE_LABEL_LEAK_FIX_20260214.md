# DEV_DXF_MANIFEST_DATASET_CACHE_LABEL_LEAK_FIX_20260214

## Goal

Fix a correctness bug in `DXFManifestDataset` graph caching: cached samples must not reuse label tensors across different manifests/label-spaces.

This was observed when reusing a shared disk graph cache directory across runs:

- Run 1 (fine labels, many classes) writes cache entries containing label IDs like `43`.
- Run 2 (normalized coarse buckets, fewer classes) loads cached label IDs, causing:
  - `IndexError: Target 43 is out of bounds.` from `torch.nn.functional.cross_entropy`

## Root Cause

File: `src/ml/train/dataset_2d.py`

`DXFManifestDataset` cached `(graph, label)` pairs in memory and on disk.

While graph tensors are reusable across manifests, **labels are not**, because label IDs depend on the label map created from the manifest (and its ordering/normalization/cleaning).

## Fix

File: `src/ml/train/dataset_2d.py`

- Cache **graph tensors only** (memory + disk).
- Recompute the label tensor from the active manifest sample every time.
- When loading cached graphs, refresh metadata (`file_name`, `relative_path`, `file_path`) to match the current manifest row.
- Disk cache remains backward-compatible: older payloads that included a `label` field are accepted and the label is ignored.

## Tests

Added: `tests/unit/test_dxf_manifest_dataset_disk_cache_label_is_not_cached.py`

- Creates two manifests with the same DXF files but different label ordering so the same file would get a different `label_id`.
- Ensures dataset instance 2:
  - reads the graph from disk cache (no `ezdxf.readfile` call)
  - returns the **new** manifest label ID (not the cached one)

## Validation

- `.venv/bin/python -m pytest tests/unit/test_dxf_manifest_dataset_disk_cache.py tests/unit/test_dxf_manifest_dataset_graph_cache.py tests/unit/test_dxf_manifest_dataset_disk_cache_label_is_not_cached.py -v` (passed)

## Notes

- This fix is important for workflows that reuse disk cache across multiple pipeline runs (fine labels vs normalized coarse buckets).

