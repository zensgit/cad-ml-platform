# DEV_DXF_GRAPH_BATCH_MANIFEST_20260123

## Summary
- Added manifest-based file selection for local DXF batch analysis.

## Implementation
- `scripts/batch_analyze_dxf_local.py` now accepts `--manifest` and resolves paths via `relative_path`, `file_name`, and `source_dir`.
- Falls back to directory scan when `--manifest` is not provided.

## Validation
- `python3 -m py_compile scripts/batch_analyze_dxf_local.py`
- `python3 scripts/batch_analyze_dxf_local.py --help` stalled due to app import; aborted.
