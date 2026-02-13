# DEV_GRAPH2D_DIAGNOSE_MANIFEST_STRICT_AND_REL_PATH_20260213

## Goal

Make `scripts/diagnose_graph2d_on_dxf_dir.py --manifest-csv ...` safer and more useful for nested DXF directories:

- Fail fast when a manifest path is provided but missing/empty, to avoid silently producing "manifest" truth mode summaries with zero coverage.
- Include `relative_path` in `predictions.csv` so nested datasets (or duplicate file names) are diagnosable and manifest matching is traceable.

## Changes

File: `scripts/diagnose_graph2d_on_dxf_dir.py`

- `--manifest-csv` now:
  - Raises `SystemExit` if the manifest file does not exist.
  - Raises `SystemExit` if it contains no usable rows (expects `file_name` + `label_cn`).
- `predictions.csv` now includes:
  - `relative_path`: the path of the DXF file relative to `--dxf-dir`.
- Manifest lookup prefers:
  - `relative_path` (if present in the manifest)
  - fallback to `file_name`

## Validation

- `.venv/bin/python -m py_compile scripts/diagnose_graph2d_on_dxf_dir.py` (passed)
- `.venv/bin/python -m pytest tests/unit/test_diagnose_graph2d_manifest_truth.py -q` (passed)

