# DEV_DXF_MANIFEST_RELATIVE_PATH_AND_SUBSAMPLING_20260208

## Goal
Make the DXF Graph2D training/evaluation pipeline robust for:

- DXF datasets stored under subdirectories (manifest provides `relative_path`)
- Small smoke runs using `--max-samples` without accidentally collapsing to a single class

This enables automated evaluation without manual drawing review.

## Changes

### 1) Manifest Dataset Path Resolution
- Updated `src/ml/train/dataset_2d.py` (`DXFManifestDataset`) to:
  - read optional `relative_path` from manifest CSV
  - resolve DXF files via `dxf_dir/relative_path` (with fallbacks for older manifests)
  - log parse errors with the resolved file path for easier debugging

### 2) Manifest Builder Improvements
- Updated `scripts/build_dxf_label_manifest.py` to support:
  - `--label-mode filename` (uses `FilenameClassifier`; recommended for real DXF datasets)
  - `--label-mode parent_dir` (labels from parent directory; useful for synthetic datasets)
  - optional synonym standardization (disable via `--no-standardize`)
  - default output path: `reports/experiments/<YYYYMMDD>/DXF_LABEL_MANIFEST_<YYYYMMDD>.csv`

### 3) Stratified Subsampling for Smoke Runs
- Updated `scripts/train_2d_graph.py`:
  - added `--max-samples-strategy {head|random|stratified}` (default `stratified`)
  - changed `--max-samples` behavior to subsample indices (instead of truncating the manifest head)
- Updated `scripts/eval_2d_graph.py` with the same `--max-samples-strategy` behavior.

### 4) Tests
- Added `tests/unit/test_dxf_manifest_dataset_paths.py`:
  - validates `DXFManifestDataset` reads DXF under subdirectories via `relative_path`
  - validates `file_name` itself can be a relative path (no `relative_path` column)

## Verification

### Unit Tests
```bash
python3 -m pytest -q tests/unit/test_dxf_manifest_dataset_paths.py
```
Result: `2 passed`

### End-to-End Smoke (Synthetic DXF)
1) Build a manifest from the repo synthetic dataset (`data/synthetic_v2`) using parent directory labels:
```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/build_dxf_label_manifest.py \
  --input-dir data/synthetic_v2 \
  --recursive \
  --label-mode parent_dir \
  --no-standardize \
  --output-csv /tmp/synth_manifest.csv
```
Result: `Wrote 4000 rows`

2) Train a 1-epoch smoke checkpoint using stratified subsampling (`--max-samples 600`):
```bash
PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME=/tmp/xdg-cache python3 scripts/train_2d_graph.py \
  --manifest /tmp/synth_manifest.csv \
  --dxf-dir data/synthetic_v2 \
  --epochs 1 \
  --batch-size 16 \
  --node-dim 19 \
  --model gcn \
  --loss focal \
  --class-weighting sqrt \
  --sampler balanced \
  --max-samples 600 \
  --output /tmp/graph2d_synth_smoke.pth \
  --save-best
```
Key output:
- `classes=5` (no single-class truncation)
- `val_acc=0.434`

3) Evaluate the smoke checkpoint (stratified subsample, default `val_split=0.2`):
```bash
mkdir -p reports/experiments/20260208
PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME=/tmp/xdg-cache python3 scripts/eval_2d_graph.py \
  --manifest /tmp/synth_manifest.csv \
  --dxf-dir data/synthetic_v2 \
  --checkpoint /tmp/graph2d_synth_smoke.pth \
  --max-samples 600 \
  --output-metrics reports/experiments/20260208/GRAPH2D_SYNTH_V2_VAL_METRICS_20260208.csv \
  --output-errors reports/experiments/20260208/GRAPH2D_SYNTH_V2_VAL_ERRORS_20260208.csv
```
Observed summary:
- `acc=0.443`, `top2=0.583`
- `macro_f1=0.388`, `weighted_f1=0.388`

Artifacts:
- `reports/experiments/20260208/GRAPH2D_SYNTH_V2_VAL_METRICS_20260208.csv`
- `reports/experiments/20260208/GRAPH2D_SYNTH_V2_VAL_ERRORS_20260208.csv`

## Notes / Next
- For real DXF datasets with meaningful filenames, prefer:
  - `scripts/build_dxf_label_manifest.py --label-mode filename`
  - then train with `scripts/train_2d_graph.py` on that manifest.
- This report validates pipeline correctness; it does not claim the synthetic smoke accuracy is production-representative.

