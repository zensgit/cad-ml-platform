# DEV_GRAPH2D_LOCAL_PIPELINE_AND_MANIFEST_DIAG_20260213

## Goal

Make local Graph2D iteration reproducible and make diagnosis accuracy meaningful when the training labels are **normalized/cleaned** (coarse buckets).

Previously, `scripts/diagnose_graph2d_on_dxf_dir.py` could only score accuracy when "truth" came from parent-dir names or weak filename parsing. Once we normalize labels via a manifest (for example mapping 144 fine labels into 11 buckets), diagnosis accuracy/confusions become incorrect unless diagnosis uses the **same manifest truth**.

## Changes

### 1) Manifest-Based Truth Mode for Diagnose

File: `scripts/diagnose_graph2d_on_dxf_dir.py`

- Added `--manifest-csv`:
  - Expects at minimum `file_name` and `label_cn`.
  - If present, it will be used as ground-truth label source.
  - It prefers `relative_path` (when present) and falls back to `file_name`.
- Enforced mutual exclusivity of truth sources:
  - `--labels-from-parent-dir`
  - `--labels-from-filename`
  - `--manifest-csv`
- Summary output now correctly reports:
  - `true_labels.source = "manifest"` when using `--manifest-csv`
  - `accuracy`, `top_confusions`, `per_class_accuracy` when truth is enabled

### 2) Local Graph2D Pipeline Orchestrator

File: `scripts/run_graph2d_pipeline_local.py`

New local-only orchestrator to run the full loop:

1. Build manifest (`scripts/build_dxf_label_manifest.py`)
2. Filter weak labels by confidence
3. Optional normalize labels (`scripts/normalize_dxf_label_manifest.py`)
4. Optional clean labels by min count (`scripts/clean_dxf_label_manifest.py`)
5. Train checkpoint (`scripts/train_2d_graph.py`)
6. Eval (`scripts/eval_2d_graph.py`)
7. Diagnose using **manifest truth** (`scripts/diagnose_graph2d_on_dxf_dir.py --manifest-csv ...`)

Notes:

- Outputs go to `/tmp/graph2d_pipeline_local_<timestamp>` by default.
- A local `.gitignore` is written into the work dir to avoid accidentally tracking artifacts.

### 3) Regression Test (Torch-Free)

File: `tests/unit/test_diagnose_graph2d_manifest_truth.py`

- Adds a unit regression test for `--manifest-csv` truth mode.
- Stubs `src.ml.vision_2d.Graph2DClassifier` via `sys.modules` to keep the test fast and torch-free.

## Usage

### Quick Local Pipeline

```bash
.venv/bin/python scripts/run_graph2d_pipeline_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --normalize-labels \
  --clean-min-count 2 \
  --model edge_sage \
  --loss cross_entropy \
  --class-weighting inverse \
  --sampler balanced \
  --epochs 15 \
  --max-samples 0 \
  --diagnose-max-files 200
```

### Standalone Diagnosis Against a Manifest

```bash
.venv/bin/python scripts/diagnose_graph2d_on_dxf_dir.py \
  --dxf-dir "/path/to/dxfs" \
  --model-path "/path/to/checkpoint.pth" \
  --manifest-csv "/path/to/manifest.cleaned.csv" \
  --max-files 200 \
  --output-dir "/tmp/graph2d_diagnose"
```

## Validation

### Static Checks

- `.venv/bin/python -m py_compile scripts/run_graph2d_pipeline_local.py scripts/diagnose_graph2d_on_dxf_dir.py`
- `.venv/bin/python -m flake8 scripts/diagnose_graph2d_on_dxf_dir.py scripts/run_graph2d_pipeline_local.py tests/unit/test_diagnose_graph2d_manifest_truth.py`

### Unit Test

- `.venv/bin/python -m pytest tests/unit/test_diagnose_graph2d_manifest_truth.py -q` (passed)

### Local Smoke Run (End-to-End)

Command:

```bash
.venv/bin/python scripts/run_graph2d_pipeline_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --normalize-labels \
  --clean-min-count 2 \
  --epochs 1 \
  --max-samples 80 \
  --diagnose-max-files 30 \
  --model edge_sage \
  --loss cross_entropy \
  --class-weighting inverse \
  --sampler balanced
```

Result (artifact path):

- `work_dir=/tmp/graph2d_pipeline_local_20260213_190656`
- `diagnose/summary.json` includes `true_labels.source="manifest"` and computed `accuracy`.

