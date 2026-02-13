# DEV_GRAPH2D_DISTILLATION_PIPELINE_WIRING_20260214

## Goal

Make Graph2D knowledge-distillation training reproducible from the local pipeline runner (`scripts/run_graph2d_pipeline_local.py`) by wiring distillation flags through to `scripts/train_2d_graph.py`, and record the distillation configuration in the pipeline summary artifact.

## Changes

File: `scripts/run_graph2d_pipeline_local.py`

- Added distillation CLI flags:
  - `--distill`
  - `--teacher filename|titleblock|hybrid`
  - `--distill-alpha`
  - `--distill-temp`
  - `--distill-mask-filename auto|true|false` (default: `auto`)
- Added `_resolve_distill_mask(...)` and `_build_train_cmd(...)` helpers so CLI wiring is testable.
- Pipeline summary now includes a `distillation` section with resolved parameters.

### Unit Tests

File: `tests/unit/test_run_graph2d_pipeline_local_distill_wiring.py`

- Verifies:
  - `--distill` wiring includes `--teacher/--distill-alpha/--distill-temp`
  - `auto` masking enables `--distill-mask-filename` for `hybrid/titleblock` teachers
  - `auto` masking does not add the flag for the `filename` teacher
  - `mask=false` suppresses `--distill-mask-filename`

## Validation

### Static Checks

- `.venv/bin/python -m py_compile scripts/run_graph2d_pipeline_local.py` (passed)

### Unit Tests

- `.venv/bin/pytest tests/unit/test_run_graph2d_pipeline_local_distill_wiring.py -q` (passed)

### Pipeline Smoke (Training Drawings, 1 Epoch / 40 Samples)

Baseline (no distill):

```bash
/usr/bin/time -p .venv/bin/python scripts/run_graph2d_pipeline_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --epochs 1 \
  --max-samples 40 \
  --diagnose-max-files 20 \
  --model edge_sage \
  --loss cross_entropy \
  --class-weighting inverse \
  --sampler balanced \
  --graph-cache both \
  --empty-edge-fallback knn \
  --empty-edge-knn-k 8
```

Observed:

- Work dir: `/tmp/graph2d_pipeline_local_20260214_011458`
- Eval: `acc=0.021 top2=0.043 macro_f1=0.001 weighted_f1=0.001`
- Diagnose: `accuracy=0.0` (20 sampled)

Distill (teacher=hybrid, masked filename):

```bash
/usr/bin/time -p .venv/bin/python scripts/run_graph2d_pipeline_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --epochs 1 \
  --max-samples 40 \
  --diagnose-max-files 20 \
  --model edge_sage \
  --loss cross_entropy \
  --class-weighting inverse \
  --sampler balanced \
  --graph-cache both \
  --empty-edge-fallback knn \
  --empty-edge-knn-k 8 \
  --distill \
  --teacher hybrid
```

Observed:

- Work dir: `/tmp/graph2d_pipeline_local_20260214_011410`
- Eval: `acc=0.021 top2=0.064 macro_f1=0.001 weighted_f1=0.001`
- Diagnose: `accuracy=0.0` (20 sampled)
- Pipeline summary now records:
  - `distillation.enabled=true`
  - `distillation.teacher=hybrid`
  - `distillation.mask_filename=true` (auto-resolved)

Distill (teacher=titleblock, masked filename):

```bash
/usr/bin/time -p .venv/bin/python scripts/run_graph2d_pipeline_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --epochs 1 \
  --max-samples 40 \
  --diagnose-max-files 20 \
  --model edge_sage \
  --loss cross_entropy \
  --class-weighting inverse \
  --sampler balanced \
  --graph-cache both \
  --empty-edge-fallback knn \
  --empty-edge-knn-k 8 \
  --distill \
  --teacher titleblock
```

Observed:

- Work dir: `/tmp/graph2d_pipeline_local_20260214_011532`
- Eval: `acc=0.021 top2=0.064 macro_f1=0.001 weighted_f1=0.001`
- Diagnose: `accuracy=0.0` (20 sampled)

## Notes / Caveats

- These runs are smoke tests (1 epoch / 40 samples) to validate wiring and artifact recording, not a meaningful performance benchmark.
- Teacher labels are derived from rule/text classifiers and can be outside the student's label map depending on dataset cleaning/normalization; this is expected and handled (uniform soft labels fallback).

