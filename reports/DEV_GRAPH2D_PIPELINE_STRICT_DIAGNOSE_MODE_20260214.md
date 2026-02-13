# DEV_GRAPH2D_PIPELINE_STRICT_DIAGNOSE_MODE_20260214

## Goal

Make the local Graph2D pipeline runner (`scripts/run_graph2d_pipeline_local.py`) able to run the **strict diagnosis mode** (no DXF text entities + masked filename) automatically as part of the end-to-end workflow.

This keeps the pipeline self-contained for regression checks while iterating on distillation and geometry-only student experiments.

## Changes

File: `scripts/run_graph2d_pipeline_local.py`

- Added flag: `--diagnose-no-text-no-filename`
  - When enabled, the pipeline passes `--strip-text-entities --mask-filename` to
    `scripts/diagnose_graph2d_on_dxf_dir.py`.
- Added helper: `_build_diagnose_cmd(...)` for testable command wiring.
- `pipeline_summary.json` now records:
  - `diagnose.no_text_no_filename`

### Unit Tests

- Updated: `tests/unit/test_run_graph2d_pipeline_local_distill_wiring.py`
  - Adds `diagnose_no_text_no_filename` to the args namespace.
- Added: `tests/unit/test_run_graph2d_pipeline_local_diagnose_strict_wiring.py`
  - Verifies strict diagnose flags are present when enabled.

## Validation

### Unit Tests

- `.venv/bin/python -m pytest tests/unit/test_run_graph2d_pipeline_local_distill_wiring.py tests/unit/test_run_graph2d_pipeline_local_diagnose_strict_wiring.py -v` (passed)

### Pipeline Smoke (Training Drawings + Strict Diagnose)

Command:

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
  --diagnose-no-text-no-filename
```

Observed:

- Work dir: `/tmp/graph2d_pipeline_local_20260214_020925`
- Diagnose invoked with `--strip-text-entities --mask-filename`
- Completed successfully (artifacts in `/tmp`)
- Timing: `real 25.85s`

## Notes / Caveats

- This smoke run is for **wiring validation** only (1 epoch / 40 samples). Accuracy metrics are not meaningful at this size.
- Strict diagnose mode is slower than default diagnosis because it strips DXF text entities via a bytes round-trip.

