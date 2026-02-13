# DEV_GRAPH2D_DIAGNOSE_NO_TEXT_NO_FILENAME_20260214

## Goal

Provide a repeatable evaluation mode for Graph2D that simulates the strictest inference environment:

- **No DXF text/annotation entities** (geometry-only bytes input)
- **No filename signal** (pass `"masked.dxf"` as the filename input)

This is useful for stable regression metrics while developing distillation (teacher may use text/filename, but the student must work without them).

## Changes

File: `scripts/diagnose_graph2d_on_dxf_dir.py`

- Added CLI flags:
  - `--strip-text-entities`: strips DXF text/annotation entities prior to inference using `strip_dxf_text_entities_from_bytes(..., strip_blocks=True)`.
  - `--mask-filename`: passes `"masked.dxf"` as the filename input during inference (and for `--labels-from-filename` truth mode).
- `main()` now accepts an optional `argv` list to make the script unit-testable.
- `predictions.csv` now includes `model_file_name` so the evaluation configuration is explicit in per-file artifacts.
- CSV writer now supports rows with extra keys (stable union of fieldnames), avoiding failures when some rows include `"error"`.
- `summary.json` includes:
  - `eval_options.strip_text_entities`
  - `eval_options.mask_filename`

### Unit Tests

File: `tests/unit/test_diagnose_graph2d_no_text_no_filename_flags.py`

- Stubs `Graph2DClassifier` and asserts:
  - strip function is invoked when `--strip-text-entities` is set
  - inference receives `file_name == "masked.dxf"` when `--mask-filename` is set
  - `summary.json` records the enabled options

## Validation

### Unit Tests

- `.venv/bin/python -m pytest tests/unit/test_diagnose_graph2d_no_text_no_filename_flags.py -v` (passed)

### Real DXF Smoke (No Text + Masked Filename)

Commands:

```bash
.venv/bin/python scripts/build_dxf_label_manifest.py \
  --input-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --recursive \
  --label-mode filename \
  --output-csv /tmp/dxf_manifest_diag_20260214.csv

.venv/bin/python scripts/diagnose_graph2d_on_dxf_dir.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --model-path "models/graph2d_training_dxf_oda_titleblock_distill_20260210.pth" \
  --manifest-csv /tmp/dxf_manifest_diag_20260214.csv \
  --true-label-min-confidence 0.8 \
  --max-files 20 \
  --seed 42 \
  --output-dir "/tmp/graph2d_diagnose_no_text_no_filename_20260214_014538" \
  --strip-text-entities \
  --mask-filename
```

Observed:

- Output: `/tmp/graph2d_diagnose_no_text_no_filename_20260214_014538`
- `status_counts.ok=20`
- `accuracy=0.2` (weak labels from manifest)
- Timing: `real 6.31s` for 20 files (strip mode is slower as expected)

## Notes / Caveats

- `--strip-text-entities` is intentionally slower (bytes strip + re-parse). Use smaller `--max-files` for iteration.
- `--mask-filename` does not change Graph2D predictions today (Graph2D does not consume filename), but it ensures filename-based truth mode can be disabled and keeps the evaluation contract explicit.

