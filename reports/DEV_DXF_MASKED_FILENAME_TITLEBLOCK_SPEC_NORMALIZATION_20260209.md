# DEV_DXF_MASKED_FILENAME_TITLEBLOCK_SPEC_NORMALIZATION_20260209

## Goal
Validate DXF part recognition when filenames carry **no semantic signal** (masked upload names), and fix a systematic low-confidence case where titleblock part names include trailing specs (e.g. `DN1500`).

## Change
- `src/ml/titleblock_extractor.py`
  - Normalize titleblock `part_name` by stripping common trailing spec suffixes (`DN/PN/M...`) before synonym matching.
  - Emit `title_block_info.part_name_normalized` for debugging.
- `scripts/batch_analyze_dxf_local.py`
  - Add `--mask-filename` to upload DXFs as anonymous names (`file_0001.dxf`) for evaluation.

## Verification Run (110 DXF, masked filename)
Command:
```bash
PYTHONPYCACHEPREFIX=/tmp/pycache \
MPLCONFIGDIR=/tmp/mplconfig \
XDG_CACHE_HOME=/tmp/xdg-cache \
API_KEY=test \
HYBRID_CLASSIFIER_ENABLED=true \
FILENAME_CLASSIFIER_ENABLED=true \
TITLEBLOCK_ENABLED=true \
TITLEBLOCK_OVERRIDE_ENABLED=true \
PROCESS_FEATURES_ENABLED=true \
GRAPH2D_ENABLED=true \
GRAPH2D_FUSION_ENABLED=true \
FUSION_ANALYZER_ENABLED=true \
  .venv-graph/bin/python scripts/batch_analyze_dxf_local.py \
    --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123" \
    --output-dir reports/experiments/20260209/batch_analyze_training_dxf_oda_masked_filename \
    --max-files 110 \
    --seed 22 \
    --min-confidence 0.6 \
    --mask-filename
```

Outputs:
- Aggregated:
  - `reports/experiments/20260209/batch_analyze_training_dxf_oda_masked_filename/summary.json`
  - `reports/experiments/20260209/batch_analyze_training_dxf_oda_masked_filename/label_distribution.csv`
- Per-file outputs are gitignored by the output directory `.gitignore`.

## Results Summary
From `summary.json`:
- Filename classifier: `label_present_rate=0.0` (expected; masked filenames)
- Titleblock:
  - `status_counts.matched=110` (previously included `partial_match` for `拖车DN1500`)
  - confidence mean/median: `0.85 / 0.85`
- Hybrid decision:
  - `source_counts.titleblock=110`
  - `low_confidence_count=0`

Unit test added:
- `tests/unit/test_titleblock_extractor.py` verifies `拖车DN1500 -> 拖车` normalization and matched status.

