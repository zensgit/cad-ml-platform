# DEV_DXF_BATCH_ANALYZE_TRAINING_SET_20260212

## Goal
Smoke-evaluate the DXF analyze classification pipeline on a real local DXF
directory, focusing on "filename vs titleblock" signal coverage when Graph2D is
disabled (no torch in this environment).

## Runs
Sampled `30` DXF files from:
- `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf`

### Run A: With Original Filenames
Command:
```bash
GRAPH2D_ENABLED=false TITLEBLOCK_ENABLED=false \
HYBRID_CLASSIFIER_ENABLED=true FILENAME_CLASSIFIER_ENABLED=true PROCESS_FEATURES_ENABLED=true \
.venv/bin/python scripts/batch_analyze_dxf_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --output-dir "reports/experiments/20260212/batch_analyze_training_dxf/with_filename" \
  --max-files 30 --seed 42 --min-confidence 0.6
```

Key summary (`summary.json`):
- Success: `30/30`, low-confidence: `0`
- Filename:
  - `label_present_rate=1.0`, `match_type_counts.exact=30`, `mean_conf=0.95`
- Hybrid decision:
  - `source_counts.filename=30`, `mean_conf=0.95`

Artifacts:
- `reports/experiments/20260212/batch_analyze_training_dxf/with_filename/summary.json`
- `reports/experiments/20260212/batch_analyze_training_dxf/with_filename/label_distribution.csv`

### Run B: Masked Filenames + Titleblock Enabled
Command:
```bash
GRAPH2D_ENABLED=false TITLEBLOCK_ENABLED=true TITLEBLOCK_OVERRIDE_ENABLED=true TITLEBLOCK_MIN_CONF=0.75 \
HYBRID_CLASSIFIER_ENABLED=true FILENAME_CLASSIFIER_ENABLED=true PROCESS_FEATURES_ENABLED=true \
.venv/bin/python scripts/batch_analyze_dxf_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --output-dir "reports/experiments/20260212/batch_analyze_training_dxf/masked_filename_titleblock" \
  --max-files 30 --seed 42 --min-confidence 0.6 --mask-filename
```

Key summary (`summary.json`):
- Success: `30/30`, low-confidence: `0`
- Filename:
  - `label_present_rate=0.0` (expected due to masked filenames)
- Titleblock:
  - `enabled=true`
  - `texts_present_rate=1.0`, `label_present_rate=1.0`, `status_counts.matched=30`
  - `mean_conf=0.85`
- Hybrid decision:
  - `source_counts.titleblock=30`, `mean_conf=0.85`

Artifacts:
- `reports/experiments/20260212/batch_analyze_training_dxf/masked_filename_titleblock/summary.json`
- `reports/experiments/20260212/batch_analyze_training_dxf/masked_filename_titleblock/label_distribution.csv`

## Notes
- In this environment `torch` is not installed, so Graph2D and V16/V6 ML
  classifiers remain unavailable; the evaluation isolates rule/filename/titleblock
  behavior.

