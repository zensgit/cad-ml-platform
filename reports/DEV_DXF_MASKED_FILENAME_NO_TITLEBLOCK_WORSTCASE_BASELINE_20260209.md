# DEV_DXF_MASKED_FILENAME_NO_TITLEBLOCK_WORSTCASE_BASELINE_20260209

## Goal
Quantify the “worst case” DXF classification baseline when:
- upload filenames are masked (no filename signal), and
- titleblock extraction is disabled (no titleblock signal).

This approximates production scenarios where we receive anonymous files or filename/titleblock cannot be trusted.

## Run
Dataset:
- `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123` (110 DXF)

Command:
```bash
PYTHONPYCACHEPREFIX=/tmp/pycache \
MPLCONFIGDIR=/tmp/mplconfig \
XDG_CACHE_HOME=/tmp/xdg-cache \
API_KEY=test \
HYBRID_CLASSIFIER_ENABLED=true \
FILENAME_CLASSIFIER_ENABLED=true \
TITLEBLOCK_ENABLED=false \
TITLEBLOCK_OVERRIDE_ENABLED=false \
PROCESS_FEATURES_ENABLED=true \
GRAPH2D_ENABLED=true \
GRAPH2D_FUSION_ENABLED=true \
FUSION_ANALYZER_ENABLED=true \
  .venv-graph/bin/python scripts/batch_analyze_dxf_local.py \
    --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123" \
    --output-dir reports/experiments/20260209/batch_analyze_training_dxf_oda_masked_filename_no_titleblock \
    --max-files 110 \
    --seed 22 \
    --min-confidence 0.6 \
    --mask-filename
```

Outputs:
- Aggregated:
  - `reports/experiments/20260209/batch_analyze_training_dxf_oda_masked_filename_no_titleblock/summary.json`
  - `reports/experiments/20260209/batch_analyze_training_dxf_oda_masked_filename_no_titleblock/label_distribution.csv`
- Per-file outputs are gitignored by the output directory `.gitignore`.

## Results Summary
From `summary.json`:
- Filename classifier: `label_present_rate=0.0` (expected)
- Titleblock: disabled
- Hybrid decision coverage:
  - `source_counts.process=69`, `fallback=41`
  - `label_present_rate=0.6273`
  - confidence mean/median (all): `0.320 / 0.433`
- Overall output confidence buckets:
  - `gte_0_8=37`, `0_6_0_8=33`, `0_4_0_6=40`
  - `low_confidence_count=73` (<= `0.6`)
- Label distribution collapses to coarse categories / rules:
  - `complex_assembly=40`, `moderate_component=33`
  - `机械制图=14`, `盖=10`, `挡板=5` ...

Interpretation:
- Without filename/titleblock, the current pipeline mainly falls back to:
  - process heuristics (when they trigger), else
  - rule-level defaults and graph2d drawing-type filtering (Graph2D is largely ignored as “drawing-type” for this checkpoint).
- This is a strong signal that if we require “anonymous DXF” recognition, we need a dedicated geometry model task (likely coarse family first) and/or distillation.

