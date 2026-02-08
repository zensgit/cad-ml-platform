# DEV_PART_FAMILY_SHADOW_EVAL_TRAINING_DXF_20260208

## Goal
Run a larger, non-interactive evaluation on training DXFs to validate that:
- the V16 provider can run end-to-end in shadow mode
- `part_classifier_prediction` + normalized `part_family*` fields are populated consistently

## Run
Environment:
- Used `.venv-graph` (torch-enabled) for V16/V6 runtime.

Command:
```bash
DISABLE_MODEL_SOURCE_CHECK=True \
MPLCONFIGDIR=/tmp/mplconfig \
GRAPH2D_ENABLED=false \
HYBRID_CLASSIFIER_ENABLED=false \
FUSION_ANALYZER_ENABLED=false \
  .venv-graph/bin/python scripts/eval_part_family_shadow.py \
    --input-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
    --suffixes dxf \
    --max-files 50 \
    --seed 22 \
    --output-csv reports/experiments/20260208/part_family_shadow_training_dxf_50.csv
```

Output:
- `reports/experiments/20260208/part_family_shadow_training_dxf_50.csv`

## Results (50 DXF sample)
- `part_classifier_status`: `ok=50`
- `part_family` distribution (top):
  - `其他=32`
  - `壳体类=7`
  - `连接件=6`
  - `轴类=4`
  - `传动件=1`
- `part_family_needs_review`: `true=24`, `false=26`
- `part_family_confidence`:
  - avg `0.822`
  - min `0.402`
  - max `0.987`

Notes:
- The hoster-connectivity warning may still appear in some environments; runs succeed when local model artifacts are present.
- `needs_review=true` should be treated as a downstream UI/workflow signal (shadow-only evaluation).

