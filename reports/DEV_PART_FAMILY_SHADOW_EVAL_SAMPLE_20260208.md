# DEV_PART_FAMILY_SHADOW_EVAL_SAMPLE_20260208

## Goal
Run a small, non-interactive evaluation to confirm the **PartClassifier shadow-only** integration produces:
- `classification.part_classifier_prediction`
- normalized `classification.part_family*` fields

without requiring manual drawing review.

## Run
Environment:
- Used `.venv-graph` (torch-enabled) because `.venv` does not include torch.
- Disabled Graph2D/Hybrid/Fusion feature flags in the shell command to reduce unrelated signal surface.

Command:
```bash
GRAPH2D_ENABLED=false \
HYBRID_CLASSIFIER_ENABLED=false \
FUSION_ANALYZER_ENABLED=false \
  .venv-graph/bin/python scripts/eval_part_family_shadow.py \
    --input-dir "/Users/huazhou/Downloads/训练图纸/训练图纸" \
    --suffixes dxf \
    --max-files 10 \
    --output-csv reports/experiments/20260208/part_family_shadow_sample_10.csv
```

Output:
- `reports/experiments/20260208/part_family_shadow_sample_10.csv`

## Results (Sample)
The input directory contained 6 readable `.dxf` files (the rest are primarily `.dwg`).

Summary:
- `part_classifier_status`: `ok=6`
- `part_family` distribution: `其他=4`, `连接件=1`, `壳体类=1`
- `part_family_needs_review`: `true=4`, `false=2`

Notes:
- Several samples were flagged `needs_review` due to low confidence, which is expected behavior for shadow evaluation and should be handled downstream (UI or review tooling).

## Verification
- CSV produced successfully and includes `part_family*` columns populated from the provider output.

