# DEV_DXF_TITLEBLOCK_TUNED_FULL_SYNONYMS_BATCH_ANALYSIS_20260125

## Objective
Re-run the full tuned batch after adding title-block synonyms to achieve complete label coverage.

## Execution
- Command:
```
TITLEBLOCK_ENABLED=true TITLEBLOCK_MIN_CONF=0.75 TITLEBLOCK_FUSION_WEIGHT=0.2 \
TITLEBLOCK_REGION_X_RATIO=0.7 TITLEBLOCK_REGION_Y_RATIO=0.5 \
GRAPH2D_ENABLED=true GRAPH2D_FUSION_ENABLED=true FUSION_ANALYZER_ENABLED=true \
GRAPH2D_SOFT_OVERRIDE_MIN_CONF=0.19 \
GRAPH2D_MODEL_PATH=models/graph2d_training_oda_node19_normalized_cleaned_augmented_20260123.pth \
GRAPH2D_TEMPERATURE_CALIBRATION_PATH=reports/experiments/20260123/GRAPH2D_TEMPERATURE_CALIBRATION_20260124.json \
.venv-graph/bin/python scripts/batch_analyze_dxf_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123" \
  --output-dir "reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_titleblock_tuned_full_synonyms_20260125" \
  --max-files 120
```

## Outputs
- `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_titleblock_tuned_full_synonyms_20260125/summary.json`
- `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_titleblock_tuned_full_synonyms_20260125/batch_results.csv`
- `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_titleblock_tuned_full_synonyms_20260125/label_distribution.csv`
- `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_titleblock_tuned_full_synonyms_20260125/batch_low_confidence.csv`
- `reports/experiments/20260123/titleblock_tuned_full_synonyms_coverage_20260125.csv`
- `reports/experiments/20260123/titleblock_tuned_full_synonyms_raw_texts_buckets_20260125.csv`
- `reports/experiments/20260123/titleblock_tuned_full_synonyms_region_entities_buckets_20260125.csv`

## Findings
- Total analyzed: 110 (success 110).
- Title-block labels produced: 110/110 (100%).
- Title-block raw texts present: 110/110 (100%).
- Title-block confidence >= 0.75: 107/110 (97.27%).
- `轴类` label appears for 2 samples where title-block part name is `小减速机轴`.
