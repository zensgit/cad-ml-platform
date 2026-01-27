# DEV_DXF_TITLEBLOCK_BATCH_ANALYSIS_20260125

## Objective
Assess title-block extraction and hybrid fusion behavior on training DXF ODA data with title-block signals enabled.

## Execution
- Command:
```
TITLEBLOCK_ENABLED=true TITLEBLOCK_MIN_CONF=0.75 TITLEBLOCK_FUSION_WEIGHT=0.2 \
GRAPH2D_ENABLED=true GRAPH2D_FUSION_ENABLED=true FUSION_ANALYZER_ENABLED=true \
GRAPH2D_SOFT_OVERRIDE_MIN_CONF=0.19 \
GRAPH2D_MODEL_PATH=models/graph2d_training_oda_node19_normalized_cleaned_augmented_20260123.pth \
GRAPH2D_TEMPERATURE_CALIBRATION_PATH=reports/experiments/20260123/GRAPH2D_TEMPERATURE_CALIBRATION_20260124.json \
.venv-graph/bin/python scripts/batch_analyze_dxf_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123" \
  --output-dir "reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_titleblock_20260125" \
  --max-files 120
```

## Outputs
- `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_titleblock_20260125/summary.json`
- `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_titleblock_20260125/batch_results.csv`
- `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_titleblock_20260125/label_distribution.csv`
- `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_titleblock_20260125/batch_low_confidence.csv`
- `reports/experiments/20260123/titleblock_coverage_20260125.csv`
- `reports/experiments/20260123/titleblock_raw_texts_buckets_20260125.csv`
- `reports/experiments/20260123/titleblock_region_entities_buckets_20260125.csv`

## Findings
- Total analyzed: 110 (success 110).
- Title-block labels produced: 0/110 (0.0%).
- Title-block raw texts present: 40/110 (36.36%); region entities present: 40/110.
- Confidence buckets from summary: >=0.8 (57), 0.6–0.8 (25), 0.4–0.6 (28).
- Soft override candidates: 2.
- Label distribution (top): complex_assembly (28), moderate_component (25), 机械制图 (14), 盖 (11).

## Notes
- Title-block text is present in ~36% of samples but did not match the current synonym/label mapping.
