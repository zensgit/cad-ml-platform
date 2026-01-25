# DEV_TRAINING_DXF_LOW_CONF_ANALYSIS_TRAINING_DXF_ODA_GRAPH2D_NODE19_NORMALIZED_CLEANED_AUGMENTED_VALIDATION_20260123

## Validation Steps
- Loaded `batch_results.csv` and `batch_low_confidence.csv` from the augmented batch output.
- Regenerated low-confidence aggregates and confirmed output CSV/JSON files exist.

## Outcome
- total=110, low_confidence_count=53 (ratio=0.4818).
- confidence_min=0.55, confidence_max=0.60, mean=0.5557, median=0.55.
- confidence_source counts: rules=53.
- fusion_label=Standard_Part for all low-confidence entries.
- graph2d_label distribution: 传动件=33, 罐体=18, 设备=2.
- fusion vs graph2d mismatches=53.
