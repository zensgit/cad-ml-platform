# DEV_GRAPH2D_TEMPERATURE_CALIBRATION_20260124

## Objective
Calibrate the Graph2D Node19 normalized-cleaned-augmented classifier with temperature scaling to improve confidence calibration on DXF validation samples.

## Inputs
- Script: `scripts/calibrate_graph2d_temperature.py`
- Manifest: `reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_ODA_NORMALIZED_CLEANED_20260123.csv`
- DXF directory: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123`
- Checkpoint: `models/graph2d_training_oda_node19_normalized_cleaned_augmented_20260123.pth`
- Environment: `.venv-graph`

## Command
```
.venv-graph/bin/python scripts/calibrate_graph2d_temperature.py \
  --manifest "reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_ODA_NORMALIZED_CLEANED_20260123.csv" \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123" \
  --checkpoint "models/graph2d_training_oda_node19_normalized_cleaned_augmented_20260123.pth" \
  --output-calibration "reports/experiments/20260123/GRAPH2D_TEMPERATURE_CALIBRATION_20260124.json" \
  --output-predictions "reports/experiments/20260123/GRAPH2D_TEMPERATURE_PREDICTIONS_20260124.csv"
```

## Results
- Validation samples: 18 (stratified 20% split)
- Temperature: 0.965337
- NLL: 1.976248 -> 1.973759
- ECE: 0.217827 -> 0.216095

## Outputs
- Calibration JSON: `reports/experiments/20260123/GRAPH2D_TEMPERATURE_CALIBRATION_20260124.json`
- Predictions CSV: `reports/experiments/20260123/GRAPH2D_TEMPERATURE_PREDICTIONS_20260124.csv`

## Notes
- Calibration is a post-hoc scaling step; it does not change model accuracy, only the probability calibration.
