# DEV_TRAINING_DXF_LABEL_MANIFEST_ODA_NORMALIZED_CLEANED_20260123

## Summary
- Collapsed low-frequency normalized labels into `other` to reduce class noise.

## Command
- `python3 scripts/clean_dxf_label_manifest.py --input-csv "reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_ODA_NORMALIZED_20260123.csv" --output-csv "reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_ODA_NORMALIZED_CLEANED_20260123.csv" --min-count 5 --other-label other`

## Output
- `reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_ODA_NORMALIZED_CLEANED_20260123.csv`
