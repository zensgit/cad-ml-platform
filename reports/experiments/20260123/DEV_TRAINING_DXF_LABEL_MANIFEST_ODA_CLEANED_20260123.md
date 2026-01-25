# DEV_TRAINING_DXF_LABEL_MANIFEST_ODA_CLEANED_20260123

## Summary
- Cleaned the ODA DXF label manifest by mapping low-frequency labels to `other`.

## Command
- `.venv-graph/bin/python scripts/clean_dxf_label_manifest.py --input-csv "reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_ODA_20260123.csv" --output-csv "reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_ODA_CLEANED_20260123.csv" --min-count 3 --other-label other`

## Output
- `reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_ODA_CLEANED_20260123.csv`
