# DEV_TRAINING_DXF_LABEL_MANIFEST_CLEANED_VALIDATION_20260123

## Checks
- Cleaned the DXF manifest with a minimum label count threshold of 3.
- Summarized label distribution after merging low-frequency classes.

## Runtime Output
- Command:
  - `python3 scripts/clean_dxf_label_manifest.py --input-csv reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_20260123.csv --output-csv reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_CLEANED_20260123.csv --min-count 3 --other-label other`
- Result:
  - `rows_in=110`, `rows_out=110`
  - `labels_in=47`, `labels_kept=11`
  - `replaced=72`, `dropped=0`
  - Top labels: other(72), 罐体部分(5), 上封头组件(5), 过滤托架(4)
