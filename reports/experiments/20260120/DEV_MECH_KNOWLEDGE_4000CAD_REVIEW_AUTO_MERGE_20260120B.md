# DEV_MECH_KNOWLEDGE_4000CAD_REVIEW_AUTO_MERGE_20260120B

## Summary
Re-ran auto-review after applying label-merge rules to reduce conflicts between
suggested labels and Graph2D predictions.

## Command
```
./.venv-graph/bin/python scripts/auto_review_dxf_sheet.py \
  --input reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_20260120.csv \
  --output reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_AUTO_20260120.csv \
  --conflicts reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_CONFLICTS_20260120.csv \
  --confidence-threshold 0.05
```

## Outputs
- Auto-reviewed sheet: `reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_AUTO_20260120.csv`
- Conflict list: `reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_CONFLICTS_20260120.csv`
- Conflict summary: `reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_CONFLICTS_SUMMARY_20260120.csv`

## Snapshot
- Conflicts after merge: 68 rows
