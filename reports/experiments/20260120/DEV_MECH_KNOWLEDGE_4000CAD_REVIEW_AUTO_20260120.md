# DEV_MECH_KNOWLEDGE_4000CAD_REVIEW_AUTO_20260120

## Summary
Auto-reviewed the 200-sample DXF review sheet using Graph2D predictions and
synonym normalization, producing a confirmed subset and a conflict list for
manual follow-up.

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
- Script: `scripts/auto_review_dxf_sheet.py`

## Snapshot
- Confirmed: 66
- Needs follow-up: 134
- Auto-review marker: `review_status=confirmed`, `review_notes=auto_review:graph2d_match`

## Notes
- Most conflicts are Graph2D predicting `机械制图` while the suggested label is more
  specific. This set is a good target for manual review or future label merges.
