# DEV_MECH_KNOWLEDGE_4000CAD_REVIEW_AUTO_RERUN_20260120

## Summary
Re-ran the DXF auto-review after updating the script to preserve manual
priority decisions and mark confirmed matches in `review_status`.

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
- Confirmed: 74 (64 auto + 10 manual overrides)
- Needs follow-up: 126
- Manual overrides preserved: `auto_review_verdict=manual_confirmed`

## Notes
- Manual priority decisions remain `review_status=confirmed` with
  `auto_review_reason=manual_override`.
- Auto-confirmed matches now add `review_notes=auto_review:graph2d_match`.
