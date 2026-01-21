# DEV_MECH_KNOWLEDGE_4000CAD_AUTO_LABEL_REVIEW_APPLIED_20260121

## Summary
- Applied manual confirmations for all 27 auto-labeled rows (using suggested labels, plus GB27-88 -> 铰制螺栓).
- Updated the merged manifest with revised labels for the unlabeled subset.

## Inputs
- Auto review sheet: `reports/experiments/20260121/MECH_4000_DWG_REVIEW_SHEET_AUTO_20260121.csv`
- Auto label output: `reports/experiments/20260121/MECH_4000_DWG_UNLABELED_AUTO_20260121.csv`

## Outputs
- Applied review sheet: `reports/experiments/20260121/MECH_4000_DWG_REVIEW_SHEET_APPLIED_20260121.csv`
- Updated manifest: `reports/experiments/20260121/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260121.csv`

## Decisions
- Used `suggested_label_cn` as the final label for all rows.
- Special case: `GB27-88` labeled as `铰制螺栓` (standard part inferred from filename + DXF text).
