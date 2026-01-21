# DEV_MECH_KNOWLEDGE_4000CAD_MANUAL_REVIEW_20260120

## Summary
Reviewed the nine numeric-named practice drawings (1,2,3,4,6,7,8,9,10) after
normalizing DXF text. The extracted strings contain only dimensions or symbols
with no part names, so the generic label `练习零件图` is retained and marked as
manual-review complete.

## Steps
- Extracted DXF text via `scripts/auto_label_unlabeled_dxf._extract_text` for 1-10.
- Confirmed that files 1,2,3,4,6,7,8,9,10 lack part names beyond dimensions.
- Added `manual_review:dimension_only` to the notes field for those rows.

## Findings
- `1.dwg`: dimensions only (e.g., R11, R8, 118°).
- `2.dwg`: dimensions only (R9, R17, 30°).
- `3.dwg`: dimensions only (1x45, ⌀22).
- `4.dwg`: dimensions only (20, R15, ⌀10).
- `6.dwg`: dimensions only (⌀, R values).
- `7.dwg`: dimensions only (R2, ⌀).
- `8.dwg`: dimensions only (⌀ values).
- `9.dwg`: dimensions only (R20, R40).
- `10.dwg`: dimensions only (⌀28, R3).

## Outputs
- `reports/MECH_4000_DWG_UNLABELED_LABELS_TEMPLATE_20260119.csv`

## Notes
- These rows remain labeled as `练习零件图` with confidence 0.70.
