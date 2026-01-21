# DEV_MECH_KNOWLEDGE_4000CAD_OCR_TEXT_DIAGNOSTIC_20260120

## Summary
Scanned the 12 unlabeled 4000CAD DXFs for text presence across modelspace and
paperspace (including inserted block text). Only 4 files expose text in modelspace;
none expose text in paperspace. The remaining 8 files have no layout text, so OCR
cannot recover labels from rendered views.

## Inputs
- Unlabeled template: `reports/MECH_4000_DWG_UNLABELED_LABELS_TEMPLATE_20260119.csv`
- DXF directory: `/Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf`

## Output
- CSV: `reports/MECH_4000_DWG_OCR_TEXT_SCAN_20260120.csv`

## Results
- Missing rows scanned: 12
- Modelspace text present (direct + inserted blocks): 4
- Paperspace text present (direct + inserted blocks): 0
- Files with modelspace text:
  - `3.dwg`
  - `FU200-02-01-1.DWG`
  - `FU200-02-01-2.DWG`
  - `ZHITUI.DWG`

## Notes
- For 8 files there is no layout text; OCR cannot help without a separate metadata source.
- OCR improvements should focus on the 4 modelspace-text files (e.g., higher DPI or
  title-block cropping) if we want to avoid manual labels.
