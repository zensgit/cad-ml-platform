# DEV_MECH_KNOWLEDGE_4000CAD_AUTO_LABEL_ATTEMPT_20260119

## Summary
Auto-labeled the high-confidence subset of 27 previously unlabeled DWG files
using decoded DXF text + knowledge rules plus layout-aware OCR (paperspace +
model). Low-confidence filename fallbacks were cleared for manual review.
INSERT attribute text extraction was added to cover title-block fields stored
as block attributes.

## Steps
- `python3 scripts/auto_label_unlabeled_dxf.py --dxf-dir /Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf --min-confidence 0.7 --enable-ocr --ocr-layout both`

## Results
- Unlabeled rows: 27
- Auto-labeled rows (>= 0.7 confidence): 15
- Cleared for review: 12
- Output: `reports/MECH_4000_DWG_UNLABELED_LABELS_TEMPLATE_20260119.csv`

## Notes
- `label_confidence` + `auto_label_reason` are now recorded per row.
- PaddleOCR + PaddlePaddle + matplotlib are installed and OCR runs, but no text was extracted
  from the rendered DXFs (`ocr_used=0` on all rows) even with paperspace fallback.
- `ocr_used` + `ocr_confidence` + `ocr_layout` are recorded; confidence is 0 when no OCR text is returned.
- INSERT attribute text extraction did not change the labeled count in this slice.
- Low-confidence filename fallbacks (e.g., `练习零件图`) were cleared.
- Rule-based labels cover items like `差动机构`, `蜗杆`, `轴承座`.
