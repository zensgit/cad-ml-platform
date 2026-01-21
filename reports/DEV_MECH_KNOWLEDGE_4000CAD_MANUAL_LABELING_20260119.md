# DEV_MECH_KNOWLEDGE_4000CAD_MANUAL_LABELING_20260119

## Summary
Prepared manual labeling templates for the 4000CAD DWG slice, covering
unlabeled files and a 20-sample evaluation subset with model predictions.

## Outputs
- Unlabeled mapping template:
  `reports/MECH_4000_DWG_UNLABELED_LABELS_TEMPLATE_20260119.csv`
- Manual evaluation template:
  `reports/MECH_4000_DWG_MANUAL_EVAL_TEMPLATE_20260119.csv`
- Instructions:
  `reports/MECH_4000_DWG_MANUAL_EVAL_INSTRUCTIONS_20260119.md`

## Notes
- Fill `label_cn`/`label_en` for unlabeled files to expand the manifest.
- Fill `reviewer_label_cn` in the eval template to compute true Top-1/Top-3.
- Auto-labeling now keeps only high-confidence matches (>= 0.7): 15/27 rows.
  The remaining 12 rows are cleared for manual review (previously filename
  fallbacks like `练习零件图`).
- OCR metadata columns (`ocr_used`, `ocr_confidence`, `ocr_layout`) are available for triage.
  OCR ran (paperspace + model), but returned no text in this dataset slice (`ocr_used=0`).
- English labels are auto-filled when a synonym exists; verify terminology as needed.
- The manual-eval template has been auto-filled from filename labels to allow
  a provisional evaluation; replace with human labels when available.
