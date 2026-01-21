# DEV_MECH_KNOWLEDGE_4000CAD_AUTO_LABEL_UPDATE_20260120

## Summary
Improved DXF text normalization for CJK spacing, expanded heuristic suffixes, and raised
confidence for numeric filename fallbacks so that all remaining unlabeled 4000CAD rows
can be auto-labeled. Added new label synonyms and rebuilt geometry rules. Exported a
small candidate vocabulary from extracted DXF text.

## Steps
- `python3 scripts/auto_label_unlabeled_dxf.py --dxf-dir /Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf --min-confidence 0.7`
- `python3 scripts/build_geometry_rules_from_manifest.py --manifest reports/MECH_4000_DWG_LABEL_MANIFEST_20260119.csv --synonyms-json data/knowledge/label_synonyms_template.json`
- Generated DXF text vocab exports (full + candidate-only) from the same DXF directory.

## Results
- Unlabeled template now fully labeled (27/27 rows).
- New labels added from title-block text: `压盖`, `支腿`.
- Numeric filenames auto-labeled as `练习零件图` (confidence 0.70).
- Manifest updated to 50/50 labeled rows.
- Geometry rules updated with 2 new dataset rules (`压盖`, `支腿`).

## Outputs
- `reports/MECH_4000_DWG_UNLABELED_LABELS_TEMPLATE_20260119.csv`
- `reports/MECH_4000_DWG_LABEL_MANIFEST_20260119.csv`
- `reports/MECH_4000_DWG_TEXT_VOCAB_20260120.csv`
- `reports/MECH_4000_DWG_TEXT_VOCAB_CANDIDATES_20260120.csv`
- `data/knowledge/geometry_rules.json`

## Notes
- CJK spacing normalization enables matching of title-block text like “压 盖” and “轴 承 座”.
- Candidate vocabulary focuses on label-like suffixes and known labels for quick review.
