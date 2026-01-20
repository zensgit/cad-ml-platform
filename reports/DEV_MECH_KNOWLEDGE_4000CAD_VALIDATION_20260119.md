# DEV_MECH_KNOWLEDGE_4000CAD_VALIDATION_20260119

## Summary
Validated updated geometry rules, DWG->DXF conversion, and graph2d weak-label baseline
for the 4000CAD dataset slice.

## Tests
- `pytest tests/unit/test_geometry_rules_dataset.py -v`

## Conversion Validation
- `python3 scripts/convert_dwg_batch.py --input-dir ... --input-dir ... --recursive --output-dir /Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf --log-csv reports/MECH_4000_DWG_TO_DXF_LOG_20260119.csv`
- Result: 50/50 conversions succeeded.

## Graph2D Weak-Label Baseline
- Sample size: 20 (seed=17)
- Top-1 accuracy: 0.15
- Top-3 accuracy: 0.20
- Labels are derived from filenames (weak labels).

## Manual-Eval Baseline (Auto-Filled)
- Sample size: 20
- Top-1 accuracy: 0.05
- Top-3 accuracy: 0.15
- Reviewer labels were auto-filled from filenames (not human verified).

## Auto-Label Attempt
- Unlabeled rows: 27
- Auto-labeled rows: 27 (rule + heuristic + filename fallbacks)
- Report: `reports/DEV_MECH_KNOWLEDGE_4000CAD_AUTO_LABEL_ATTEMPT_20260119.md`
