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
