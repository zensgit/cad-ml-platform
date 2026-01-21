# DEV_MECH_KNOWLEDGE_4000CAD_SAMPLE_EXPANSION_20260120B

## Summary
Expanded the 4000CAD training pool again by converting additional DWG drawings with
Chinese labels in filenames, then appended them to the weak-label manifest and
regenerated geometry rules.

## Steps
- Selected 120 new DWG files with non-empty `label_cn` in the filename.
- Converted DWG → DXF into `.../机械CAD图纸_dxf` using the configured converter.
- Appended successful conversions to `reports/MECH_4000_DWG_LABEL_MANIFEST_20260119.csv`.
- Regenerated geometry rules from the updated manifest.

## Results
- Conversion log: `reports/MECH_4000_DWG_TO_DXF_LOG_20260120B.csv` (120 ok, 0 errors).
- Manifest size: 223 labeled rows total.
- Geometry rules: +42 dataset rules (total 185).

## Outputs
- `reports/MECH_4000_DWG_LABEL_MANIFEST_20260119.csv`
- `reports/MECH_4000_DWG_TO_DXF_LOG_20260120B.csv`
- `data/knowledge/geometry_rules.json`
