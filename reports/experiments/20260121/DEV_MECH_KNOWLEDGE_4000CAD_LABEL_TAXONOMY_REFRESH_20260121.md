# DEV_MECH_KNOWLEDGE_4000CAD_LABEL_TAXONOMY_REFRESH_20260121

## Summary
- Reviewed merged manifest labels and produced label counts + training label map.
- Enriched English synonyms for all labels present in the merged manifest.
- Regenerated geometry rules using the merged manifest and updated synonyms.

## Inputs
- Manifest: `reports/experiments/20260121/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260121.csv`
- Synonyms: `data/knowledge/label_synonyms_template.json`

## Outputs
- Label counts: `reports/experiments/20260121/MECH_4000_DWG_LABEL_COUNTS_20260121.csv`
- Label map snapshot: `reports/experiments/20260121/MECH_4000_DWG_LABEL_MAP_20260121.csv`
- Synonym gaps: `reports/experiments/20260121/MECH_4000_DWG_LABEL_SYNONYMS_GAPS_20260121.csv`
- Geometry rules: `data/knowledge/geometry_rules.json`

## Key Stats
- Labeled samples: 223
- Unique labels: 33
- Top labels: 机械制图(101), 零件图(52), 装配图(16), 练习零件图(9), 模板(8)

## Commands
- `python3 scripts/build_geometry_rules_from_manifest.py --manifest reports/experiments/20260121/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260121.csv --synonyms-json data/knowledge/label_synonyms_template.json`

## Notes
- Synonym gaps for manifest labels resolved (no missing/empty entries).
- Geometry rules reported: 2 added, 0 updated, total 112 rules.
