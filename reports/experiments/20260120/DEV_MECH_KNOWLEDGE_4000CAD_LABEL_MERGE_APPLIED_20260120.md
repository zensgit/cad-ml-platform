# DEV_MECH_KNOWLEDGE_4000CAD_LABEL_MERGE_APPLIED_20260120

## Summary
Applied the label-merge suggestions to produce a consolidated manifest with fewer
classes, and pruned dataset-derived geometry rules to match the merged label set.

## Steps
- Applied `reports/MECH_4000_DWG_LABEL_MERGE_SUGGESTIONS_20260120.csv` to the
  manifest, merging low-frequency labels by keyword groups.
- Wrote the merged manifest to a new CSV.
- Pruned `data/knowledge/geometry_rules.json` to remove obsolete
  `dataset_manifest` rules, then regenerated rules from the merged manifest.

## Results
- Manifest rows: 223
- Unique labels: 114 → 84
- Merge map entries: 50
- Geometry rules: 109 total (post-prune + regenerate)

## Outputs
- `reports/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260120.csv`
- `data/knowledge/geometry_rules.json`
