# DEV_MECH_KNOWLEDGE_4000CAD_LABEL_SYNONYMS_AUTO_20260120

## Summary
Auto-filled English label synonyms using the OCR vocabulary as a whitelist of
Chinese mechanical terms plus a small CN→EN dictionary, then rebuilt the
geometry rules with the updated synonym list.

## Inputs
- OCR vocab: `reports/experiments/20260120/MECH_4000_DWG_TEXT_VOCAB_20260120.csv`
- Manifest: `reports/experiments/20260120/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260120.csv`
- Synonyms: `data/knowledge/label_synonyms_template.json`

## Updates
- Labels with synonyms: 37 / 84 (after auto-fill)
- Auto-fill summary: `reports/experiments/20260120/MECH_4000_DWG_LABEL_SYNONYMS_AUTO_20260120.csv`
- Full mapping snapshot: `reports/experiments/20260120/MECH_4000_DWG_LABEL_SYNONYMS_FINAL_20260120.csv`

## Geometry Rules
Command:
```
python3 scripts/build_geometry_rules_from_manifest.py \
  --synonyms-json data/knowledge/label_synonyms_template.json
```

Result:
- Added 10 rules, updated 23 rules, total rules: 109
