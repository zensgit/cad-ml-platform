# DEV_MECH_KNOWLEDGE_2D_GRAPH_SYNONYMS_20260119

## Summary
- Generated a 47-label English synonym glossary from the DWG manifest.
- Updated dataset-derived geometry rules to include bilingual keywords.
- Added de-hyphenated keyword variants for better substring matching.
- Re-validated rule matching with unit tests.

## Steps
- `python3 scripts/build_label_synonyms_template.py`
- Populated `data/knowledge/label_synonyms_template.json` with English synonyms.
- Updated `.gitignore` to track `data/knowledge/*.json` while keeping other data ignored.
- `python3 scripts/build_geometry_rules_from_manifest.py --synonyms-json data/knowledge/label_synonyms_template.json`
- `pytest tests/unit/test_geometry_rules_dataset.py -v`
  - Re-ran after adding de-hyphenated keyword variants.

## Results
- Rules update: 0 added, 47 updated (synonyms merged into keywords), then 4 updated (de-hyphenated variants).
- Unit tests: 2 passed.

## Notes
- English synonyms are draft translations derived from drawing labels and should be reviewed for domain accuracy.
