# DEV_MECH_KNOWLEDGE_4000CAD_FREQ_FILTER_20260120

## Summary
Filtered the 4000CAD manifest to keep labels with frequency >= 2, producing a
smaller, cleaner training set for the graph2d model.

## Steps
- Counted label frequencies across `reports/MECH_4000_DWG_LABEL_MANIFEST_20260119.csv`.
- Generated a filtered manifest including only labels with count >= 2.

## Results
- Original manifest: 223 rows, 114 unique labels.
- Filtered manifest: 130 rows, 21 unique labels.

## Output
- `reports/MECH_4000_DWG_LABEL_MANIFEST_FREQ2_20260120.csv`
