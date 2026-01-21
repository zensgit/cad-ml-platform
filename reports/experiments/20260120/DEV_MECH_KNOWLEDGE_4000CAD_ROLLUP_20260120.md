# DEV_MECH_KNOWLEDGE_4000CAD_ROLLUP_20260120

## Summary
Consolidated the 4000CAD Graph2D pipeline: expanded labeled data, merged low-frequency
labels, and selected the merged-label checkpoint as the default model after A/B
comparisons.

## Data & Labels
- Manifest (baseline): `reports/MECH_4000_DWG_LABEL_MANIFEST_20260119.csv` (223 rows).
- Merged manifest: `reports/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260120.csv` (84 labels).
- Merged2 manifest: `reports/MECH_4000_DWG_LABEL_MANIFEST_MERGED2_20260120.csv` (74 labels).
- Merge mapping: `reports/MECH_4000_DWG_LABEL_MERGE_SUGGESTIONS_20260120.csv`.

## Models
- Default: `models/graph2d_merged_latest.pth`
- Previous: `models/graph2d_latest.pth`
- Experimental: `models/experimental/graph2d_merged2_latest.pth`

## Evaluations
- Full vs merged (merged manifest): merged Top-1/Top-3 higher.
- Merged vs merged2: merged still best.
- Label diagnostic shows confusion dominated by `机械制图` class.

## Key Reports
- A/B merged eval: `reports/DEV_MECH_KNOWLEDGE_4000CAD_AB_MERGED_EVAL_20260120.md`
- A/B merged2 eval: `reports/DEV_MECH_KNOWLEDGE_4000CAD_AB_MERGED2_EVAL_20260120.md`
- Label diagnostic: `reports/DEV_MECH_KNOWLEDGE_4000CAD_LABEL_DIAGNOSTIC_20260120.md`
- Default model (merged): `reports/DEV_MECH_KNOWLEDGE_4000CAD_DEFAULT_MODEL_MERGED_20260120.md`

## Validation
- `GRAPH2D_ENABLED=true GRAPH2D_MODEL_PATH=models/graph2d_merged_latest.pth pytest tests/integration/test_analyze_dxf_fusion.py -v` (passed).

## Recommendation
- Keep the merged-label model as default and continue cleaning labels that collapse into
  `机械制图`.
