# DEV_MECH_KNOWLEDGE_4000CAD_DEFAULT_MODEL_MERGED_20260120

## Summary
Updated the default Graph2D model to use the merged-label checkpoint based on the
A/B evaluation against the full model.

## Decision
- Default model: `models/graph2d_merged_latest.pth`
- Previous default: `models/graph2d_latest.pth` (kept as fallback)

## Updates
- `.env.example` now points `GRAPH2D_MODEL_PATH` at the merged checkpoint.

## Reference
- A/B report: `reports/DEV_MECH_KNOWLEDGE_4000CAD_AB_MERGED_EVAL_20260120.md`
