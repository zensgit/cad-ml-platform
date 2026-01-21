# DEV_MECH_KNOWLEDGE_4000CAD_MODEL_CLEANUP_20260120

## Summary
Finalized the default Graph2D model selection and moved experimental checkpoints
into a dedicated folder for clarity.

## Actions
- Kept default model: `models/graph2d_merged_latest.pth`
- Moved experimental checkpoint:
  - `models/graph2d_merged2_latest.pth` → `models/experimental/graph2d_merged2_latest.pth`

## Notes
- The merged2 checkpoint remains available for reference but is not recommended
  as the default model.
