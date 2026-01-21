# DEV_MECH_KNOWLEDGE_4000CAD_AB_EVAL_20260120

## Summary
Compared graph2d predictions on the freq>=2 manifest using the full and freq2 checkpoints.

## Metrics
- full: total=130, top1=72 (0.554), top3=93 (0.715), empty_pred=2, missing_dxf=0, parse_error=0
- freq2: total=130, top1=71 (0.546), top3=93 (0.715), empty_pred=2, missing_dxf=0, parse_error=0

## Outputs
- reports/MECH_4000_DWG_AB_PRED_20260120.csv

## Recommendation
- Prefer the `full` checkpoint as default; it edges out top-1 accuracy while matching
  top-3 and empty-prediction rates.
