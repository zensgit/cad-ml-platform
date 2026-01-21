# DEV_MECH_KNOWLEDGE_4000CAD_AB_MERGED2_EVAL_20260120

## Summary
Compared graph2d predictions on the merged2 manifest using the merged and merged2 checkpoints.

## Metrics
- merged: total=223, top1=123 (0.552), top3=138 (0.619), empty_pred=2, missing_dxf=0, parse_error=0
- merged2: total=223, top1=121 (0.543), top3=138 (0.619), empty_pred=2, missing_dxf=0, parse_error=0

## Outputs
- reports/MECH_4000_DWG_AB_MERGED2_PRED_20260120.csv

## Recommendation
- Keep `models/graph2d_merged_latest.pth` as default; merged2 does not improve metrics.
