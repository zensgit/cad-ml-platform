# DEV_MECH_KNOWLEDGE_4000CAD_AB_MERGED_EVAL_20260120

## Summary
Compared graph2d predictions on the merged manifest using the full and merged checkpoints.
Full-model predictions were mapped through the merge map for fair comparison.

## Metrics
- full: total=223, top1=72 (0.323), top3=95 (0.426), empty_pred=2, missing_dxf=0, parse_error=0
- merged: total=223, top1=79 (0.354), top3=105 (0.471), empty_pred=2, missing_dxf=0, parse_error=0

## Outputs
- reports/MECH_4000_DWG_AB_MERGED_PRED_20260120.csv

## Recommendation
- Prefer the merged checkpoint (`models/graph2d_merged_latest.pth`) as default for the
  consolidated label set.
