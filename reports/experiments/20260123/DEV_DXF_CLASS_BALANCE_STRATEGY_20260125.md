# DEV_DXF_CLASS_BALANCE_STRATEGY_20260125

## Objective
Introduce configurable class-imbalance handling for Graph2D training and expand evaluation metrics to include Macro/Weighted F1.

## Implementation
- `src/ml/class_balancer.py`
  - Added configurable focal alpha/gamma and logit-adjustment tau.
  - Supports strategies: weights / focal / logit_adj.
- `scripts/train_2d_graph.py`
  - Uses `ClassBalancer` to select loss strategy based on CLI flags.
  - Logs imbalance ratio and chosen strategy.
  - Added `--focal-alpha` flag.
- `scripts/eval_2d_graph.py`
  - Added per-class precision/recall/F1.
  - Added macro_f1 and weighted_f1 summary metrics.

## Configuration
- `--loss` (cross_entropy | focal | logit_adjusted)
- `--class-weighting` (none | inverse | sqrt)
- `--focal-alpha`, `--focal-gamma`
- `--logit-adjustment-tau`

## Notes
- Weighting is applied only when requested; focal loss can run with or without class weights.
- Macro/Weighted F1 metrics are written into the evaluation CSV (overall row).
