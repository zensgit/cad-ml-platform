# DEV_GRAPH2D_DISTILLATION_HARD_LOSS_BALANCING_20260214

## Goal

Make Graph2D knowledge distillation respect the selected **hard-label loss** strategy
(class weights / focal / logit-adjusted) instead of always using plain cross-entropy.

This keeps distillation experiments meaningful when iterating on strict-mode accuracy.

## Changes

### DistillationLoss: Support Custom Hard-Label Loss

File: `src/ml/knowledge_distillation.py`

- Extended `DistillationLoss` with an optional `hard_loss_fn`.
- When provided, the distillation hard-loss term uses `hard_loss_fn(student_logits, hard_labels)`.
- Default behavior remains unchanged (falls back to `torch.nn.functional.cross_entropy`).

### Train Script: Pass Balanced Criterion Into DistillationLoss

File: `scripts/train_2d_graph.py`

- The training script already builds a `criterion` using `ClassBalancer` based on:
  - `--loss` (`cross_entropy|focal|logit_adjusted`)
  - `--class-weighting` (`none|inverse|sqrt`)
- When `--distill` is enabled, the script now instantiates:
  - `DistillationLoss(..., hard_loss_fn=criterion)`

## Unit Tests

- Added: `tests/unit/test_knowledge_distillation_loss_hard_loss_fn.py`
  - Validates `hard_loss_fn` is used when `alpha=1.0`.

## Validation

### Unit Tests

```bash
.venv/bin/python -m pytest \
  tests/unit/test_knowledge_distillation_loss_hard_loss_fn.py \
  -v
```

### Strict Sweep Regression (Coarse Buckets, Geometry-Only Student)

Command:

```bash
/usr/bin/time -p .venv/bin/python scripts/sweep_graph2d_strict_mode.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --normalize-labels \
  --clean-min-count 5 \
  --student-geometry-only \
  --epochs 3 \
  --diagnose-max-files 200 \
  --max-runs 6
```

Observed:
- Artifacts: `/tmp/graph2d_strict_sweep_20260214_115800`
- Aggregated metrics: `/tmp/graph2d_strict_sweep_20260214_115800/sweep_results.csv`
- Key result (strict accuracy):
  - `distill_titleblock_alpha_0_1_focal`: `0.2273`

## Notes / Caveats

- This change makes `--loss`/`--class-weighting` affect distillation runs via the hard-loss term.
- The KL term still depends on teacher quality. If titleblock parsing yields mostly `"no_match"`,
  teacher logits will be near-uniform and provide limited benefit.

