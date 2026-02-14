# DEV_GRAPH2D_GEOMETRY_ONLY_FRAME_CAP_SAMPLING_20260214

## Goal
Reduce Graph2D strict-mode collapse when running in production-like conditions:
- DXF text/annotation entities stripped
- filename masked

In this mode, importance sampling was frequently dominated by border/titleblock "frame" entities (highly repetitive), making graphs too similar.

## Changes
### 1) ImportanceSampler: cap frame entities
- Added `DXF_FRAME_PRIORITY_RATIO` (0..1) to cap the fraction of sampled nodes that are frame entities:
  - frame entities := `border_hint` or `title_block_hint`
  - default `1.0` (no cap)
- Fixed the sampling logic so the cap is *actually enforced*:
  - after selecting `frame_entities[:max_frame]`, we now prefer filling remaining slots with **non-frame geometry**;
  - remaining frame entities are used only as a fallback if we still need to fill slots.

Code: `src/ml/importance_sampler.py`

### 2) Cache key isolation
Included `DXF_FRAME_PRIORITY_RATIO` in the DXF graph cache key to prevent cache collisions.

Code: `src/ml/train/dataset_2d.py`

### 3) Training/eval/pipeline wiring
- Added CLI flag `--dxf-frame-priority-ratio` to:
  - `scripts/train_2d_graph.py`
  - `scripts/eval_2d_graph.py`
  - `scripts/run_graph2d_pipeline_local.py`
- When running `scripts/run_graph2d_pipeline_local.py --student-geometry-only`, default `--dxf-frame-priority-ratio=0.1` unless explicitly set.

## Verification
### Unit tests
```bash
.venv/bin/python -m pytest tests/unit/test_importance_sampler_frame_ratio.py -v
.venv/bin/python -m pytest \
  tests/unit/test_sweep_graph2d_strict_mode_parsers.py \
  tests/unit/test_knowledge_distillation_loss_hard_loss_fn.py -v
```

### Strict sweep (local DXFs)
Command:
```bash
.venv/bin/python scripts/sweep_graph2d_strict_mode.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --normalize-labels \
  --clean-min-count 5 \
  --student-geometry-only \
  --epochs 3 \
  --diagnose-max-files 200 \
  --max-runs 6
```

Artifacts:
- `/tmp/graph2d_strict_sweep_20260214_123346`

Observed strict accuracies (strip DXF text + mask filename):
- `baseline_ce_none_sampler_none`: `0.1909` (collapsed to `传动件`)
- `distill_titleblock_alpha_0_1_focal`: `0.2091` (best in this sweep)

## Notes
- This change is intended to improve *signal diversity* in geometry-only graphs; it does not replace the need for stronger geometry features or larger training sets.
- The new cap is configurable (env/CLI) and defaults to "no cap" unless running geometry-only via the local pipeline helper.
