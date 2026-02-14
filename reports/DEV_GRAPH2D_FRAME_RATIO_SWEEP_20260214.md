# DEV_GRAPH2D_FRAME_RATIO_SWEEP_20260214

## Goal
Tune `DXF_FRAME_PRIORITY_RATIO` for **strict-mode** Graph2D evaluation:
- student graphs built with `DXF_STRIP_TEXT_ENTITIES=true` (geometry-only)
- diagnose runs with `--strip-text-entities --mask-filename`

This isolates whether border/titleblock "frame" entities should be capped more aggressively.

## Setup
- DXF set: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf` (110 files)
- Coarse bucket labels: `--normalize-labels --clean-min-count 5`
- Model/training config held constant across runs:
  - `--distill --teacher titleblock --distill-alpha 0.1`
  - `--loss focal --class-weighting sqrt --sampler balanced`
  - `--epochs 3 --seed 42`

Artifacts root:
- `/tmp/graph2d_frame_ratio_sweep_20260214_125929`

## Commands
```bash
# ratio=0.0
.venv/bin/python scripts/run_graph2d_pipeline_local.py \
  --work-dir /tmp/graph2d_frame_ratio_sweep_20260214_125929/ratio_0_0 \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --epochs 3 --batch-size 4 --hidden-dim 64 --lr 0.001 \
  --graph-cache both --graph-cache-dir /tmp/graph2d_frame_ratio_sweep_20260214_125929/graph_cache \
  --student-geometry-only \
  --normalize-labels --clean-min-count 5 \
  --distill --teacher titleblock --distill-alpha 0.1 \
  --loss focal --class-weighting sqrt --sampler balanced \
  --diagnose-max-files 200 --diagnose-no-text-no-filename \
  --dxf-frame-priority-ratio 0.0

# ratio=0.05
.venv/bin/python scripts/run_graph2d_pipeline_local.py \
  --work-dir /tmp/graph2d_frame_ratio_sweep_20260214_125929/ratio_0_05 \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --epochs 3 --batch-size 4 --hidden-dim 64 --lr 0.001 \
  --graph-cache both --graph-cache-dir /tmp/graph2d_frame_ratio_sweep_20260214_125929/graph_cache \
  --student-geometry-only \
  --normalize-labels --clean-min-count 5 \
  --distill --teacher titleblock --distill-alpha 0.1 \
  --loss focal --class-weighting sqrt --sampler balanced \
  --diagnose-max-files 200 --diagnose-no-text-no-filename \
  --dxf-frame-priority-ratio 0.05

# ratio=0.1
.venv/bin/python scripts/run_graph2d_pipeline_local.py \
  --work-dir /tmp/graph2d_frame_ratio_sweep_20260214_125929/ratio_0_1 \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --epochs 3 --batch-size 4 --hidden-dim 64 --lr 0.001 \
  --graph-cache both --graph-cache-dir /tmp/graph2d_frame_ratio_sweep_20260214_125929/graph_cache \
  --student-geometry-only \
  --normalize-labels --clean-min-count 5 \
  --distill --teacher titleblock --distill-alpha 0.1 \
  --loss focal --class-weighting sqrt --sampler balanced \
  --diagnose-max-files 200 --diagnose-no-text-no-filename \
  --dxf-frame-priority-ratio 0.1

# ratio=0.2
.venv/bin/python scripts/run_graph2d_pipeline_local.py \
  --work-dir /tmp/graph2d_frame_ratio_sweep_20260214_125929/ratio_0_2 \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --epochs 3 --batch-size 4 --hidden-dim 64 --lr 0.001 \
  --graph-cache both --graph-cache-dir /tmp/graph2d_frame_ratio_sweep_20260214_125929/graph_cache \
  --student-geometry-only \
  --normalize-labels --clean-min-count 5 \
  --distill --teacher titleblock --distill-alpha 0.1 \
  --loss focal --class-weighting sqrt --sampler balanced \
  --diagnose-max-files 200 --diagnose-no-text-no-filename \
  --dxf-frame-priority-ratio 0.2
```

## Results (Strict Diagnose)
Strict accuracy is taken from `diagnose/summary.json` for each run.

| DXF_FRAME_PRIORITY_RATIO | strict_accuracy | strict_conf_p50 | top_pred |
|---:|---:|---:|---|
| 0.0  | 0.2091 | 0.1405 | 传动件 |
| 0.05 | 0.2091 | 0.1406 | 传动件 |
| 0.1  | 0.2091 | 0.1424 | 传动件 |
| 0.2  | 0.1909 | 0.1423 | 传动件 |

## Conclusion
- For this dataset + config, `DXF_FRAME_PRIORITY_RATIO` in `[0.0, 0.1]` performed the same.
- Allowing more frame entities (`0.2`) regressed strict accuracy back toward the collapsed baseline.
- Keep the geometry-only default at `0.1` (current behavior) or consider lowering only if needed for other corpora.
