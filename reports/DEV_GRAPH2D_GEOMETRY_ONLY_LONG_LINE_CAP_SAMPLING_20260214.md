# DEV_GRAPH2D_GEOMETRY_ONLY_LONG_LINE_CAP_SAMPLING_20260214

## Goal
Further harden Graph2D strict-mode (no DXF text + masked filename) against sampling collapse by limiting how many **non-frame long LINE entities** can dominate a sampled graph.

Rationale: even after capping border/titleblock frame entities, geometry-only graphs can still be dominated by long straight lines (often outlines / repetitive structure). We want to preserve higher-information entities (CIRCLE/ARC/POLYLINE/etc.) and keep graphs more diverse.

## Changes
### 1) ImportanceSampler: cap non-frame long lines
- Added `DXF_LONG_LINE_RATIO` (0..1) to cap the fraction of sampled nodes that are **non-frame long lines**.
  - long line := `dtype=LINE` and `length >= 0.5 * max_dim` (same threshold used by existing priority logic)
  - non-frame := not `border_hint` and not `title_block_hint`
  - default: `1.0` (no cap)

Behavior:
- When `DXF_LONG_LINE_RATIO < 1.0`, we fill sampled nodes by:
  1. text (capped)
  2. frame (capped)
  3. non-frame non-long-line geometry
  4. long lines (capped)
  5. overflow long lines only as a last-resort fallback to keep graph size stable

Code: `src/ml/importance_sampler.py`

### 2) Cache key isolation
Included `DXF_LONG_LINE_RATIO` in DXF graph cache keys.

Code: `src/ml/train/dataset_2d.py`

### 3) Training/eval/pipeline wiring
- Added CLI flag `--dxf-long-line-ratio` to:
  - `scripts/train_2d_graph.py`
  - `scripts/eval_2d_graph.py`
  - `scripts/run_graph2d_pipeline_local.py`
- For `scripts/run_graph2d_pipeline_local.py --student-geometry-only`, default `--dxf-long-line-ratio=0.4` unless explicitly set.

## Verification
### Unit tests
```bash
.venv/bin/python -m pytest tests/unit/test_importance_sampler_long_line_ratio.py -v
.venv/bin/python -m pytest tests/unit/test_importance_sampler_frame_ratio.py -q
```

### Micro-sweep (local DXFs)
Command pattern:
```bash
.venv/bin/python scripts/run_graph2d_pipeline_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --student-geometry-only \
  --normalize-labels --clean-min-count 5 \
  --distill --teacher titleblock --distill-alpha 0.1 \
  --loss focal --class-weighting sqrt --sampler balanced \
  --diagnose-no-text-no-filename \
  --dxf-frame-priority-ratio 0.1 \
  --dxf-long-line-ratio <value>
```

Artifacts:
- `/tmp/graph2d_long_line_ratio_sweep_20260214_131521`

Observed strict accuracies (strip DXF text + mask filename):
- `dxf_long_line_ratio=1.0`: `0.2091`
- `dxf_long_line_ratio=0.6`: `0.2091`
- `dxf_long_line_ratio=0.4`: `0.2091`
- `dxf_long_line_ratio=0.2`: `0.2091`

## Notes
- On this dataset/config, strict accuracy did not change across the tested long-line caps. The knob is kept as a safety valve for other corpora where long-line dominance is stronger.
