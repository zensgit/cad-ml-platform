# DEV_GRAPH2D_STRICT_PROFILE_AND_SEED_SWEEP_20260215

## Goal

Implement and validate:

1. A reusable Graph2D strict training profile so the best-known local setup can be replayed with one flag.
2. A seed sweep runner to quantify stability (mean/min/max strict accuracy across multiple random seeds).

## Implementation

### 1) Reusable training profile in local pipeline

Updated `scripts/run_graph2d_pipeline_local.py`:

- Added `--training-profile` with choices:
  - `none` (default; existing behavior unchanged)
  - `strict_node23_edgesage_v1`
- Added `TRAINING_PROFILES` + `_apply_training_profile(args)` to inject opinionated settings.

`strict_node23_edgesage_v1` sets:

- `model=edge_sage`
- `node_dim=23`
- `hidden_dim=128`
- `epochs=10`
- `loss=focal`, `class_weighting=sqrt`, `sampler=balanced`
- `distill=true`, `teacher=titleblock`, `distill_alpha=0.1`, `distill_temp=3.0`
- strict diagnosis path defaults:
  - `student_geometry_only=true`
  - `diagnose_no_text_no_filename=true`
  - `normalize_labels=true`
  - `clean_min_count=5`
  - `dxf_enhanced_keypoints=true`
  - `dxf_edge_augment_knn_k=0`
  - `dxf_eps_scale=0.001`

Also added `training_profile` to `pipeline_summary.json`.

### 2) Seed sweep runner

Added `scripts/sweep_graph2d_profile_seeds.py`:

- Runs `scripts/run_graph2d_pipeline_local.py` for each seed under one profile.
- Outputs:
  - `seed_sweep_results.csv`
  - `seed_sweep_results.json`
  - `seed_sweep_summary.json` (mean/min/max strict accuracy)

## Tests

Added unit tests:

- `tests/unit/test_run_graph2d_pipeline_local_profile.py`
  - validates profile application (`none` and `strict_node23_edgesage_v1`)
- `tests/unit/test_sweep_graph2d_profile_seeds.py`
  - validates seed list parsing

Validation run:

- `.venv/bin/python -m pytest tests/unit/test_run_graph2d_pipeline_local_profile.py tests/unit/test_sweep_graph2d_profile_seeds.py tests/unit/test_run_graph2d_pipeline_local_distill_wiring.py tests/unit/test_run_graph2d_pipeline_local_diagnose_strict_wiring.py -v`
- Result: all passed.

## Seed Sweep Verification

Command:

```bash
.venv/bin/python scripts/sweep_graph2d_profile_seeds.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --training-profile strict_node23_edgesage_v1 \
  --seeds 7,21,42
```

Artifact root:

- `/tmp/graph2d_profile_seed_sweep_20260215_013344`

Strict accuracy by seed:

- seed `7`: `0.3818`
- seed `21`: `0.2818`
- seed `42`: `0.3545`

Aggregate (`seed_sweep_summary.json`):

- mean: `0.339394`
- min: `0.281818`
- max: `0.381818`

## Conclusion

- The strict profile is now reusable from a single flag and preserves backward compatibility (`none` remains default).
- Multi-seed results confirm performance gains over the old strict baseline (`0.2364`), with observable seed variance; current validated range is ~`0.28` to `0.38`.
