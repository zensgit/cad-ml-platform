# DEV_GRAPH2D_STRICT_SWEEP_RUNNER_20260214

## Goal

Make strict-mode Graph2D experimentation repeatable by:

1. Allowing the local pipeline runner to train/eval with **geometry-only student graphs** (DXF text stripped).
2. Providing a sweep runner that executes a small grid of pipeline configs and aggregates strict-mode metrics.

Strict mode is defined as:
- Inference strips DXF text/annotation entities.
- Inference masks the filename (always `"masked.dxf"`).

## Changes

### Pipeline Runner: Geometry-Only Student Flag

File: `scripts/run_graph2d_pipeline_local.py`

- Added flag: `--student-geometry-only`
  - Sets `DXF_STRIP_TEXT_ENTITIES=true` so the student graph builder strips text entities during **train/eval**.
  - `pipeline_summary.json` records `graph_build.student_geometry_only`.

### New Strict Sweep Runner

File: `scripts/sweep_graph2d_strict_mode.py`

- Runs multiple `scripts/run_graph2d_pipeline_local.py` configurations in sequence.
- Forces strict diagnosis for every run:
  - `--diagnose-no-text-no-filename` (pipeline passes `--strip-text-entities --mask-filename` to diagnosis)
- Writes aggregated results:
  - `<work_root>/sweep_results.csv`
  - `<work_root>/sweep_results.json`

Notes:
- Work roots live in `/tmp` by default and contain local file names/paths; do not commit them.
- The runner shares a disk graph cache across runs via `--graph-cache both`.

## Unit Tests

- Added: `tests/unit/test_sweep_graph2d_strict_mode_parsers.py`
  - Verifies sweep CSV/JSON parsing helpers.
- Existing pipeline wiring tests remain valid:
  - `tests/unit/test_run_graph2d_pipeline_local_distill_wiring.py`
  - `tests/unit/test_run_graph2d_pipeline_local_diagnose_strict_wiring.py`

## Validation

### Unit Tests

```bash
.venv/bin/python -m pytest \
  tests/unit/test_sweep_graph2d_strict_mode_parsers.py \
  tests/unit/test_run_graph2d_pipeline_local_distill_wiring.py \
  tests/unit/test_run_graph2d_pipeline_local_diagnose_strict_wiring.py \
  -v
```

### Local Strict Sweep (Training Drawings)

```bash
/usr/bin/time -p .venv/bin/python scripts/sweep_graph2d_strict_mode.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --normalize-labels \
  --clean-min-count 5 \
  --student-geometry-only \
  --epochs 3 \
  --diagnose-max-files 200
```

Observed:
- Artifacts: `/tmp/graph2d_strict_sweep_20260214_114741`
- Aggregated metrics: `/tmp/graph2d_strict_sweep_20260214_114741/sweep_results.csv`
- Timing: `real 451.11s`

