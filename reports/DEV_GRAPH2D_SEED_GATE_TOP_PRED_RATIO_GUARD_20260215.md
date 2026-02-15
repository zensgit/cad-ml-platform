# DEV_GRAPH2D_SEED_GATE_TOP_PRED_RATIO_GUARD_20260215

## Goal

Continue hardening Graph2D seed gate with a prediction-concentration guard:

- fail when top predicted label ratio is too high (collapse indicator),
- while retaining existing accuracy/diversity gates.

## Implementation

Updated `scripts/sweep_graph2d_profile_seeds.py`:

- Added per-run extracted metrics from diagnose summary:
  - `strict_top_pred_label`
  - `strict_top_pred_count`
  - `strict_top_pred_ratio`
- Added gate threshold:
  - `--max-strict-top-pred-ratio` (disabled by default)
- Gate now fails when any seed exceeds this ratio threshold.
- Added summary aggregates:
  - `strict_top_pred_ratio_mean`
  - `strict_top_pred_ratio_max`
- Added gate metadata:
  - `max_strict_top_pred_ratio`.

Updated configs:

- `config/graph2d_seed_gate.yaml`:
  - `max_strict_top_pred_ratio: 0.90`
- `config/graph2d_seed_gate_strict.yaml`:
  - `max_strict_top_pred_ratio: 0.90`

Updated CI summary renderer:

- `scripts/ci/summarize_graph2d_seed_gate.py`
  - new row: `Top-pred ratio (mean/max)`.

## Tests

Updated tests:

- `tests/unit/test_sweep_graph2d_profile_seeds.py`
  - added failure coverage for top-pred ratio threshold.
- `tests/unit/test_graph2d_seed_gate_summary.py`
  - added summary assertions for top-pred ratio row.

Validation:

```bash
.venv/bin/python -m pytest \
  tests/unit/test_sweep_graph2d_profile_seeds.py \
  tests/unit/test_graph2d_seed_gate_summary.py -q
```

Result: `12 passed`.

## Runtime Verification

### A) Standard channel

```bash
make validate-graph2d-seed-gate
```

Summary (`/tmp/graph2d-seed-gate/seed_sweep_summary.json`):

- strict mean/min/max: `0.362500 / 0.291667 / 0.433333`
- top-pred ratio mean/max: `0.600000 / 0.708333`
- distinct labels min/max: `5 / 5`
- gate: `passed=true`

### B) Strict channel

```bash
make validate-graph2d-seed-gate-strict
```

Summary (`/tmp/graph2d-seed-gate-strict/seed_sweep_summary.json`):

- strict mean/min/max: `0.945833 / 0.941667 / 0.950000`
- top-pred ratio mean/max: `0.258333 / 0.275000`
- distinct labels min/max: `5 / 5`
- gate: `passed=true`

### C) Summary output check

```bash
.venv/bin/python scripts/ci/summarize_graph2d_seed_gate.py \
  --summary-json /tmp/graph2d-seed-gate/seed_sweep_summary.json \
  --title "Graph2D Seed Gate Local Check"
```

Confirmed table includes `Top-pred ratio (mean/max)`.

## Notes

- `ezdxf` cache warnings still appear locally and do not affect gate results.
