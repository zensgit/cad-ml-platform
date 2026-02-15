# DEV_GRAPH2D_SEED_GATE_STABLE_BASELINE_PATH_20260215

## Goal

Stabilize Graph2D seed-gate baseline management so CI and Make targets no longer depend on a hardcoded date path.

## Implementation

### 1) Stable baseline file

Added:

- `config/graph2d_seed_gate_baseline.json`

This becomes the canonical baseline file used by default in local/CI regression checks.

### 2) Baseline update utility

Added:

- `scripts/ci/update_graph2d_seed_gate_baseline.py`

Function:

- reads current standard/strict seed-gate summaries,
- writes canonical baseline:
  - `config/graph2d_seed_gate_baseline.json`,
- writes dated snapshot:
  - `reports/experiments/<YYYYMMDD>/graph2d_seed_gate_baseline_snapshot_<YYYYMMDD>.json`.

### 3) Makefile integration

Updated:

- default baseline path for:
  - `validate-graph2d-seed-gate-regression`
  - `validate-graph2d-seed-gate-strict-regression`
  now points to `config/graph2d_seed_gate_baseline.json`.
- added:
  - `update-graph2d-seed-gate-baseline`

### 4) CI integration

Updated `.github/workflows/ci.yml`:

- standard/strict baseline regression checks now use:
  - `config/graph2d_seed_gate_baseline.json`

## Tests

Added:

- `tests/unit/test_graph2d_seed_gate_baseline_update.py`

Validation:

```bash
pytest tests/unit/test_graph2d_seed_gate_baseline_update.py \
       tests/unit/test_graph2d_seed_gate_regression_check.py \
       tests/unit/test_graph2d_seed_gate_regression_summary.py \
       tests/unit/test_graph2d_seed_gate_summary.py \
       tests/unit/test_graph2d_seed_gate_trend.py \
       tests/unit/test_sweep_graph2d_profile_seeds.py -q
```

Result: `19 passed`.

## Runtime verification

```bash
make validate-graph2d-seed-gate-regression
make validate-graph2d-seed-gate-strict-regression
make update-graph2d-seed-gate-baseline
```

Result:

- regression checks passed for both channels,
- baseline update command successfully wrote:
  - `config/graph2d_seed_gate_baseline.json`
  - `reports/experiments/20260215/graph2d_seed_gate_baseline_snapshot_20260215.json`

