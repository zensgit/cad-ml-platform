# DEV_GRAPH2D_SEED_GATE_BASELINE_REGRESSION_GUARD_20260215

## Goal

Continue hardening Graph2D seed gate by adding an automatic baseline regression check:

- compare current seed-gate summary against frozen baseline snapshot,
- fail CI when key metrics regress beyond configured tolerances,
- expose regression result in CI step summary and artifacts.

## Implementation

### 1) Regression checker

Added `scripts/ci/check_graph2d_seed_gate_regression.py`.

Inputs:

- `--summary-json`
- `--baseline-json`
- `--channel {standard|strict}`

Checked metrics:

- `strict_accuracy_mean` (min floor: baseline - allowed drop)
- `strict_accuracy_min` (min floor: baseline - allowed drop)
- `strict_top_pred_ratio_max` (max ceiling: baseline + allowed increase)
- `strict_low_conf_ratio_max` (max ceiling: baseline + allowed increase)
- `manifest_distinct_labels_min` (min floor: baseline - allowed drop)

Default tolerances:

- `max_accuracy_mean_drop=0.08`
- `max_accuracy_min_drop=0.08`
- `max_top_pred_ratio_increase=0.10`
- `max_low_conf_ratio_increase=0.05`
- `max_distinct_labels_drop=0`

Exit code:

- `0` pass
- `3` regression failed
- `2` input/baseline invalid

### 2) CI regression summary

Added `scripts/ci/summarize_graph2d_seed_gate_regression.py`.

- Renders markdown table from regression report json.
- Includes channel, pass/fail, baseline vs current metrics, tolerance config, and failure list.

### 3) Make targets

Updated `Makefile`:

- `validate-graph2d-seed-gate-regression`
- `validate-graph2d-seed-gate-strict-regression`

### 4) CI workflow integration

Updated `.github/workflows/ci.yml`:

- after standard seed gate:
  - run baseline regression check
  - upload regression log/report artifacts
  - append regression summary to `GITHUB_STEP_SUMMARY`
- after strict seed gate (optional channel):
  - run strict baseline regression check
  - upload regression log/report artifacts
  - append strict regression summary

## Tests

Added tests:

- `tests/unit/test_graph2d_seed_gate_regression_check.py`
- `tests/unit/test_graph2d_seed_gate_regression_summary.py`

Validation command:

```bash
pytest tests/unit/test_graph2d_seed_gate_regression_check.py \
       tests/unit/test_graph2d_seed_gate_regression_summary.py \
       tests/unit/test_graph2d_seed_gate_summary.py \
       tests/unit/test_graph2d_seed_gate_trend.py \
       tests/unit/test_sweep_graph2d_profile_seeds.py -q
```

Result: `18 passed`.

## Local runtime validation

### A) Direct script checks

```bash
.venv/bin/python scripts/ci/check_graph2d_seed_gate_regression.py \
  --summary-json /tmp/graph2d-seed-gate/seed_sweep_summary.json \
  --baseline-json reports/experiments/20260215/graph2d_seed_gate_baseline_snapshot_20260215.json \
  --channel standard \
  --output-json /tmp/graph2d-seed-gate/regression_check.json
```

```bash
.venv/bin/python scripts/ci/check_graph2d_seed_gate_regression.py \
  --summary-json /tmp/graph2d-seed-gate-strict/seed_sweep_summary.json \
  --baseline-json reports/experiments/20260215/graph2d_seed_gate_baseline_snapshot_20260215.json \
  --channel strict \
  --output-json /tmp/graph2d-seed-gate-strict/regression_check.json
```

Result: both `passed`.

### B) Make targets

```bash
make validate-graph2d-seed-gate-regression
make validate-graph2d-seed-gate-strict-regression
```

Result: both passed.

