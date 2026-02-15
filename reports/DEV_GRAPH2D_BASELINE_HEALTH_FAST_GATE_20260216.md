# DEV_GRAPH2D_BASELINE_HEALTH_FAST_GATE_20260216

## Goal

Add a fast baseline-health gate that does not depend on current seed sweep outputs, so CI can fail fast on stale/corrupt baseline metadata before expensive Graph2D training/evaluation.

## Implementation

### 1) Baseline health mode in regression checker

Updated `scripts/ci/check_graph2d_seed_gate_regression.py`:

- `--summary-json` is now optional,
- added `--use-baseline-as-current`:
  - uses baseline channel metrics as `current`,
  - enables baseline-only health checks.
- report `threshold_source` now includes `current_source`.

### 2) Make target

Updated `Makefile`:

- added `validate-graph2d-seed-gate-baseline-health`
  - runs baseline health check for both `standard` and `strict` channels.

### 3) CI integration (Python 3.11 lane)

Updated `.github/workflows/ci.yml`:

- added pre-seed-gate baseline health check step:
  - runs standard + strict checks using `--use-baseline-as-current`,
  - writes two report json files,
  - uploads logs/reports artifacts,
  - appends two markdown summaries to `GITHUB_STEP_SUMMARY`.

### 4) Regression summary enhancement

Updated `scripts/ci/summarize_graph2d_seed_gate_regression.py`:

- threshold-source row now shows `current=<...>` for easier audit.

## Tests

Updated tests:

- `tests/unit/test_graph2d_seed_gate_regression_check.py`
  - added `_resolve_current_summary` baseline-current mode case.
- `tests/unit/test_graph2d_seed_gate_regression_summary.py`
  - asserts `current_source` appears in summary output.

Validation:

```bash
pytest tests/unit/test_graph2d_seed_gate_regression_check.py \
       tests/unit/test_graph2d_seed_gate_regression_summary.py \
       tests/unit/test_graph2d_seed_gate_baseline_update.py \
       tests/unit/test_graph2d_seed_gate_summary.py \
       tests/unit/test_graph2d_seed_gate_trend.py \
       tests/unit/test_sweep_graph2d_profile_seeds.py -q
```

Result: `26 passed`.

## Runtime verification

```bash
make validate-graph2d-seed-gate-baseline-health
make validate-graph2d-seed-gate-regression
make validate-graph2d-seed-gate-strict-regression
```

Result:

- all checks passed,
- baseline-health summary confirms:
  - `current=baseline_channel`
  - date freshness ok
  - snapshot exists
  - snapshot metrics/hash match.

