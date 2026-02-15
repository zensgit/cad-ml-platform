# DEV_GRAPH2D_SEED_GATE_REGRESSION_THRESHOLDS_CONFIG_20260215

## Goal

Continue hardening Graph2D seed-gate regression checks by moving threshold values from script defaults to a dedicated config file with channel-aware overrides.

## Implementation

### 1) Threshold config file

Added:

- `config/graph2d_seed_gate_regression.yaml`

Structure:

- global defaults under `graph2d_seed_gate_regression`,
- optional per-channel overrides in `channels.standard` / `channels.strict`.

### 2) Regression checker enhancement

Updated `scripts/ci/check_graph2d_seed_gate_regression.py`:

- added `--config` (default: `config/graph2d_seed_gate_regression.yaml`),
- loads yaml thresholds by section `graph2d_seed_gate_regression`,
- resolves thresholds in precedence order:
  - built-in defaults
  - config global values
  - config channel values
  - CLI explicit overrides
- writes `threshold_source` into report json:
  - config path,
  - whether config loaded,
  - applied CLI overrides.

### 3) CI and Make integration

Updated:

- `Makefile`
  - regression targets pass `--config $GRAPH2D_SEED_GATE_REGRESSION_CONFIG` (default to config file).
- `.github/workflows/ci.yml`
  - standard/strict regression steps now pass `--config config/graph2d_seed_gate_regression.yaml`.

### 4) Summary rendering update

Updated `scripts/ci/summarize_graph2d_seed_gate_regression.py`:

- added `Threshold source` row in markdown summary.

## Tests

Updated/added tests:

- `tests/unit/test_graph2d_seed_gate_regression_check.py`
  - added threshold resolution precedence test.
- `tests/unit/test_graph2d_seed_gate_regression_summary.py`
  - added threshold-source assertions.

Validation:

```bash
pytest tests/unit/test_graph2d_seed_gate_regression_check.py \
       tests/unit/test_graph2d_seed_gate_regression_summary.py \
       tests/unit/test_graph2d_seed_gate_baseline_update.py \
       tests/unit/test_graph2d_seed_gate_summary.py \
       tests/unit/test_graph2d_seed_gate_trend.py \
       tests/unit/test_sweep_graph2d_profile_seeds.py -q
```

Result: `20 passed`.

## Runtime verification

```bash
make validate-graph2d-seed-gate-regression
make validate-graph2d-seed-gate-strict-regression
```

Result:

- both checks passed,
- output includes:
  - `"threshold_source": {"config":"config/graph2d_seed_gate_regression.yaml","config_loaded":true,"cli_overrides":{}}`

