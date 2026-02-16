# DEV_GRAPH2D_BASELINE_CONTEXT_MISMATCH_SEVERITY_20260216

## Goal

Add graded handling for baseline-context drift in Graph2D seed-gate regression:

- `fail`: block gate,
- `warn`: pass with warnings,
- `ignore`: pass without warning.

## Implementation

### 1) Regression policy extension

Updated `scripts/ci/check_graph2d_seed_gate_regression.py`:

- Added policy key:
  - `context_mismatch_mode` (`fail|warn|ignore`)
- Added CLI option:
  - `--context-mismatch-mode` (`auto|fail|warn|ignore`)
- Added context mismatch router:
  - mismatch routed to `failures` / `warnings` / ignored by mode.

### 2) Result model extension

Regression report now includes:

- `warnings` array,
- `status` can be:
  - `failed`
  - `passed`
  - `passed_with_warnings`
- `thresholds.context_mismatch_mode`,
- `baseline_metadata.context_mismatch_mode`.

Exit code behavior:

- only `status=failed` returns non-zero,
- `passed_with_warnings` returns zero for non-blocking mode.

### 3) Config + summary

Updated `config/graph2d_seed_gate_regression.yaml`:

- `context_mismatch_mode: fail` (default behavior unchanged).

Updated `scripts/ci/summarize_graph2d_seed_gate_regression.py`:

- regression status now treats `passed_with_warnings` as pass,
- added warning count row,
- thresholds row includes `context_mode`,
- warning block rendered when `warnings` is non-empty.

## Tests

Updated `tests/unit/test_graph2d_seed_gate_regression_check.py`:

- policy resolution includes `context_mismatch_mode`,
- added `warn` mode test (`status=passed_with_warnings`),
- added `ignore` mode test (`status=passed` with context diff retained).

Updated `tests/unit/test_graph2d_seed_gate_regression_summary.py`:

- asserts warning count and context mode rendering,
- added warnings rendering test.

## Validation

```bash
pytest tests/unit/test_graph2d_seed_gate_regression_check.py \
       tests/unit/test_graph2d_seed_gate_regression_summary.py -q
```

Result: `19 passed`.

```bash
pytest tests/unit/test_graph2d_seed_gate_baseline_update.py \
       tests/unit/test_graph2d_seed_gate_summary.py \
       tests/unit/test_graph2d_seed_gate_trend.py \
       tests/unit/test_sweep_graph2d_profile_seeds.py -q
```

Result: `15 passed`.

```bash
make validate-graph2d-seed-gate-baseline-health
make validate-graph2d-seed-gate-regression
make validate-graph2d-seed-gate-strict-regression
```

Result:

- all checks passed,
- default pipeline stays in `context_mismatch_mode=fail`,
- no behavior regression on current baseline.
