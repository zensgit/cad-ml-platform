# DEV_GRAPH2D_SEED_GATE_BASELINE_FRESHNESS_GUARD_20260215

## Goal

Continue hardening Graph2D seed-gate regression checks by adding baseline freshness and snapshot existence guards, while keeping thresholds config-driven.

## Implementation

### 1) Regression checker: baseline metadata guard

Updated `scripts/ci/check_graph2d_seed_gate_regression.py`:

- added baseline policy defaults:
  - `max_baseline_age_days`
  - `require_snapshot_ref_exists`
- added metadata checks:
  - baseline date format + age check,
  - snapshot ref existence check.
- report now includes:
  - `baseline_metadata`:
    - `date`
    - `age_days`
    - `snapshot_ref`
    - `snapshot_path`
    - `snapshot_exists`
  - thresholds also include:
    - `max_baseline_age_days`
    - `require_snapshot_ref_exists`

### 2) Regression config extension

Updated `config/graph2d_seed_gate_regression.yaml`:

- `max_baseline_age_days: 365`
- `require_snapshot_ref_exists: true`

### 3) Regression summary extension

Updated `scripts/ci/summarize_graph2d_seed_gate_regression.py`:

- threshold row now shows baseline-age/snapshot policy,
- added `Baseline metadata` row.

### 4) Tests

Updated tests:

- `tests/unit/test_graph2d_seed_gate_regression_check.py`
  - baseline policy resolution precedence,
  - stale baseline failure case,
  - missing snapshot-ref failure case.
- `tests/unit/test_graph2d_seed_gate_regression_summary.py`
  - baseline metadata + policy summary assertions.

## Validation

### Unit tests

```bash
pytest tests/unit/test_graph2d_seed_gate_regression_check.py \
       tests/unit/test_graph2d_seed_gate_regression_summary.py \
       tests/unit/test_graph2d_seed_gate_baseline_update.py \
       tests/unit/test_graph2d_seed_gate_summary.py \
       tests/unit/test_graph2d_seed_gate_trend.py \
       tests/unit/test_sweep_graph2d_profile_seeds.py -q
```

Result: `23 passed`.

### Runtime checks

```bash
make validate-graph2d-seed-gate-regression
make validate-graph2d-seed-gate-strict-regression
```

Result:

- both channels passed,
- reports include:
  - thresholds with `max_baseline_age_days=365`, `require_snapshot_ref_exists=true`,
  - metadata with `date=2026-02-15`, `age_days=0`, `snapshot_exists=true`.

