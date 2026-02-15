# DEV_GRAPH2D_SEED_GATE_SNAPSHOT_MATCH_GUARD_20260215

## Goal

Continue strengthening Graph2D seed-gate regression safety by ensuring the stable baseline and its referenced dated snapshot stay in sync.

## Implementation

### 1) Regression checker enhancement

Updated `scripts/ci/check_graph2d_seed_gate_regression.py`:

- added policy key:
  - `require_snapshot_metrics_match`
- when enabled:
  - reads `source.snapshot_ref` payload,
  - checks current channel (`standard|strict`) exists in snapshot,
  - compares key regression metrics with stable baseline channel:
    - `strict_accuracy_mean`
    - `strict_accuracy_min`
    - `strict_top_pred_ratio_max`
    - `strict_low_conf_ratio_max`
    - `manifest_distinct_labels_min`
- fails regression check when mismatch exists.

Report extensions:

- `thresholds.require_snapshot_metrics_match`
- `baseline_metadata`:
  - `snapshot_channel_present`
  - `snapshot_metrics_match`
  - `snapshot_metrics_diff`

### 2) Config update

Updated `config/graph2d_seed_gate_regression.yaml`:

- `require_snapshot_metrics_match: true`

### 3) Summary rendering update

Updated `scripts/ci/summarize_graph2d_seed_gate_regression.py`:

- threshold row includes `snapshot_match=...`,
- baseline metadata row includes `snapshot_metrics_match=...`.

## Tests

Updated tests:

- `tests/unit/test_graph2d_seed_gate_regression_check.py`
  - baseline policy resolution now checks `require_snapshot_metrics_match`,
  - added failure case for snapshot metrics mismatch.
- `tests/unit/test_graph2d_seed_gate_regression_summary.py`
  - summary assertions include snapshot match fields.

Validation:

```bash
pytest tests/unit/test_graph2d_seed_gate_regression_check.py \
       tests/unit/test_graph2d_seed_gate_regression_summary.py \
       tests/unit/test_graph2d_seed_gate_baseline_update.py \
       tests/unit/test_graph2d_seed_gate_summary.py \
       tests/unit/test_graph2d_seed_gate_trend.py \
       tests/unit/test_sweep_graph2d_profile_seeds.py -q
```

Result: `24 passed`.

## Runtime verification

```bash
make validate-graph2d-seed-gate-regression
make validate-graph2d-seed-gate-strict-regression
```

Result:

- both channels passed,
- reports include:
  - `require_snapshot_metrics_match=true`,
  - `snapshot_metrics_match=true`,
  - empty `snapshot_metrics_diff`.

