# DEV_GRAPH2D_SEED_GATE_INTEGRITY_HASH_GUARD_20260216

## Goal

Continue hardening Graph2D seed-gate regression by adding hash-based integrity validation for stable baseline and referenced snapshot.

## Implementation

### 1) Baseline generation adds integrity signatures

Updated `scripts/ci/update_graph2d_seed_gate_baseline.py`:

- baseline payload now includes:
  - `integrity.algorithm = "sha256-canonical-json"`
  - `integrity.standard_channel_sha256`
  - `integrity.strict_channel_sha256`
  - `integrity.payload_core_sha256`

### 2) Regression checker validates integrity

Updated `scripts/ci/check_graph2d_seed_gate_regression.py`:

- baseline policy gains:
  - `require_integrity_hash_match`
- when enabled:
  - validates baseline channel hash exists and matches recomputed hash,
  - validates baseline core hash exists and matches,
  - validates snapshot channel hash exists and matches recomputed hash,
  - validates snapshot channel hash equals baseline channel hash.
- report adds metadata:
  - `baseline_channel_hash_expected/actual/match`
  - `baseline_core_hash_expected/actual/match`
  - `snapshot_channel_hash_expected/actual/match`
  - `snapshot_vs_baseline_hash_match`

### 3) Config and summary updates

Updated `config/graph2d_seed_gate_regression.yaml`:

- `require_integrity_hash_match: true`

Updated `scripts/ci/summarize_graph2d_seed_gate_regression.py`:

- threshold row includes `integrity_match`,
- baseline metadata row includes hash-match statuses.

## Tests

Updated tests:

- `tests/unit/test_graph2d_seed_gate_baseline_update.py`
  - asserts integrity fields exist and are valid sha256 strings.
- `tests/unit/test_graph2d_seed_gate_regression_check.py`
  - added integrity mismatch failure case.
- `tests/unit/test_graph2d_seed_gate_regression_summary.py`
  - asserts integrity fields appear in summary.

Validation:

```bash
pytest tests/unit/test_graph2d_seed_gate_regression_check.py \
       tests/unit/test_graph2d_seed_gate_regression_summary.py \
       tests/unit/test_graph2d_seed_gate_baseline_update.py \
       tests/unit/test_graph2d_seed_gate_summary.py \
       tests/unit/test_graph2d_seed_gate_trend.py \
       tests/unit/test_sweep_graph2d_profile_seeds.py -q
```

Result: `25 passed`.

## Runtime verification

```bash
make update-graph2d-seed-gate-baseline
make validate-graph2d-seed-gate-regression
make validate-graph2d-seed-gate-strict-regression
```

Result:

- baseline refreshed to:
  - `config/graph2d_seed_gate_baseline.json`
  - `reports/experiments/20260216/graph2d_seed_gate_baseline_snapshot_20260216.json`
- standard/strict regression checks passed with:
  - `require_integrity_hash_match=true`,
  - `baseline_channel_hash_match=true`,
  - `snapshot_channel_hash_match=true`,
  - `snapshot_vs_baseline_hash_match=true`.

