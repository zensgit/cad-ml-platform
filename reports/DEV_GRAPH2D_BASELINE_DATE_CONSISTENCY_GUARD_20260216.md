# DEV_GRAPH2D_BASELINE_DATE_CONSISTENCY_GUARD_20260216

## Goal

Continue hardening Graph2D baseline governance by enforcing date consistency between:

- stable baseline `date`,
- snapshot payload `date`,
- snapshot-ref filename date stamp (`YYYYMMDD`).

## Implementation

### 1) Regression checker: date consistency policy

Updated `scripts/ci/check_graph2d_seed_gate_regression.py`:

- added policy keys:
  - `require_snapshot_date_match`
  - `require_snapshot_ref_date_match`
- added checks:
  - snapshot payload date equals baseline date,
  - snapshot_ref filename stamp (e.g. `..._20260216.json`) equals baseline date stamp.
- report metadata now includes:
  - `snapshot_payload_date`
  - `snapshot_date_match`
  - `snapshot_ref_date_stamp`
  - `expected_date_stamp`
  - `snapshot_ref_date_match`

### 2) Regression config update

Updated `config/graph2d_seed_gate_regression.yaml`:

- `require_snapshot_date_match: true`
- `require_snapshot_ref_date_match: true`

### 3) Summary update

Updated `scripts/ci/summarize_graph2d_seed_gate_regression.py`:

- threshold row now includes both date-match switches,
- baseline metadata row now shows both date-match results.

## Tests

Updated tests:

- `tests/unit/test_graph2d_seed_gate_regression_check.py`
  - baseline policy resolution includes new switches,
  - added failing case for snapshot payload date mismatch,
  - added failing case for snapshot_ref filename date mismatch.
- `tests/unit/test_graph2d_seed_gate_regression_summary.py`
  - asserts date-match fields are rendered.

Validation:

```bash
pytest tests/unit/test_graph2d_seed_gate_regression_check.py \
       tests/unit/test_graph2d_seed_gate_regression_summary.py \
       tests/unit/test_graph2d_seed_gate_baseline_update.py \
       tests/unit/test_graph2d_seed_gate_summary.py \
       tests/unit/test_graph2d_seed_gate_trend.py \
       tests/unit/test_sweep_graph2d_profile_seeds.py -q
```

Result: `28 passed`.

## Runtime verification

```bash
make validate-graph2d-seed-gate-baseline-health
make validate-graph2d-seed-gate-regression
make validate-graph2d-seed-gate-strict-regression
```

Result:

- all checks passed,
- baseline metadata reports:
  - `snapshot_date_match=true`
  - `snapshot_ref_date_match=true`.

