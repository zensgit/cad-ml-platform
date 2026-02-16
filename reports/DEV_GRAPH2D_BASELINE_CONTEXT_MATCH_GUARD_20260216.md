# DEV_GRAPH2D_BASELINE_CONTEXT_MATCH_GUARD_20260216

## Goal

Prevent false-positive regression passes caused by comparing metrics from different seed-sweep contexts (different profile/label mode/sample caps).

## Implementation

### 1) Baseline payload now stores channel context

Updated `scripts/ci/update_graph2d_seed_gate_baseline.py`:

- Added channel `context` block with:
  - `config`
  - `training_profile`
  - `manifest_label_mode`
  - `seeds`
  - `num_runs`
  - `max_samples`
  - `min_label_confidence`
  - `force_normalize_labels`
  - `force_clean_min_count`
  - `strict_low_conf_threshold`

### 2) Regression checker enforces context match

Updated `scripts/ci/check_graph2d_seed_gate_regression.py`:

- Added policy switches:
  - `require_context_match` (default true via config)
  - `context_keys` (default key set; configurable)
- Added CLI options:
  - `--require-context-match`
  - `--context-keys`
- Added compare logic between:
  - `baseline_channel.context`
  - current summary context
- Added report fields:
  - `thresholds.require_context_match`
  - `thresholds.context_keys`
  - `baseline_metadata.context_match`
  - `baseline_metadata.context_diff`
  - `baseline_context`
  - `current_context`
- Baseline health mode (`--use-baseline-as-current`) now also resolves current context from baseline channel context.

### 3) Regression config update

Updated `config/graph2d_seed_gate_regression.yaml`:

- `require_context_match: true`
- `context_keys` default list:
  - `training_profile`
  - `manifest_label_mode`
  - `max_samples`
  - `min_label_confidence`
  - `strict_low_conf_threshold`

### 4) Markdown regression summary update

Updated `scripts/ci/summarize_graph2d_seed_gate_regression.py`:

- Threshold row now renders `context_match` and `context_keys`.
- Baseline metadata row now renders `context_match`.

## Tests

Updated tests:

- `tests/unit/test_graph2d_seed_gate_baseline_update.py`
  - verifies baseline channel `context` fields.
- `tests/unit/test_graph2d_seed_gate_regression_check.py`
  - policy resolution covers `require_context_match/context_keys`,
  - context resolver for baseline-health mode,
  - passing case when context matches,
  - failing case when context mismatches.
- `tests/unit/test_graph2d_seed_gate_regression_summary.py`
  - summary rendering includes context guard fields.

Validation:

```bash
pytest tests/unit/test_graph2d_seed_gate_regression_check.py \
       tests/unit/test_graph2d_seed_gate_regression_summary.py \
       tests/unit/test_graph2d_seed_gate_baseline_update.py \
       tests/unit/test_graph2d_seed_gate_summary.py \
       tests/unit/test_graph2d_seed_gate_trend.py \
       tests/unit/test_sweep_graph2d_profile_seeds.py -q
```

Result: `31 passed`.

## Runtime verification

```bash
make update-graph2d-seed-gate-baseline
make validate-graph2d-seed-gate-baseline-health
make validate-graph2d-seed-gate-regression
make validate-graph2d-seed-gate-strict-regression
```

Result:

- all checks passed,
- both channels report `context_match=true`,
- baseline/snapshot integrity and date checks remain green.
