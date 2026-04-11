# Hybrid Blind Family-Slice Drift Hardening (2026-03-13)

## Goal

Extend the existing hybrid blind drift system from:

- global metrics
- label-slice metrics

to an additional **family-slice** layer, so drift can be detected at a coarser and more stable semantic granularity.

## Implemented

### 1) Archive: family aggregation from label slices

Updated:
- `scripts/ci/archive_hybrid_blind_eval_history.py`

New behavior:
- Build `metrics.family_slices[]` by aggregating `label_slices`.
- Support optional label->family mapping JSON.
- Fallback grouping by normalized label prefix.

New args:
- `--family-prefix-len`
- `--family-map-json`
- `--family-slice-max-slices`

New persisted fields:
- `metrics.family_slices`
- `metrics.family_slice_meta`

### 2) Drift check: family-slice gates

Updated:
- `scripts/ci/check_hybrid_blind_drift_alerts.py`

New behavior:
- Parse and compare family slices between previous and latest snapshots.
- Compute worst family drops:
  - hybrid accuracy drop
  - hybrid gain vs Graph2D drop
- Optional family-level fail thresholds.
- Expose diagnostics in JSON + Markdown.

New args:
- `--family-slice-enable`
- `--family-slice-min-common`
- `--family-slice-min-support`
- `--family-slice-max-hybrid-accuracy-drop`
- `--family-slice-max-gain-drop`

### 3) CI + Make integration

Updated:
- `.github/workflows/evaluation-report.yml`
- `Makefile`

Added:
- workflow_dispatch inputs for family-slice drift controls
- env defaults for family-slice thresholds
- family-slice output fields in drift step
- family drift lines in job summary + PR comment
- local make target parameter pass-through

### 4) Weekly summary extension

Updated:
- `scripts/ci/generate_eval_weekly_summary.py`

Added rollup metrics:
- `hybrid_blind_family_slice_count_mean`
- `hybrid_blind_family_slice_count_latest`

### 5) Schema extension

Updated:
- `docs/eval_history.schema.json`

Added schema fields:
- `metrics.family_slices`
- `metrics.family_slice_meta`

### 6) Bootstrap hardening (direct execution + Make fallback)

Updated:
- `scripts/ci/bootstrap_hybrid_blind_eval_history.py`
- `Makefile` (`hybrid-blind-history-bootstrap`)

New behavior:
- `bootstrap_hybrid_blind_eval_history.py` now supports direct execution via:
  - `python scripts/ci/bootstrap_hybrid_blind_eval_history.py ...`
  - by auto-inserting repo root to `sys.path` when `scripts.ci` import fails.
- Make target now auto-falls back when default summary/gate path is absent:
  - summary fallback: `reports/evaluations/hybrid_blind_summary.json`
  - gate fallback: `reports/evaluations/hybrid_blind_gate_report.json`
- If summary still missing, target exits with explicit hint:
  - `make hybrid-blind-eval && make hybrid-blind-gate`

### 7) Slice-overlap anti-noise (auto-cap min_common)

Updated:
- `scripts/ci/check_hybrid_blind_drift_alerts.py`
- `Makefile` (`hybrid-blind-drift-alert` vars/flags)

New behavior:
- Added optional auto-cap for overlap threshold:
  - `--label-slice-auto-cap-min-common` / `--no-label-slice-auto-cap-min-common`
  - `--family-slice-auto-cap-min-common` / `--no-family-slice-auto-cap-min-common`
- Default is enabled (`true`), so required `min_common` is capped to reachable range
  when both latest/previous slice sets are smaller than configured threshold.
- Report now includes:
  - `metrics.effective_label_slice_min_common`
  - `metrics.effective_family_slice_min_common`
  - corresponding threshold toggles in `thresholds.*`

### 8) Threshold suggester + workflow auto-cap controls

Updated:
- `scripts/ci/suggest_hybrid_blind_drift_thresholds.py`
- `.github/workflows/evaluation-report.yml`
- `Makefile` (`hybrid-blind-drift-suggest-thresholds`)

New behavior:
- Added history-based threshold suggestion command:
  - reads hybrid_blind snapshots
  - computes consecutive drop distributions
  - outputs suggested thresholds (`json` + `md`)
- Added workflow_dispatch inputs for auto-cap overrides:
  - `hybrid_blind_drift_alert_label_slice_auto_cap_min_common`
  - `hybrid_blind_drift_alert_family_slice_auto_cap_min_common`
- Drift step now exports:
  - `label_slice_auto_cap_min_common`
  - `label_slice_effective_min_common`
  - `family_slice_auto_cap_min_common`
  - `family_slice_effective_min_common`
- Job summary + PR comment now include these drift diagnostics.

### 9) Suggestion-to-GitHub variables automation

Added:
- `scripts/ci/apply_hybrid_blind_drift_suggestion_to_gh_vars.py`
- `make hybrid-blind-drift-apply-suggestion-gh`

New behavior:
- reads suggestion JSON and maps values to `HYBRID_BLIND_DRIFT_ALERT_*` repo variables
- supports preview mode (default) and execute mode (`--apply`)
- used to batch-update drift thresholds without manual copy/paste

### 10) Strict-real GH dispatch hardening

Updated:
- `scripts/ci/dispatch_hybrid_blind_strict_real_workflow.py`
- `Makefile` (`hybrid-blind-strict-real-e2e-gh`)

New behavior:
- supports explicit `--repo` routing for dispatch/list/watch/view
- pre-dispatch remote workflow input check:
  - detects missing strict-real dispatch inputs in remote workflow file
  - fails fast with actionable missing-key list instead of waiting for 422

## Tests

Updated / added:
- `tests/unit/test_archive_hybrid_blind_eval_history.py`
- `tests/unit/test_check_hybrid_blind_drift_alerts.py`
- `tests/unit/test_generate_eval_weekly_summary.py`
- `tests/unit/test_hybrid_calibration_make_targets.py`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
- `tests/unit/test_bootstrap_hybrid_blind_eval_history.py`
  - added direct-script execution regression case
- `tests/unit/test_check_hybrid_blind_drift_alerts.py`
  - added auto-cap enabled/disabled behavior regression tests

## Validation

### Targeted regression

```bash
pytest -q tests/unit/test_archive_hybrid_blind_eval_history.py \
  tests/unit/test_check_hybrid_blind_drift_alerts.py \
  tests/unit/test_generate_eval_weekly_summary.py \
  tests/unit/test_hybrid_calibration_make_targets.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  tests/unit/test_validate_eval_history_hybrid_blind.py \
  tests/unit/test_validate_eval_history_history_sequence.py
```

Result:
- `39 passed`

### Workflow integration regression

```bash
make validate-hybrid-blind-workflow
```

Result:
- `55 passed`

### Runtime check (local, no-arg make fallback)

```bash
make hybrid-blind-history-bootstrap
```

Result:
- command succeeded
- fallback summary/gate paths were auto-selected
- 3 snapshots written under `reports/eval_history/*_hybrid_blind.json`

### Drift activation check (local end-to-end)

```bash
make hybrid-blind-drift-activate \
  HYBRID_BLIND_DRIFT_ALERT_MIN_REPORTS=3 \
  HYBRID_BLIND_DRIFT_ALERT_LABEL_SLICE_ENABLE=1 \
  HYBRID_BLIND_DRIFT_ALERT_FAMILY_SLICE_ENABLE=1
```

Result:
- command succeeded
- drift status: `passed`
- family slice overlap and worst-drop metrics produced
- no overlap warning under small-slice scenario due effective auto-cap:
  - `label_slice_min_common=3`, `effective_label_slice_min_common=2`

### Threshold suggestion check (local)

```bash
make hybrid-blind-drift-suggest-thresholds
```

Result:
- command succeeded (`status=ok`)
- suggestion artifacts generated:
  - `reports/eval_history/hybrid_blind_drift_threshold_suggestion.json`
  - `reports/eval_history/hybrid_blind_drift_threshold_suggestion.md`

## Next

1. In production history, either increase label overlap or tune `HYBRID_BLIND_DRIFT_ALERT_LABEL_SLICE_MIN_COMMON` to reduce non-actionable warnings.
2. Provide/maintain `config/hybrid_blind_family_map.json` with domain taxonomy to improve family-slice stability.
3. Run strict-real GH dispatch end-to-end on real DXF data and archive that run for baseline.
