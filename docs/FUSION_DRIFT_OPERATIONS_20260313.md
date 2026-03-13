# Fusion + Drift Operations Runbook (2026-03-13)

## Scope

This runbook covers the Hybrid blind drift lifecycle:

1. archive history snapshots
2. suggest drift thresholds from history
3. run drift alert gate with slice-aware checks
4. tune workflow dispatch/env overrides

## Recommended Order

1. Bootstrap or accumulate history snapshots.
2. Run threshold suggestion from recent history.
3. Apply suggested values to CI vars (or workflow dispatch inputs).
4. Run drift alert gate and verify warning/noise level.

## Local Commands

### 1) Bootstrap history snapshots

```bash
make hybrid-blind-history-bootstrap
```

### 2) Suggest thresholds from history

```bash
make hybrid-blind-drift-suggest-thresholds
```

Artifacts:

- `reports/eval_history/hybrid_blind_drift_threshold_suggestion.json`
- `reports/eval_history/hybrid_blind_drift_threshold_suggestion.md`

### 2.1) Apply suggestion to GitHub variables

Preview:

```bash
make hybrid-blind-drift-apply-suggestion-gh
```

Execute:

```bash
make hybrid-blind-drift-apply-suggestion-gh \
  HYBRID_BLIND_DRIFT_SUGGEST_APPLY_EXECUTE=1 \
  HYBRID_BLIND_DRIFT_SUGGEST_APPLY_REPO=owner/repo
```

### 3) Run drift alerts

```bash
make hybrid-blind-drift-alert
```

Or one-shot:

```bash
make hybrid-blind-drift-activate
```

## Key Controls

### Auto-cap overlap thresholds

Drift checker supports:

- `--label-slice-auto-cap-min-common` / `--no-label-slice-auto-cap-min-common`
- `--family-slice-auto-cap-min-common` / `--no-family-slice-auto-cap-min-common`

Default is enabled. This prevents unreachable overlap requirements when slice cardinality is small.

### Make variables

- `HYBRID_BLIND_DRIFT_ALERT_LABEL_SLICE_AUTO_CAP_MIN_COMMON`
- `HYBRID_BLIND_DRIFT_ALERT_FAMILY_SLICE_AUTO_CAP_MIN_COMMON`
- `HYBRID_BLIND_DRIFT_SUGGEST_QUANTILE`
- `HYBRID_BLIND_DRIFT_SUGGEST_SAFETY_MULTIPLIER`
- `HYBRID_BLIND_DRIFT_SUGGEST_MIN_FLOOR_*`
- `HYBRID_BLIND_DRIFT_SUGGEST_FLOOR_LABEL_*`
- `HYBRID_BLIND_DRIFT_SUGGEST_FLOOR_FAMILY_*`

## Workflow Dispatch Inputs (evaluation-report)

New optional overrides:

- `hybrid_blind_drift_alert_label_slice_auto_cap_min_common`
- `hybrid_blind_drift_alert_family_slice_auto_cap_min_common`

Drift step now emits extra outputs:

- `label_slice_auto_cap_min_common`
- `label_slice_effective_min_common`
- `family_slice_auto_cap_min_common`
- `family_slice_effective_min_common`

These fields are included in job summary and PR comment.

## Strict-Real E2E Notes

`hybrid-blind-strict-real-e2e-gh` now supports explicit repo routing (`--repo`) and
pre-dispatch remote input checks. If remote workflow does not yet contain required
strict-real dispatch inputs, command fails fast with missing key list.

Typical command:

```bash
make hybrid-blind-strict-real-e2e-gh \
  HYBRID_BLIND_STRICT_E2E_REPO=owner/repo \
  HYBRID_BLIND_STRICT_E2E_REF=main
```

## Tuning Guidance

1. Start with quantile `0.90` and safety multiplier `1.20`.
2. Keep floor thresholds non-zero to resist overfitting to short windows.
3. If drift warnings are noisy due sparse slices, keep auto-cap enabled.
4. Disable auto-cap only when taxonomy is dense and stable.

## Rollback

1. Set both auto-cap switches to `false` in env/dispatch if strict fixed overlap is required.
2. Restore previous drift thresholds by reapplying prior CI vars.
3. Keep archive snapshots; only threshold values should be rolled back.
