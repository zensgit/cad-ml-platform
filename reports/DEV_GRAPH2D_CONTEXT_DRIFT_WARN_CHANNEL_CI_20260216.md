# DEV_GRAPH2D_CONTEXT_DRIFT_WARN_CHANNEL_CI_20260216

## Goal

Wire the new `context_mismatch_mode=warn` into CI as a non-blocking drift observability channel.

## Implementation

### 1) Makefile target

Updated `Makefile`:

- Added target:
  - `validate-graph2d-seed-gate-context-drift-warn`
- Target behavior:
  - compares current **standard** seed summary against **strict** baseline channel,
  - forces relaxed metric thresholds (`1.0`) to avoid metric-failure noise,
  - runs with `--context-mismatch-mode warn`,
  - surfaces context drift as `status=passed_with_warnings`.

### 2) CI workflow integration

Updated `.github/workflows/ci.yml` (tests job, Python 3.11):

- Added non-blocking probe step:
  - `Run Graph2D context drift warn probe (3.11 only, non-blocking)`
  - `continue-on-error: true`
- Added artifact upload for probe log/report:
  - `/tmp/graph2d-context-drift-warn-ci.log`
  - `/tmp/graph2d-context-drift-warn-ci-<python>.json`
- Added step-summary append using:
  - `scripts/ci/summarize_graph2d_seed_gate_regression.py`

## Verification

Local execution:

```bash
make validate-graph2d-seed-gate-context-drift-warn
```

Observed result:

- `status=passed_with_warnings`
- warning includes:
  - `context: mismatch on keys [training_profile]`
- gate remains non-blocking (exit code 0).

## Notes

- This probe is intentionally observational and does not replace blocking standard/strict regression gates.
- Current blocking behavior remains in `context_mismatch_mode=fail` for normal channels.
