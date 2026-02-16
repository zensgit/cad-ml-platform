# DEV_GRAPH2D_CONTEXT_DRIFT_ALERT_THRESHOLDS_20260216

## Goal

Add threshold-based drift alerts on top of context-drift history trend, with CI warning annotations and non-blocking behavior.

## Implementation

### 1) New alert checker script

Added `scripts/ci/check_graph2d_context_drift_alerts.py`:

- input:
  - history json,
  - recent window size (`--recent-runs`),
  - default key threshold (`--default-key-threshold`),
  - optional per-key overrides (`--key-threshold key=count`),
- output:
  - report json (`status`, `alerts`, key totals, thresholds),
  - markdown summary,
- default behavior is non-blocking (`exit 0`), optional `--fail-on-alert` supported.

### 2) CI integration

Updated `.github/workflows/ci.yml` (tests job, Python 3.11):

- added non-blocking alert-check step:
  - `Check Graph2D context drift alerts (3.11 only, non-blocking)`
- current thresholds:
  - recent window: `5`
  - default key threshold: `3`
  - override: `max_samples=2`
- appended alert markdown to Step Summary,
- emitted GitHub warning annotations via:
  - `scripts/ci/emit_graph2d_context_drift_warnings.py`
  - output format: `::warning title=Graph2D Context Drift::...`
- uploaded alert artifacts (`json/md/log`).

### 3) Tests

Added `tests/unit/test_graph2d_context_drift_alerts.py`:

- default-threshold alert triggering,
- per-key threshold override behavior,
- below-threshold clear behavior,
- markdown rendering with alert lines.

Added `tests/unit/test_graph2d_context_drift_warning_emit.py`:

- warning-line generation from alert payload,
- empty-alert handling.

## Validation

```bash
pytest tests/unit/test_graph2d_context_drift_alerts.py \
       tests/unit/test_graph2d_context_drift_history.py \
       tests/unit/test_graph2d_context_drift_key_counts.py \
       tests/unit/test_graph2d_context_drift_warning_emit.py -q
```

Result: `13 passed`.

Local alert smoke:

```bash
.venv/bin/python scripts/ci/check_graph2d_context_drift_alerts.py \
  --history-json /tmp/graph2d-context-drift-history-ci-local.json \
  --recent-runs 5 \
  --default-key-threshold 3 \
  --key-threshold max_samples=2 \
  --title "Graph2D Context Drift Alerts (Local)" \
  --output-json /tmp/graph2d-context-drift-alerts-ci-local.json \
  --output-md /tmp/graph2d-context-drift-alerts-ci-local.md
```

Observed:

- status `alerted`,
- alert message for `max_samples`,
- markdown includes threshold table and alert block.

Local warning emission smoke:

```bash
.venv/bin/python scripts/ci/emit_graph2d_context_drift_warnings.py \
  --report-json /tmp/graph2d-context-drift-alerts-ci-local.json
```

Observed:

- emitted one `::warning` annotation line,
- printed `warning_annotations=1`.
