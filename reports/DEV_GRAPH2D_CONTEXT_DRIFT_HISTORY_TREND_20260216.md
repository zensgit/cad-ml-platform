# DEV_GRAPH2D_CONTEXT_DRIFT_HISTORY_TREND_20260216

## Goal

Provide cross-run context drift trend visibility (recent N runs) in CI summary.

## Implementation

### 1) History updater

Added `scripts/ci/update_graph2d_context_drift_history.py`:

- reads one or more regression report json files,
- extracts drift keys from `baseline_metadata.context_diff`,
- builds a run snapshot with:
  - run metadata (`run_id`, `run_number`, `ref_name`, `sha`),
  - aggregate status (`passed` / `passed_with_warnings` / `failed`),
  - warning/failure totals,
  - `drift_key_counts`,
  - per-report breakdown,
- appends to history json and trims to `--max-runs`.

### 2) History renderer

Added `scripts/ci/render_graph2d_context_drift_history.py`:

- renders markdown from history json:
  - per-run table (status/warn/fail/key counts),
  - recent-window aggregated drift key totals.

### 3) CI integration (with cache-backed persistence)

Updated `.github/workflows/ci.yml` (tests job, Python 3.11):

- restore cache for history json at job start,
- after warn probe and key-count summary:
  - update history json from current run reports,
  - render history markdown,
  - append history markdown to `GITHUB_STEP_SUMMARY`,
  - upload history json/md/log artifacts,
- save updated history json back to cache at job end.

## Tests

Added `tests/unit/test_graph2d_context_drift_history.py`:

- append+trim behavior,
- same-run-id replacement behavior,
- history markdown rendering,
- empty history handling.

## Validation

```bash
pytest tests/unit/test_graph2d_context_drift_history.py \
       tests/unit/test_graph2d_context_drift_key_counts.py \
       tests/unit/test_graph2d_seed_gate_regression_check.py \
       tests/unit/test_graph2d_seed_gate_regression_summary.py -q
```

Result: `26 passed`.

Local history smoke:

```bash
.venv/bin/python scripts/ci/update_graph2d_context_drift_history.py \
  --history-json /tmp/graph2d-context-drift-history-ci-local.json \
  --output-json /tmp/graph2d-context-drift-history-ci-local.json \
  --max-runs 20 \
  --run-id 1001 --run-number 501 --ref-name main --sha deadbeef \
  --report-json /tmp/graph2d-seed-gate-regression-ci-local.json \
  --report-json /tmp/graph2d-context-drift-warn-ci-local.json

.venv/bin/python scripts/ci/render_graph2d_context_drift_history.py \
  --history-json /tmp/graph2d-context-drift-history-ci-local.json \
  --title "Graph2D Context Drift History (Local)" \
  --output-md /tmp/graph2d-context-drift-history-ci-local.md
```

Observed:

- history entry generated with `status=passed_with_warnings`,
- markdown includes per-run row and recent key totals table.
