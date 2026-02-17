# DEV_GRAPH2D_CONTEXT_DRIFT_PIPELINE_INDEX_20260217

## Goal

Deliver the requested `1+2+3` follow-up:

1. add structured alert `summary` output aligned with render summaries,
2. add a one-command local pipeline target (update + render + alerts),
3. produce a single index JSON that aggregates all context-drift artifacts.

## Implementation

### 1) Structured alert summary alignment

Updated `scripts/ci/check_graph2d_context_drift_alerts.py`:

- added `build_summary(report)` output schema:
  - `status`
  - `history_entries`
  - `recent_window`
  - `alert_count`
  - `key_totals`
  - `rows`
  - `policy_source`
- report JSON now includes top-level `summary` field while preserving existing keys.

Updated tests:

- `tests/unit/test_graph2d_context_drift_alerts.py`:
  - added structured summary assertions.
- `tests/unit/test_graph2d_context_drift_scripts_e2e.py`:
  - added script-level assertion for emitted `summary`.

### 2) One-command local drift pipeline

Updated `Makefile` with:

- `validate-graph2d-context-drift-pipeline`

Pipeline steps:

- update history (`update_graph2d_context_drift_history.py`)
- render key counts (`render_graph2d_context_drift_key_counts.py`)
- render history trend (`render_graph2d_context_drift_history.py`)
- check drift alerts (`check_graph2d_context_drift_alerts.py`)
- build artifact index (`index_graph2d_context_drift_artifacts.py`)

All outputs are env-configurable with sensible `/tmp/...-local.*` defaults.

### 3) Single artifact index for automation

Added `scripts/ci/index_graph2d_context_drift_artifacts.py`:

- merges:
  - alerts report,
  - history summary,
  - key-count summary,
  - optional raw history
- emits:
  - `overview` (`status`, `alert_count`, `history_entries`, `top_drift_key`, etc.)
  - `artifacts` (path + exists flags)
  - `policy_sources`
  - `summaries`

Added tests:

- `tests/unit/test_graph2d_context_drift_artifact_index.py`
  - build function behavior,
  - missing-input fallback,
  - realistic payload integration.

CI integration (`.github/workflows/ci.yml`):

- added step:
  - `Build Graph2D context drift artifact index (3.11 only)`
- added upload:
  - `graph2d-context-drift-index-ci-${{ matrix.python-version }}`

## Validation

```bash
pytest -q \
  tests/unit/test_graph2d_context_drift_alerts.py \
  tests/unit/test_graph2d_context_drift_history.py \
  tests/unit/test_graph2d_context_drift_key_counts.py \
  tests/unit/test_graph2d_context_drift_scripts_e2e.py \
  tests/unit/test_graph2d_context_drift_artifact_index.py \
  tests/unit/test_graph2d_context_drift_warning_emit.py
```

Result: `32 passed`.

```bash
python3 - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path('.github/workflows/ci.yml').read_text(encoding='utf-8'))
print('ci.yml: ok')
PY
```

Result: `ci.yml: ok`.

```bash
make validate-graph2d-context-drift-pipeline
```

Result: local pipeline completed end-to-end and produced:

- `/tmp/graph2d-context-drift-history-local.json`
- `/tmp/graph2d-context-drift-key-counts-local.json`
- `/tmp/graph2d-context-drift-history-summary-local.json`
- `/tmp/graph2d-context-drift-alerts-local.json`
- `/tmp/graph2d-context-drift-index-local.json`
