# DEV_GRAPH2D_CONTEXT_DRIFT_RENDER_JSON_OUTPUT_20260217

## Goal

Add machine-readable outputs for Graph2D context-drift render scripts so CI artifacts can be consumed by downstream automation without parsing Markdown.

## Implementation

### 1) History render script supports JSON output

Updated `scripts/ci/render_graph2d_context_drift_history.py`:

- added `build_summary(...)` that returns:
  - `history_entries`
  - `recent_window`
  - `requested_recent_runs`
  - `recent_key_totals`
  - `rows`
  - `policy_source`
- added CLI argument:
  - `--output-json` (optional)
- retained existing Markdown output behavior.

### 2) Key-count render script supports JSON output

Updated `scripts/ci/render_graph2d_context_drift_key_counts.py`:

- added `build_summary(...)` that returns:
  - `report_count`
  - `rows`
  - `key_counts`
  - `policy_source`
- added CLI argument:
  - `--output-json` (optional)
- retained existing Markdown output behavior.

### 3) CI workflow now uploads render-summary JSON artifacts

Updated `.github/workflows/ci.yml`:

- key-count render step writes:
  - `/tmp/graph2d-context-drift-key-counts-ci-${{ matrix.python-version }}.json`
- history render step writes:
  - `/tmp/graph2d-context-drift-history-summary-ci-${{ matrix.python-version }}.json`
- artifact upload steps now include the new JSON files.

### 4) Tests

Updated/added tests:

- `tests/unit/test_graph2d_context_drift_history.py`
  - summary structure + policy source assertions
- `tests/unit/test_graph2d_context_drift_key_counts.py`
  - summary structure + policy source assertions
- `tests/unit/test_graph2d_context_drift_scripts_e2e.py`
  - history/key-count scripts write `--output-json`
  - JSON payload field assertions

## Validation

```bash
pytest -q \
  tests/unit/test_graph2d_context_drift_history.py \
  tests/unit/test_graph2d_context_drift_key_counts.py \
  tests/unit/test_graph2d_context_drift_alerts.py \
  tests/unit/test_graph2d_context_drift_warning_emit.py \
  tests/unit/test_graph2d_context_drift_scripts_e2e.py
```

Result: `27 passed`.

```bash
python3 - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path('.github/workflows/ci.yml').read_text(encoding='utf-8'))
print('ci.yml: ok')
PY
```

Result: `ci.yml: ok`.
