# DEV_GRAPH2D_CONTEXT_DRIFT_INDEX_SUMMARY_20260217

## Goal

Continue the Graph2D context-drift pipeline hardening by making the new index artifact human-readable in CI and local runs.

## Implementation

### 1) Index markdown summarizer script

Added `scripts/ci/summarize_graph2d_context_drift_index.py`:

- input: `--index-json`
- output: markdown table summarizing:
  - status / alert_count
  - history_entries / recent_window
  - drift_key_count / top_drift_key
  - artifact coverage and per-artifact existence
  - alert rows (when present)

### 2) CI integration

Updated `.github/workflows/ci.yml`:

- append index summary to `GITHUB_STEP_SUMMARY`
- render index markdown file:
  - `/tmp/graph2d-context-drift-index-ci-${{ matrix.python-version }}.md`
- upload index artifact now includes both:
  - json
  - md

### 3) Local pipeline integration

Updated `Makefile` target `validate-graph2d-context-drift-pipeline`:

- after index json generation, also renders:
  - `${GRAPH2D_CONTEXT_DRIFT_INDEX_MD:-/tmp/graph2d-context-drift-index-local.md}`

### 4) Tests

Added `tests/unit/test_graph2d_context_drift_index_summary.py`:

- verifies overview + artifact rows rendering
- verifies empty-payload fallback rendering

## Validation

```bash
pytest -q \
  tests/unit/test_graph2d_context_drift_index_summary.py \
  tests/unit/test_graph2d_context_drift_artifact_index.py \
  tests/unit/test_graph2d_context_drift_alerts.py \
  tests/unit/test_graph2d_context_drift_history.py \
  tests/unit/test_graph2d_context_drift_key_counts.py \
  tests/unit/test_graph2d_context_drift_scripts_e2e.py \
  tests/unit/test_graph2d_context_drift_warning_emit.py
```

Result: `34 passed`.

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

Result: pipeline passed and generated local markdown index summary:

- `/tmp/graph2d-context-drift-index-local.md`
