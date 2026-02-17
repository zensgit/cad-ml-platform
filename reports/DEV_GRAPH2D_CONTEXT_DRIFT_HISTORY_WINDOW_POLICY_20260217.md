# DEV_GRAPH2D_CONTEXT_DRIFT_HISTORY_WINDOW_POLICY_20260217

## Goal

Align context-drift history trend aggregation with alert policy so CI trend and alert windows use the same recent-run horizon by default.

## Implementation

### 1) Config-driven recent window for history rendering

Updated `scripts/ci/render_graph2d_context_drift_history.py`:

- added `--config` (default: `config/graph2d_context_drift_alerts.yaml`)
- added `--config-section` (default: `graph2d_context_drift_alerts`)
- added `--recent-runs` CLI override
- added `_resolve_recent_runs(...)` to apply precedence:
  - CLI override
  - config value
  - fallback default (`10`)
- changed recent aggregation window from hard-coded `10` to resolved policy window.

### 2) Ignore non-positive drift counts in recent aggregate

Updated recent key-total aggregation to skip `<= 0` counts, so markdown summary does not render zero-only noise rows.

### 3) CI wiring (explicit config)

Updated `.github/workflows/ci.yml` (tests job, Python 3.11):

- `Render Graph2D context drift history trend` now passes:
  - `--config config/graph2d_context_drift_alerts.yaml`

This makes CI policy source explicit and stable.

### 4) Tests

Updated `tests/unit/test_graph2d_context_drift_history.py`:

- added `test_build_history_markdown_respects_recent_window`
- added `test_resolve_recent_runs_prefers_cli_override`

## Validation

```bash
pytest -q \
  tests/unit/test_graph2d_context_drift_history.py \
  tests/unit/test_graph2d_context_drift_alerts.py \
  tests/unit/test_graph2d_context_drift_key_counts.py \
  tests/unit/test_graph2d_context_drift_warning_emit.py
```

Result: `16 passed`.

```bash
python3 - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path('.github/workflows/ci.yml').read_text(encoding='utf-8'))
print('ci.yml: ok')
PY
```

Result: `ci.yml: ok`.

Local smoke (`render_graph2d_context_drift_history.py`) showed `Recent window size: 5` when using default config, confirming policy alignment with alerts.
