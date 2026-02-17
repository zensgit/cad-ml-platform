# DEV_GRAPH2D_CONTEXT_DRIFT_POLICY_SOURCE_E2E_20260217

## Goal

Complete three follow-up hardening tasks for Graph2D context-drift CI scripts:

1. make history `max_runs` config-driven,
2. unify `policy_source` output across context-drift scripts,
3. add script-level E2E tests for missing/corrupted config fallback behavior.

## Implementation

### 1) Config-driven `max_runs` for history updater

Updated `scripts/ci/update_graph2d_context_drift_history.py`:

- added config loading args:
  - `--config` (default `config/graph2d_context_drift_alerts.yaml`)
  - `--config-section` (default `graph2d_context_drift_alerts`)
- changed `--max-runs` default from fixed value to optional override (`None`),
- added `_resolve_max_runs(...)` precedence:
  - CLI override
  - config value
  - built-in default (`20`)
- each history snapshot now includes `policy_source`.

Updated `config/graph2d_context_drift_alerts.yaml`:

- added `max_runs: 20` under `graph2d_context_drift_alerts`.

### 2) Unified `policy_source` output

Aligned output schema across scripts:

- `scripts/ci/check_graph2d_context_drift_alerts.py` (already present),
- `scripts/ci/update_graph2d_context_drift_history.py` (new),
- `scripts/ci/render_graph2d_context_drift_history.py` (new summary block),
- `scripts/ci/render_graph2d_context_drift_key_counts.py` (new summary block).

Unified shape:

- `config`
- `config_section`
- `config_loaded`
- `resolved_policy`
- `cli_overrides`

### 3) CI wiring

Updated `.github/workflows/ci.yml`:

- `render_graph2d_context_drift_key_counts.py` now passes `--config`.
- `update_graph2d_context_drift_history.py` now passes `--config`.
- removed hardcoded `--max-runs 20` from CI history-update step (now policy-driven).

### 4) Script-level E2E tests (missing/broken config)

Added `tests/unit/test_graph2d_context_drift_scripts_e2e.py`:

- invalid config for update script -> fallback to default `max_runs=20`,
- missing config for history render script -> fallback to `resolved_recent_runs=10`,
- invalid config for key-count render script -> fallback to `resolved_recent_runs=5`.

Also updated existing unit tests:

- `tests/unit/test_graph2d_context_drift_history.py`
- `tests/unit/test_graph2d_context_drift_key_counts.py`

## Validation

```bash
pytest -q \
  tests/unit/test_graph2d_context_drift_history.py \
  tests/unit/test_graph2d_context_drift_key_counts.py \
  tests/unit/test_graph2d_context_drift_alerts.py \
  tests/unit/test_graph2d_context_drift_warning_emit.py \
  tests/unit/test_graph2d_context_drift_scripts_e2e.py
```

Result: `23 passed`.

```bash
python3 - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path('.github/workflows/ci.yml').read_text(encoding='utf-8'))
print('ci.yml: ok')
PY
```

Result: `ci.yml: ok`.
