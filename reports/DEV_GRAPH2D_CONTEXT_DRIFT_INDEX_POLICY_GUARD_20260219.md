# DEV_GRAPH2D_CONTEXT_DRIFT_INDEX_POLICY_GUARD_20260219

## Goal

Continue hardening Graph2D context-drift governance by adding a policy guard on index severity and integrating it into CI/local pipelines.

## Implementation

### 1) Index severity policy config

Added `config/graph2d_context_drift_index_policy.yaml`:

- section: `graph2d_context_drift_index_policy`
- fields:
  - `max_allowed_severity: alerted`
  - `fail_on_breach: false`

### 2) Severity policy checker

Added `scripts/ci/check_graph2d_context_drift_index_policy.py`:

- input:
  - index json
  - policy yaml (or CLI overrides)
- output:
  - policy report json (`status/pass|breached`, current vs allowed severity, reason, policy source)
  - markdown summary
- fail mode:
  - optional non-zero exit when `fail_on_breach=true` and policy breached.

### 3) CI integration

Updated `.github/workflows/ci.yml`:

- after index schema validation, added:
  - `Check Graph2D context drift index policy (3.11 only, non-blocking)`
  - step summary append for policy markdown
  - policy json/md/log included in index artifact upload

### 4) Local pipeline integration

Updated `Makefile` target `validate-graph2d-context-drift-pipeline`:

- added index policy check with local json/md outputs
- archive step now includes policy json/md artifacts

### 5) Additional quality gates included in this round

Also integrated and validated together:

- index schema validator:
  - `config/graph2d_context_drift_index_schema.json`
  - `scripts/ci/validate_graph2d_context_drift_index.py`
- severity annotation emitter:
  - `scripts/ci/emit_graph2d_context_drift_index_annotations.py`
- archive exporter:
  - `scripts/ci/archive_graph2d_context_drift_artifacts.py`

## Tests

Added:

- `tests/unit/test_graph2d_context_drift_index_policy.py`
- `tests/unit/test_graph2d_context_drift_index_validation.py`
- `tests/unit/test_graph2d_context_drift_archive.py`
- `tests/unit/test_graph2d_context_drift_index_annotations.py`

Updated:

- `tests/unit/test_graph2d_context_drift_artifact_index.py`
- `tests/unit/test_graph2d_context_drift_index_summary.py`

## Validation

```bash
pytest -q \
  tests/unit/test_graph2d_context_drift_index_policy.py \
  tests/unit/test_graph2d_context_drift_artifact_index.py \
  tests/unit/test_graph2d_context_drift_index_summary.py \
  tests/unit/test_graph2d_context_drift_index_validation.py \
  tests/unit/test_graph2d_context_drift_archive.py \
  tests/unit/test_graph2d_context_drift_index_annotations.py \
  tests/unit/test_graph2d_context_drift_alerts.py \
  tests/unit/test_graph2d_context_drift_history.py \
  tests/unit/test_graph2d_context_drift_key_counts.py \
  tests/unit/test_graph2d_context_drift_scripts_e2e.py \
  tests/unit/test_graph2d_context_drift_warning_emit.py
```

Result: `43 passed`.

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
GRAPH2D_CONTEXT_DRIFT_ARCHIVE_BUCKET=graph2d_context_drift_local_policy_test \
make validate-graph2d-context-drift-pipeline
```

Observed:

- index schema validation passed
- index policy check passed (`warn <= alerted`)
- archive generated at:
  - `reports/experiments/20260219/graph2d_context_drift_local_policy_test`
- archive includes policy artifacts:
  - `graph2d-context-drift-index-policy-local.json`
  - `graph2d-context-drift-index-policy-local.md`
