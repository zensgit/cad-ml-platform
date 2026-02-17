# DEV_GRAPH2D_CONTEXT_DRIFT_SEVERITY_SCHEMA_ARCHIVE_20260217

## Goal

Implement the requested `1+2+3` enhancements:

1. add index severity model (`clear|warn|alerted|failed`) with CI-visible colored signals,
2. add JSON schema validation for index payload and enforce it in CI,
3. export context-drift artifacts to `reports/experiments/<date>/...` archive path.

## Implementation

### 1) Severity model + colored CI display

Updated `scripts/ci/index_graph2d_context_drift_artifacts.py`:

- added `overview.severity` and `overview.severity_reason`,
- added `overview.artifact_coverage` (`present/total`),
- severity decision:
  - `failed` when required artifacts missing or status failed,
  - `alerted` when threshold alerts present,
  - `warn` when drift exists but below alert threshold,
  - `clear` when no drift signal,
- added top-level `schema_version: "1.0.0"`.

Updated `scripts/ci/summarize_graph2d_context_drift_index.py`:

- added severity banner + icon mapping:
  - `🟢 clear`
  - `🟡 warn`
  - `🟠 alerted`
  - `🔴 failed`

Added `scripts/ci/emit_graph2d_context_drift_index_annotations.py`:

- emits GitHub annotations based on severity:
  - `::notice` for clear
  - `::warning` for warn/alerted
  - `::error` for failed

### 2) JSON schema validation

Added schema:

- `config/graph2d_context_drift_index_schema.json`

Added validator:

- `scripts/ci/validate_graph2d_context_drift_index.py`
  - validates index JSON via `jsonschema` Draft 2020-12
  - prints machine-readable pass/fail lines
  - non-zero exit on validation errors

CI wiring (`.github/workflows/ci.yml`):

- validates index JSON against schema,
- stores validation log in artifact bundle.

### 3) Archive export to reports/experiments/<date>

Added:

- `scripts/ci/archive_graph2d_context_drift_artifacts.py`
  - archives selected artifacts under:
    - `reports/experiments/<date>/<bucket>/`
  - writes `archive_manifest.json`
  - supports strict mode (`--require-exists`)

CI wiring:

- archives context-drift artifacts to `reports/experiments/${ARCHIVE_DATE}/${ARCHIVE_BUCKET}`,
- uploads archive directory + manifest as CI artifact.

Local wiring (`Makefile`):

- `validate-graph2d-context-drift-pipeline` now includes:
  - index schema validation
  - archive export

## Tests

Added/updated:

- `tests/unit/test_graph2d_context_drift_artifact_index.py`
- `tests/unit/test_graph2d_context_drift_index_summary.py`
- `tests/unit/test_graph2d_context_drift_index_validation.py`
- `tests/unit/test_graph2d_context_drift_archive.py`
- `tests/unit/test_graph2d_context_drift_index_annotations.py`
- plus existing context-drift suites for regression coverage.

## Validation

```bash
pytest -q \
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

Result: `40 passed`.

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
GRAPH2D_CONTEXT_DRIFT_ARCHIVE_BUCKET=graph2d_context_drift_local_test \
make validate-graph2d-context-drift-pipeline
```

Observed:

- index validation passed (`index_schema_valid=true`),
- archive directory created at:
  - `reports/experiments/20260217/graph2d_context_drift_local_test`
- `archive_manifest.json` generated with `copied_count=9`, `missing_count=0`.
