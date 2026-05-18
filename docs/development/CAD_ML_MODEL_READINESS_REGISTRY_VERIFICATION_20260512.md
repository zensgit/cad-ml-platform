# CAD ML Model Readiness Registry Verification

Date: 2026-05-12

## Scope

Verified the model readiness registry slice that replaces the static loader
readiness stub and wires registry evidence into `/ready`, `/health`, and the new
model-readiness health endpoint.

## Commands Run

### Syntax

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile \
  src/models/readiness_registry.py \
  src/models/loader.py \
  src/api/health_utils.py \
  src/api/v1/health.py \
  src/main.py \
  tests/unit/test_model_readiness_registry.py
```

Result: passed.

### Lint

```bash
.venv311/bin/flake8 \
  src/models/readiness_registry.py \
  src/models/loader.py \
  src/api/health_utils.py \
  src/api/v1/health.py \
  src/main.py \
  tests/unit/test_model_readiness_registry.py \
  --max-line-length=100
```

Result: passed.

### Targeted Tests

```bash
.venv311/bin/python -m pytest -q \
  tests/unit/test_model_readiness_registry.py \
  tests/unit/test_main_coverage.py::TestReadinessCheck \
  tests/unit/test_health_utils_coverage.py::TestBuildHealthPayload \
  tests/unit/test_model_rollback_health.py
```

Result: `35 passed, 7 warnings`.

### OpenAPI Contract

The new endpoint intentionally changes the OpenAPI schema snapshot. The snapshot
was regenerated with:

```bash
.venv311/bin/python scripts/ci/generate_openapi_schema_snapshot.py \
  --output config/openapi_schema_snapshot.json
```

The generator reported:

```text
paths=193 operations=198
```

Then contract and health checks were rerun:

```bash
.venv311/bin/python -m pytest -q \
  tests/contract/test_openapi_operation_ids.py \
  tests/contract/test_openapi_schema_snapshot.py \
  tests/test_health_and_metrics.py \
  tests/unit/test_health_extended_endpoint.py
```

Result: `7 passed, 7 warnings`.

The warnings are existing `ezdxf` / `pyparsing` deprecation warnings:

- `addParseAction` deprecated
- `oneOf` deprecated
- `setResultsName` deprecated
- `infixNotation` deprecated

They are dependency warnings, not failures from this readiness slice.

## Verified Behavior

- `src/models/loader.py` no longer assumes `_loaded = True`.
- Missing local checkpoints are represented as `fallback`, not `loaded`.
- Local development remains runnable when models are missing and fallbacks exist.
- Required-model gates can make readiness fail through
  `MODEL_READINESS_REQUIRED_MODELS`.
- Strict mode can turn degraded model state into readiness failure through
  `MODEL_READINESS_STRICT=true`.
- `/ready` still uses the existing `models_loaded` check name for response
  compatibility, but the implementation now comes from registry evidence.
- `/health` now exposes the full registry at
  `config.ml.readiness.model_registry`.
- `/api/v1/health/model-readiness` returns model-by-model readiness state.
- `config/openapi_schema_snapshot.json` includes the new model-readiness endpoint.
- Existing model rollback health behavior remains covered.

## Remaining Verification Gaps

- Load-failure fixtures for incompatible Graph2D, UV-Net, and PointNet checkpoints
  are still pending because this slice avoids eager heavyweight model loading.
- End-to-end production verification should be repeated in an environment with
  real checkpoint files so checksum, `available`, and `loaded` states can be
  observed together.
