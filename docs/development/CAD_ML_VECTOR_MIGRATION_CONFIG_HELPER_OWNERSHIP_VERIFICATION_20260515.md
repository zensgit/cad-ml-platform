# CAD ML Vector Migration Config Helper Ownership Verification 20260515

## Scope

Verification for moving migration config helpers from `src/api/v1/vectors.py`
into `src/core/vector_migration_config.py` while preserving these facade
exports:

- `src.api.v1.vectors._resolve_vector_migration_scan_limit`
- `src.api.v1.vectors._resolve_vector_migration_target_version`
- `src.api.v1.vectors._coerce_int`

## Results

| Check | Result |
| --- | --- |
| Python compile | PASS |
| flake8 target files | PASS |
| pytest target slice | PASS |
| Claude Code read-only compatibility review | PASS |
| git diff whitespace check | PASS |
| trailing whitespace scan | PASS |

## Commands

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache .venv311/bin/python -m py_compile \
  src/api/v1/vectors.py \
  src/core/vector_migration_config.py \
  tests/unit/test_vector_migration_config.py \
  tests/unit/test_vectors_migration_config_delegation.py \
  tests/unit/test_vector_migrate_layouts.py \
  tests/unit/test_vectors_migration_pending_candidates_delegation.py
```

Result: PASS.

```bash
.venv311/bin/flake8 \
  src/api/v1/vectors.py \
  src/core/vector_migration_config.py \
  tests/unit/test_vector_migration_config.py \
  tests/unit/test_vectors_migration_config_delegation.py \
  tests/unit/test_vector_migrate_layouts.py \
  tests/unit/test_vectors_migration_pending_candidates_delegation.py
```

Result: PASS.

```bash
.venv311/bin/pytest -q \
  tests/unit/test_vector_migration_config.py \
  tests/unit/test_vectors_migration_config_delegation.py \
  tests/unit/test_vector_migrate_layouts.py \
  tests/unit/test_vectors_migration_pending_candidates_delegation.py \
  tests/unit/test_vector_migration_status.py \
  tests/unit/test_vectors_migration_reporting_delegation.py \
  tests/unit/test_vectors_migration_plan_delegation.py \
  tests/unit/test_vectors_migration_trends_delegation.py \
  tests/contract/test_openapi_operation_ids.py \
  tests/contract/test_openapi_schema_snapshot.py
```

Result: PASS, `24 passed, 7 warnings in 2.67s`.

Warnings are from `ezdxf/queryparser.py` PyParsing deprecations and are not
introduced by this slice.

```bash
git diff --check
```

Result: PASS.

```bash
rg -n "[[:blank:]]$" \
  src/api/v1/vectors.py \
  src/core/vector_migration_config.py \
  tests/unit/test_vector_migration_config.py \
  tests/unit/test_vectors_migration_config_delegation.py \
  tests/unit/test_vector_migrate_layouts.py \
  tests/unit/test_vectors_migration_pending_candidates_delegation.py \
  docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md \
  docs/development/CAD_ML_VECTOR_MIGRATION_CONFIG_HELPER_OWNERSHIP_DEVELOPMENT_20260515.md \
  docs/development/CAD_ML_VECTOR_MIGRATION_CONFIG_HELPER_OWNERSHIP_VERIFICATION_20260515.md
```

Result: PASS, no matches.

## Claude Code Review

Claude Code was invoked with `--tools ""` and received only targeted snippets
for `src/core/vector_migration_config.py`, `src/api/v1/vectors.py`, and relevant
tests.

Conclusion: the helper move is compatible for facade aliases and call-time env
resolution. Claude Code flagged split-router patch scope as the one important
risk. Follow-up verification confirmed split routers still use
`src.api.v1.vectors`, and `tests/unit/test_vectors_migration_config_delegation.py`
now pins scan-limit and target-version facade patch behavior at the HTTP route
level.

## Residual Risk

Future router changes could bypass the facade by importing migration config
helpers directly from core. Keep the delegation tests active while
`src.api.v1.vectors` remains the compatibility facade.
