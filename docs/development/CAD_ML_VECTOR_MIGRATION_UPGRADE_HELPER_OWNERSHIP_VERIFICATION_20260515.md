# CAD ML Vector Migration Upgrade Helper Ownership Verification 20260515

## Scope

Verification for moving `_prepare_vector_for_upgrade` from
`src/api/v1/vectors.py` into `src/core/vector_migration_upgrade.py` while
preserving `src.api.v1.vectors._prepare_vector_for_upgrade`.

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
  src/core/vector_migration_upgrade.py \
  tests/unit/test_vector_migration_upgrade.py \
  tests/unit/test_vector_migrate_layouts.py \
  tests/unit/test_vectors_migration_preview_delegation.py \
  tests/unit/test_vectors_migrate_delegation.py
```

Result: PASS.

```bash
.venv311/bin/flake8 \
  src/api/v1/vectors.py \
  src/core/vector_migration_upgrade.py \
  tests/unit/test_vector_migration_upgrade.py \
  tests/unit/test_vector_migrate_layouts.py \
  tests/unit/test_vectors_migration_preview_delegation.py \
  tests/unit/test_vectors_migrate_delegation.py
```

Result: PASS.

```bash
.venv311/bin/pytest -q \
  tests/unit/test_vector_migration_upgrade.py \
  tests/unit/test_vector_migrate_layouts.py \
  tests/unit/test_vectors_migration_preview_delegation.py \
  tests/unit/test_vectors_migrate_delegation.py \
  tests/unit/test_vector_migration_preview_pipeline.py \
  tests/unit/test_vector_migrate_pipeline.py \
  tests/contract/test_openapi_operation_ids.py \
  tests/contract/test_openapi_schema_snapshot.py
```

Result: PASS, `18 passed, 7 warnings in 3.02s`.

Warnings are from `ezdxf/queryparser.py` PyParsing deprecations and are not
introduced by this slice.

```bash
git diff --check
```

Result: PASS.

```bash
rg -n "[[:blank:]]$" \
  src/api/v1/vectors.py \
  src/core/vector_migration_upgrade.py \
  tests/unit/test_vector_migration_upgrade.py \
  tests/unit/test_vector_migrate_layouts.py \
  tests/unit/test_vectors_migration_preview_delegation.py \
  tests/unit/test_vectors_migrate_delegation.py \
  docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md \
  docs/development/CAD_ML_VECTOR_MIGRATION_UPGRADE_HELPER_OWNERSHIP_DEVELOPMENT_20260515.md \
  docs/development/CAD_ML_VECTOR_MIGRATION_UPGRADE_HELPER_OWNERSHIP_VERIFICATION_20260515.md
```

Result: PASS, no matches.

## Claude Code Review

Claude Code was invoked with `--tools ""` and received only targeted snippets
for the core helper, facade import, split routes, and relevant tests.

Conclusion: compatibility is intact. The helper remains a module attribute on
`src.api.v1.vectors`, and both preview and migrate routes read that attribute at
request time before passing it into the shared pipelines.

## Residual Risk

Future routers could bypass the facade by importing
`src.core.vector_migration_upgrade.prepare_vector_for_upgrade` directly. Keep
the preview and migrate delegation tests active until the facade patch surface
is intentionally retired.
