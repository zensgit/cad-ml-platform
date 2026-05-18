# CAD ML Vector Migration Qdrant Preview Helper Ownership Verification 20260515

## Scope

Verification for moving Qdrant migration preview sample collection from
`src/api/v1/vectors.py` into `src/core/vector_migration_qdrant_preview.py` while
preserving `src.api.v1.vectors._collect_qdrant_preview_samples`.

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
  src/core/vector_migration_qdrant_preview.py \
  tests/unit/test_vector_migration_qdrant_preview.py \
  tests/unit/test_vectors_migration_preview_delegation.py
```

Result: PASS.

```bash
.venv311/bin/flake8 \
  src/api/v1/vectors.py \
  src/core/vector_migration_qdrant_preview.py \
  tests/unit/test_vector_migration_qdrant_preview.py \
  tests/unit/test_vectors_migration_preview_delegation.py
```

Result: PASS.

```bash
.venv311/bin/pytest -q \
  tests/unit/test_vector_migration_qdrant_preview.py \
  tests/unit/test_vectors_migration_preview_delegation.py \
  tests/unit/test_vector_migration_preview_pipeline.py \
  tests/unit/test_vector_migrate_layouts.py \
  tests/contract/test_openapi_operation_ids.py \
  tests/contract/test_openapi_schema_snapshot.py
```

Result: PASS, `14 passed, 7 warnings in 2.99s`.

Warnings are from `ezdxf/queryparser.py` PyParsing deprecations and are not
introduced by this slice.

```bash
git diff --check
```

Result: PASS.

```bash
rg -n "[[:blank:]]$" \
  src/api/v1/vectors.py \
  src/core/vector_migration_qdrant_preview.py \
  tests/unit/test_vector_migration_qdrant_preview.py \
  tests/unit/test_vectors_migration_preview_delegation.py \
  docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md \
  docs/development/CAD_ML_VECTOR_MIGRATION_QDRANT_PREVIEW_HELPER_OWNERSHIP_DEVELOPMENT_20260515.md \
  docs/development/CAD_ML_VECTOR_MIGRATION_QDRANT_PREVIEW_HELPER_OWNERSHIP_VERIFICATION_20260515.md
```

Result: PASS, no matches.

## Claude Code Review

Claude Code was invoked with `--tools ""` and received only targeted snippets
for the core collector, facade alias, preview route, and relevant tests.

Conclusion: compatibility is intact. The helper remains a module attribute on
`src.api.v1.vectors`, and the preview route reads that attribute at request time
before passing it into the shared preview pipeline.

## Residual Risk

Patching `src.core.vector_migration_qdrant_preview.collect_qdrant_preview_samples`
does not affect the preview route because the route intentionally uses the
`src.api.v1.vectors` facade. Keep using the facade patch path until that
compatibility surface is intentionally retired.
