# CAD ML Vectors Admin Router Contract Verification

Date: 2026-05-15

## Scope

Verified the backend reload admin router split contract: route ownership,
single registration, HTTP method, trailing-slash absence, OpenAPI operationId,
response schema, admin-token rejection, OpenAPI global contract tests, Claude
Code read-only review, and focused regression coverage.

## Commands

```bash
<targeted snippets> | claude -p --tools "" --max-budget-usd 0.50
```

Result: Claude Code identified high-value coverage gaps in duplicate
registration, trailing-slash duplicates, response schema anchoring, operationId
uniqueness, and admin auth contract coverage. The contract tests were expanded
to cover those points.

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache .venv311/bin/python -m py_compile \
  src/api/v1/vectors.py \
  src/api/v1/vectors_admin_router.py \
  tests/unit/test_vectors_admin_router.py
```

Result: passed.

```bash
.venv311/bin/flake8 \
  src/api/v1/vectors.py \
  src/api/v1/vectors_admin_router.py \
  tests/unit/test_vectors_admin_router.py
```

Result: passed.

```bash
.venv311/bin/pytest -q \
  tests/unit/test_vectors_admin_router.py \
  tests/unit/test_vectors_backend_reload_delegation.py \
  tests/unit/test_vectors_backend_reload_admin_token.py \
  tests/unit/test_vector_backend_reload_failure.py \
  tests/contract/test_openapi_operation_ids.py \
  tests/contract/test_openapi_schema_snapshot.py
```

Result: `15 passed, 7 warnings in 2.43s`.

Warnings: existing third-party `ezdxf`/`pyparsing` deprecation warnings surfaced
during app import. No project-code failure.

```bash
git diff --check
```

Result: passed.

```bash
rg -n "[ \t]+$" \
  src/api/v1/vectors.py \
  src/api/v1/vectors_admin_router.py \
  tests/unit/test_vectors_admin_router.py \
  docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md \
  docs/development/CAD_ML_VECTORS_ADMIN_ROUTER_CONTRACT_DEVELOPMENT_20260515.md \
  docs/development/CAD_ML_VECTORS_ADMIN_ROUTER_CONTRACT_VERIFICATION_20260515.md
```

Result: no trailing whitespace matches.

## Coverage

- `POST /api/v1/vectors/backend/reload` is registered exactly once.
- The route is owned by `src.api.v1.vectors_admin_router`.
- The route exposes only `POST` and has no trailing-slash duplicate.
- OpenAPI operationId remains
  `reload_vector_backend_api_v1_vectors_backend_reload_post`.
- The operationId remains globally unique.
- OpenAPI summary, tag, query/header parameters, request-body absence, and
  `VectorBackendReloadResponse` response schema are pinned.
- Requests without an admin token remain rejected with `401`.
- No live `/api/v1/vectors` route endpoint points back to the facade module.
- Existing backend reload delegation, admin-token, failure, operationId, and
  snapshot tests remain green.

## Residual Risk

- This slice does not move additional helpers out of `vectors.py`.
- Broader Phase 3 router closeout still needs list/similarity/helper ownership
  work in separate, compatibility-preserving slices.
