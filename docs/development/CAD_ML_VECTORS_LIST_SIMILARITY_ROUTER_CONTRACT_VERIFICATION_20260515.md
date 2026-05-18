# CAD ML Vectors List Similarity Router Contract Verification

Date: 2026-05-15

## Scope

Verified vector list and batch-similarity split-router contracts: route
ownership, single registration, operationId stability, OpenAPI schema refs,
response codes, request-body shape, facade helper late-binding, pass-through
query values, Claude Code read-only review, and focused regression coverage.

## Commands

```bash
<targeted snippets> | claude -p --tools "" --max-budget-usd 0.50
```

Result: Claude Code identified high-value gaps around facade ABI protection,
late-binding monkeypatch semantics, pass-through values, response-code
contracts, request-body requiredness, and response field faithfulness. Tests were
expanded to cover those points.

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache .venv311/bin/python -m py_compile \
  src/api/v1/vectors.py \
  src/api/v1/vectors_list_router.py \
  src/api/v1/vectors_similarity_router.py \
  tests/unit/test_vectors_list_router.py \
  tests/unit/test_vectors_similarity_router.py \
  tests/unit/test_vectors_list_delegation.py \
  tests/unit/test_vectors_batch_similarity_delegation.py
```

Result: passed.

```bash
.venv311/bin/flake8 \
  src/api/v1/vectors.py \
  src/api/v1/vectors_list_router.py \
  src/api/v1/vectors_similarity_router.py \
  tests/unit/test_vectors_list_router.py \
  tests/unit/test_vectors_similarity_router.py \
  tests/unit/test_vectors_list_delegation.py \
  tests/unit/test_vectors_batch_similarity_delegation.py
```

Result: passed.

```bash
.venv311/bin/pytest -q \
  tests/unit/test_vectors_list_router.py \
  tests/unit/test_vectors_similarity_router.py \
  tests/unit/test_vectors_list_delegation.py \
  tests/unit/test_vectors_batch_similarity_delegation.py \
  tests/contract/test_openapi_operation_ids.py \
  tests/contract/test_openapi_schema_snapshot.py
```

Result: `11 passed, 7 warnings in 3.13s`.

Warnings: existing third-party `ezdxf`/`pyparsing` deprecation warnings surfaced
during app import. No project-code failure.

```bash
git diff --check
```

Result: passed.

```bash
rg -n "[ \t]+$" \
  tests/unit/test_vectors_list_router.py \
  tests/unit/test_vectors_similarity_router.py \
  tests/unit/test_vectors_list_delegation.py \
  tests/unit/test_vectors_batch_similarity_delegation.py \
  docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md \
  docs/development/CAD_ML_VECTORS_LIST_SIMILARITY_ROUTER_CONTRACT_DEVELOPMENT_20260515.md \
  docs/development/CAD_ML_VECTORS_LIST_SIMILARITY_ROUTER_CONTRACT_VERIFICATION_20260515.md
```

Result: no trailing whitespace matches.

## Coverage

- `GET /api/v1/vectors/` is registered once and owned by
  `src.api.v1.vectors_list_router`.
- The vector list route preserves operationId, summary, tag, response codes,
  parameter names, no request body, and `VectorListResponse` schema ref.
- Vector list delegation preserves pagination, filter values, model classes, and
  late-bound facade helper callables.
- `POST /api/v1/vectors/similarity/batch` is registered once and owned by
  `src.api.v1.vectors_similarity_router`.
- The batch-similarity route preserves operationId, summary, tag, response
  codes, `X-API-Key`, required request body, `BatchSimilarityRequest`, and
  `BatchSimilarityResponse`.
- Batch-similarity delegation preserves payload `ids`, `top_k`, model classes,
  late-bound facade helpers, and representative response fields.
- Global OpenAPI operationId and snapshot tests remain green.

## Residual Risk

- This slice does not move helpers out of `vectors.py`.
- Additional helper ownership cleanup still needs separate compatibility-focused
  slices.
