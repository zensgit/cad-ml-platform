# CAD ML Vector List Qdrant Helper Ownership Verification 20260515

## Scope

Verification for moving Qdrant vector-list branch logic from
`src/core/vector_list_pipeline.py` into `src/core/vector_list_qdrant.py` while
preserving source resolution, filter-builder injection, extractor patching, and
fallback behavior.

## Results

| Check | Result |
| --- | --- |
| Python compile | PASS |
| flake8 target files | PASS |
| pytest target slice | PASS |
| Claude Code read-only compatibility review | PASS |
| direct helper call-site grep | PASS |
| git diff whitespace check | PASS |
| trailing whitespace scan | PASS |

## Commands

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache .venv311/bin/python -m py_compile \
  src/core/vector_list_pipeline.py \
  src/core/vector_list_qdrant.py \
  tests/unit/test_vector_list_qdrant.py \
  tests/unit/test_vector_list_pipeline.py \
  tests/unit/test_vectors_list_delegation.py
```

Result: PASS.

```bash
.venv311/bin/flake8 \
  src/core/vector_list_pipeline.py \
  src/core/vector_list_qdrant.py \
  tests/unit/test_vector_list_qdrant.py \
  tests/unit/test_vector_list_pipeline.py \
  tests/unit/test_vectors_list_delegation.py
```

Result: PASS.

```bash
.venv311/bin/pytest -q \
  tests/unit/test_vector_list_qdrant.py \
  tests/unit/test_vector_list_pipeline.py \
  tests/unit/test_vectors_list_delegation.py \
  tests/unit/test_vectors_list_router.py \
  tests/unit/test_vectors_module_endpoints.py \
  tests/contract/test_openapi_operation_ids.py \
  tests/contract/test_openapi_schema_snapshot.py
```

Result: PASS, `30 passed, 7 warnings in 2.47s`.

Warnings are from `ezdxf/queryparser.py` PyParsing deprecations and are not
introduced by this slice.

```bash
rg -n "list_vectors_qdrant\(" src tests docs/development | head -100
```

Result: PASS. Direct helper calls are limited to
`src/core/vector_list_pipeline.py` and `tests/unit/test_vector_list_qdrant.py`,
and all calls pass `extract_label_contract_fn` explicitly.

```bash
rg -n "from src\.core\.vector_list_qdrant import extract_vector_label_contract|vector_list_qdrant import" \
  src tests docs/development | head -100
```

Result: PASS. No compatibility-shim import of
`extract_vector_label_contract` exists from `vector_list_qdrant`.

```bash
git diff --check
```

Result: PASS.

```bash
rg -n "[[:blank:]]$" \
  src/core/vector_list_pipeline.py \
  src/core/vector_list_qdrant.py \
  tests/unit/test_vector_list_qdrant.py \
  tests/unit/test_vector_list_pipeline.py \
  tests/unit/test_vectors_list_delegation.py \
  docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md \
  docs/development/CAD_ML_VECTOR_LIST_QDRANT_HELPER_OWNERSHIP_DEVELOPMENT_20260515.md \
  docs/development/CAD_ML_VECTOR_LIST_QDRANT_HELPER_OWNERSHIP_VERIFICATION_20260515.md
```

Result: PASS, no matches.

## Claude Code Review

Claude Code was invoked with `--tools ""` and received only targeted snippets
for the Qdrant helper, pipeline branch, and relevant tests.

First-pass conclusion: behavior was preserved for the Qdrant source branch,
filter-builder injection, and extractor monkeypatching. Claude Code recommended
two small hardening changes:

- add direct fallback coverage when `source=qdrant` has no Qdrant store.
- make `extract_label_contract_fn` explicit to avoid future default-binding
  patch surprises.

Both recommendations were applied. A final read-only pass found no remaining
compatibility issues.

## Residual Risk

This slice intentionally leaves Redis, memory, and route facade behavior
unchanged. Future cleanup should continue one branch/helper at a time because
list routing still relies on injected functions and environment-driven backend
resolution.
