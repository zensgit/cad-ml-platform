# CAD ML Vector Migration Pending Memory Helper Ownership Verification 20260515

## Scope

Verification for moving in-memory pending-candidate branch logic from
`src/core/vector_migration_pending_candidates.py` into
`src/core/vector_migration_pending_memory.py` while preserving the shared
entrypoint and API facade wrapper.

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
  src/core/vector_migration_pending_candidates.py \
  src/core/vector_migration_pending_memory.py \
  src/core/vector_migration_pending_qdrant.py \
  src/api/v1/vectors.py \
  tests/unit/test_vector_migration_pending_candidates.py \
  tests/unit/test_vector_migration_pending_memory.py \
  tests/unit/test_vector_migration_pending_qdrant.py \
  tests/unit/test_vectors_migration_pending_candidates_delegation.py
```

Result: PASS.

```bash
.venv311/bin/flake8 \
  src/core/vector_migration_pending_candidates.py \
  src/core/vector_migration_pending_memory.py \
  src/core/vector_migration_pending_qdrant.py \
  tests/unit/test_vector_migration_pending_candidates.py \
  tests/unit/test_vector_migration_pending_memory.py \
  tests/unit/test_vector_migration_pending_qdrant.py \
  tests/unit/test_vectors_migration_pending_candidates_delegation.py
```

Result: PASS.

```bash
.venv311/bin/pytest -q \
  tests/unit/test_vector_migration_pending_memory.py \
  tests/unit/test_vector_migration_pending_qdrant.py \
  tests/unit/test_vector_migration_pending_candidates.py \
  tests/unit/test_vectors_migration_pending_candidates_delegation.py \
  tests/unit/test_vector_migration_pending.py \
  tests/unit/test_vector_migration_pending_summary.py \
  tests/unit/test_vector_migration_plan.py \
  tests/unit/test_vector_migration_pending_run.py \
  tests/unit/test_vectors_migration_plan_delegation.py \
  tests/unit/test_vectors_migration_pending_run_delegation.py \
  tests/unit/test_vectors_migration_config_delegation.py
```

Result: PASS, `32 passed, 7 warnings in 2.30s`.

Warnings are from `ezdxf/queryparser.py` PyParsing deprecations and are not
introduced by this slice.

```bash
rg -n "collect_memory_migration_pending_candidates\(" src tests docs/development | head -100
```

Result: PASS. Direct helper calls are limited to the shared entrypoint and
focused helper tests.

```bash
git diff --check
```

Result: PASS.

```bash
rg -n "[[:blank:]]$" \
  src/core/vector_migration_pending_candidates.py \
  src/core/vector_migration_pending_memory.py \
  src/core/vector_migration_pending_qdrant.py \
  tests/unit/test_vector_migration_pending_candidates.py \
  tests/unit/test_vector_migration_pending_memory.py \
  tests/unit/test_vector_migration_pending_qdrant.py \
  tests/unit/test_vectors_migration_pending_candidates_delegation.py \
  docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md \
  docs/development/CAD_ML_VECTOR_MIGRATION_PENDING_MEMORY_HELPER_OWNERSHIP_DEVELOPMENT_20260515.md \
  docs/development/CAD_ML_VECTOR_MIGRATION_PENDING_MEMORY_HELPER_OWNERSHIP_VERIFICATION_20260515.md
```

Result: PASS, no matches.

## Claude Code Review

Claude Code was invoked with `--tools ""` and received only targeted snippets
for the memory pending helper, shared entrypoint, direct helper tests, and API
facade delegation test.

Conclusion: behavior is preserved. Claude Code specifically checked:

- filter normalization staying in the shared entrypoint.
- missing-vector skip behavior before `scanned_vectors` is incremented.
- exact memory counts and complete version distribution.
- `pending_ids[:limit]` and item capping behavior.
- unchanged API facade wrapper compatibility.

## Residual Risk

The new memory helper expects `normalized_filter` to already be normalized. Raw
request filters should continue to flow through
`collect_vector_migration_pending_candidates`, which remains the public shared
entrypoint and performs that normalization.
