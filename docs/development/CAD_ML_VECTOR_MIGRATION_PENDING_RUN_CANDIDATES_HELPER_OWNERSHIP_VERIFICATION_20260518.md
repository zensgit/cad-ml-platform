# CAD ML Vector Migration Pending Run Candidates Helper Ownership Verification 20260518

## Scope

Verification for moving vector migration pending-run candidate collection
argument mapping from `src/core/vector_migration_pending_run_pipeline.py` into
`src/core/vector_migration_pending_run_candidates.py` while preserving the shared
pending-run execution entrypoint and route-level facade behavior.

## Results

| Check | Result |
| --- | --- |
| Python compile | PASS |
| flake8 target files | PASS |
| pytest target slice | PASS |
| Claude Code read-only compatibility review | PASS |
| direct helper call-site grep | PASS |
| public facade call-site grep | PASS |
| git diff whitespace check | PASS |
| trailing whitespace scan | PASS |

## Commands

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache .venv311/bin/python -m py_compile \
  src/core/vector_migration_pending_run_candidates.py \
  src/core/vector_migration_pending_run_request.py \
  src/core/vector_migration_pending_run_guard.py \
  src/core/vector_migration_pending_run_pipeline.py \
  tests/unit/test_vector_migration_pending_run_candidates.py \
  tests/unit/test_vector_migration_pending_run_request.py \
  tests/unit/test_vector_migration_pending_run_guard.py \
  tests/unit/test_vector_migration_pending_run_pipeline.py \
  tests/unit/test_vector_migration_pending_run.py \
  tests/unit/test_vectors_migration_pending_run_delegation.py
```

Result: PASS.

```bash
.venv311/bin/flake8 \
  src/core/vector_migration_pending_run_candidates.py \
  src/core/vector_migration_pending_run_request.py \
  src/core/vector_migration_pending_run_guard.py \
  src/core/vector_migration_pending_run_pipeline.py \
  tests/unit/test_vector_migration_pending_run_candidates.py \
  tests/unit/test_vector_migration_pending_run_request.py \
  tests/unit/test_vector_migration_pending_run_guard.py \
  tests/unit/test_vector_migration_pending_run_pipeline.py \
  tests/unit/test_vector_migration_pending_run.py \
  tests/unit/test_vectors_migration_pending_run_delegation.py
```

Result: PASS.

```bash
.venv311/bin/pytest -q \
  tests/unit/test_vector_migration_pending_run_candidates.py \
  tests/unit/test_vector_migration_pending_run_request.py \
  tests/unit/test_vector_migration_pending_run_guard.py \
  tests/unit/test_vector_migration_pending_run_pipeline.py \
  tests/unit/test_vector_migration_pending_run.py \
  tests/unit/test_vectors_migration_pending_run_delegation.py \
  tests/unit/test_vector_migration_pending_candidates.py \
  tests/unit/test_vector_migration_pending_memory.py \
  tests/unit/test_vector_migration_pending_qdrant.py \
  tests/unit/test_vectors_migration_pending_candidates_delegation.py \
  tests/unit/test_vector_migration_pending.py \
  tests/unit/test_vector_migration_pending_summary.py \
  tests/unit/test_vector_migration_plan_pipeline.py \
  tests/unit/test_vector_migration_plan.py \
  tests/unit/test_vectors_migration_plan_delegation.py \
  tests/unit/test_vectors_migration_config_delegation.py
```

Result: PASS, `44 passed, 7 warnings in 3.69s`.

Warnings are from `ezdxf/queryparser.py` PyParsing deprecations and are not
introduced by this slice.

```bash
rg -n "collect_pending_run_candidates\(" src tests docs/development | head -100
```

Result: PASS. Direct helper calls are limited to the pending-run pipeline and
focused helper tests.

```bash
rg -n "run_vector_migration_pending_run_pipeline" \
  src/api/v1/vectors.py \
  src/api/v1/vectors_write_router.py \
  tests/unit/test_vectors_migration_pending_run_delegation.py
```

Result: PASS. The route-level facade patch surface remains
`src.api.v1.vectors.run_vector_migration_pending_run_pipeline`.

```bash
git diff --check
```

Result: PASS.

```bash
rg -n "[[:blank:]]$" \
  src/core/vector_migration_pending_run_candidates.py \
  src/core/vector_migration_pending_run_request.py \
  src/core/vector_migration_pending_run_guard.py \
  src/core/vector_migration_pending_run_pipeline.py \
  tests/unit/test_vector_migration_pending_run_candidates.py \
  tests/unit/test_vector_migration_pending_run_request.py \
  tests/unit/test_vector_migration_pending_run_guard.py \
  tests/unit/test_vector_migration_pending_run_pipeline.py \
  tests/unit/test_vector_migration_pending_run.py \
  tests/unit/test_vectors_migration_pending_run_delegation.py \
  docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md \
  docs/development/CAD_ML_VECTOR_MIGRATION_PENDING_RUN_CANDIDATES_HELPER_OWNERSHIP_DEVELOPMENT_20260518.md \
  docs/development/CAD_ML_VECTOR_MIGRATION_PENDING_RUN_CANDIDATES_HELPER_OWNERSHIP_VERIFICATION_20260518.md
```

Result: PASS, no matches.

## Claude Code Review

Claude Code was invoked with `--tools ""` and received only targeted snippets
for the new pending-run candidates helper, pending-run pipeline call site, and
focused tests.

Conclusion: behavior is preserved. Claude Code returned OK for:

- `payload.limit` passing to collector `limit`.
- `target_version` pass-through.
- `payload.from_version_filter` pass-through, including `None`.
- pending dict return preservation.
- unchanged public pending-run pipeline entrypoint.

## Residual Risk

The new helper intentionally owns only candidate collection argument mapping.
Partial-scan guarding, request construction, route request validation, API
schema compatibility, and facade monkeypatch surfaces remain owned by the
existing helper, pipeline, and router modules.
