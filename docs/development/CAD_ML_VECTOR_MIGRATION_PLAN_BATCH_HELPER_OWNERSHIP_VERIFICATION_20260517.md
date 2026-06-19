# CAD ML Vector Migration Plan Batch Helper Ownership Verification 20260517

## Scope

Verification for moving vector migration plan ranking, batch construction, and
run estimation from `src/core/vector_migration_plan_pipeline.py` into
`src/core/vector_migration_plan_batches.py` while preserving the shared plan
payload entrypoints and route-level facade behavior.

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
  src/core/vector_migration_plan_batches.py \
  src/core/vector_migration_plan_pipeline.py \
  tests/unit/test_vector_migration_plan_batches.py \
  tests/unit/test_vector_migration_plan_pipeline.py \
  tests/unit/test_vector_migration_plan.py \
  tests/unit/test_vectors_migration_plan_delegation.py
```

Result: PASS.

```bash
.venv311/bin/flake8 \
  src/core/vector_migration_plan_batches.py \
  src/core/vector_migration_plan_pipeline.py \
  tests/unit/test_vector_migration_plan_batches.py \
  tests/unit/test_vector_migration_plan_pipeline.py \
  tests/unit/test_vector_migration_plan.py \
  tests/unit/test_vectors_migration_plan_delegation.py
```

Result: PASS.

```bash
.venv311/bin/pytest -q \
  tests/unit/test_vector_migration_plan_batches.py \
  tests/unit/test_vector_migration_plan_pipeline.py \
  tests/unit/test_vector_migration_plan.py \
  tests/unit/test_vectors_migration_plan_delegation.py \
  tests/unit/test_vector_migration_pending_summary.py \
  tests/unit/test_vector_migration_pending_candidates.py \
  tests/unit/test_vector_migration_pending_memory.py \
  tests/unit/test_vector_migration_pending_qdrant.py \
  tests/unit/test_vectors_migration_pending_candidates_delegation.py \
  tests/unit/test_vector_migration_pending.py \
  tests/unit/test_vector_migration_pending_run.py \
  tests/unit/test_vectors_migration_pending_run_delegation.py \
  tests/unit/test_vectors_migration_config_delegation.py
```

Result: PASS, `38 passed, 7 warnings in 3.08s`.

Warnings are from `ezdxf/queryparser.py` PyParsing deprecations and are not
introduced by this slice.

```bash
rg -n "rank_observed_versions\(|build_vector_migration_plan_batches\(|estimate_migration_runs_by_version\(" \
  src tests docs/development | head -100
```

Result: PASS. Direct helper calls are limited to the plan payload module and
focused helper tests.

```bash
git diff --check
```

Result: PASS.

```bash
rg -n "[[:blank:]]$" \
  src/core/vector_migration_plan_batches.py \
  src/core/vector_migration_plan_pipeline.py \
  tests/unit/test_vector_migration_plan_batches.py \
  tests/unit/test_vector_migration_plan_pipeline.py \
  tests/unit/test_vector_migration_plan.py \
  tests/unit/test_vectors_migration_plan_delegation.py \
  docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md \
  docs/development/CAD_ML_VECTOR_MIGRATION_PLAN_BATCH_HELPER_OWNERSHIP_DEVELOPMENT_20260517.md \
  docs/development/CAD_ML_VECTOR_MIGRATION_PLAN_BATCH_HELPER_OWNERSHIP_VERIFICATION_20260517.md
```

Result: PASS, no matches.

## Claude Code Review

Claude Code was invoked with `--tools ""` and received only targeted snippets
for the new plan-batch helper and its call sites.

Conclusion: behavior is preserved. Claude Code specifically checked:

- count-descending and name-ascending rank ordering.
- `max_batches` truncation before batch construction.
- full `recommended_from_versions` retention for unplanned-version detection.
- split-batch vs single-batch note selection.
- partial-scan override propagation into payload, notes, and blocking reasons.
- ceiling-based estimated run counts across all observed versions.
- unchanged `build_vector_migration_plan_payload` public entrypoint.

## Residual Risk

The new helper intentionally owns pure planning math only. Request validation,
API schema compatibility, pending-candidate collection, and facade monkeypatch
surfaces remain owned by the existing pipeline and router modules.
