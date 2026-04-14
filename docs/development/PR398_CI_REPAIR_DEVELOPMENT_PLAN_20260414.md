# PR398 CI Repair Development Plan

Date: 2026-04-14
Owner: Codex
Branch: `submit/local-main-20260414`
PR: `#398`

## Background

After `Governance Gates` and `Evaluation Report` were confirmed green on GitHub, PR `#398` still had two code-adjacent failures worth fixing:

1. `CI Tiered Tests` / `core-fast-gate`
   - Failed on `tests/contract/test_openapi_schema_snapshot.py`
   - Root cause: `config/openapi_schema_snapshot.json` was stale and no longer matched the normalized snapshot shape emitted by the current generator

2. `CI Enhanced`
   - Failed on `Unit Tests (Shard 1)` at `Run eval_with_history regression unit tests`
   - Root cause: workflow referenced a deleted test file:
     - `tests/unit/test_eval_with_history_script_history_sequence.py`

## Goals

1. Refresh the OpenAPI snapshot baseline so `validate-core-fast` is green again.
2. Repair the `eval_with_history` regression step across all CI workflows that reference the missing test file.
3. Keep the CI repair minimal:
   - do not change runtime API routes
   - do not broaden workflow test scope unnecessarily
4. Produce verification records for both local and GitHub-triggered checks.

## Implementation Plan

### 1. Refresh OpenAPI snapshot baseline

Run the canonical snapshot generator:

```bash
.venv311/bin/python scripts/ci/generate_openapi_schema_snapshot.py \
  --output config/openapi_schema_snapshot.json
```

Then re-run the OpenAPI contract tests and `validate-core-fast`.

### 2. Repair `eval_with_history` workflow regression step

Update these workflow files:

- `.github/workflows/ci-enhanced.yml`
- `.github/workflows/ci-tiered-tests.yml`
- `.github/workflows/ci.yml`

Replace the deleted test path with the existing focused test nodes:

- `tests/unit/test_graph2d_eval_helpers.py::test_eval_with_history_script_has_valid_bash_syntax`
- `tests/unit/test_graph2d_eval_helpers.py::test_eval_with_history_writes_coarse_history_metrics`
- `tests/unit/test_validate_eval_history_history_sequence.py`

Rationale:

- `test_graph2d_eval_helpers.py` contains the surviving `eval_with_history` regression coverage
- the full file also contains unrelated `eval_trend` coverage that currently drifts independently
- using explicit node IDs preserves the original intent of the workflow gate

### 3. Verification

Run:

```bash
.venv311/bin/python -m pytest -q \
  tests/contract/test_openapi_schema_snapshot.py \
  tests/contract/test_openapi_operation_ids.py \
  tests/unit/test_api_route_uniqueness.py

make validate-core-fast

.venv311/bin/python -m pytest -q \
  tests/unit/test_graph2d_eval_helpers.py::test_eval_with_history_script_has_valid_bash_syntax \
  tests/unit/test_graph2d_eval_helpers.py::test_eval_with_history_writes_coarse_history_metrics \
  tests/unit/test_validate_eval_history_history_sequence.py

.venv311/bin/python scripts/ci/check_workflow_identity_invariants.py
python3 scripts/ci/generate_workflow_inventory_report.py \
  --workflow-root .github/workflows \
  --ci-watch-required-workflows "CI,CI Enhanced,CI Tiered Tests,Code Quality,Multi-Architecture Docker Build,Security Audit,Observability Checks,Self-Check,GHCR Publish,Evaluation Report,Governance Gates" \
  --output-json /tmp/workflow_inventory_report_ci.json \
  --output-md /tmp/workflow_inventory_report_ci.md
```

### 4. GitHub verification

Push the repair commits to PR `#398` and observe:

- `Governance Gates`
- `Evaluation Report`
- `CI Enhanced`
- `Stress and Observability Checks`

## Expected Outcome

- OpenAPI snapshot contract is restored
- `core-fast-gate` no longer fails on stale snapshot baseline
- `CI Enhanced` no longer fails on a missing test file
- watcher-required workflow inventory remains healthy
