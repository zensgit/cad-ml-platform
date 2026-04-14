# PR398 CI Repair Verification

Date: 2026-04-14
Owner: Codex
Branch: `submit/local-main-20260414`
PR: `#398`

## Change Summary

This repair batch addressed two PR failures:

1. OpenAPI snapshot drift
   - refreshed `config/openapi_schema_snapshot.json`

2. `eval_with_history` regression step drift in CI workflows
   - updated:
     - `.github/workflows/ci-enhanced.yml`
     - `.github/workflows/ci-tiered-tests.yml`
     - `.github/workflows/ci.yml`
   - replaced deleted test path with focused existing node IDs:
     - `tests/unit/test_graph2d_eval_helpers.py::test_eval_with_history_script_has_valid_bash_syntax`
     - `tests/unit/test_graph2d_eval_helpers.py::test_eval_with_history_writes_coarse_history_metrics`
     - `tests/unit/test_validate_eval_history_history_sequence.py`

## Failure Evidence Collected

### CI Enhanced

GitHub job:

- workflow: `CI Enhanced`
- run: `24392123688`
- failed job: `Unit Tests (Shard 1)`
- failed step: `Run eval_with_history regression unit tests`

Observed failure:

```text
ERROR: file or directory not found: tests/unit/test_eval_with_history_script_history_sequence.py
```

### CI Tiered Tests

GitHub job:

- workflow: `CI Tiered Tests`
- run: `24392123769`
- failed job: `core-fast-gate`

Observed failure:

- `tests/contract/test_openapi_schema_snapshot.py`
- snapshot baseline mismatch against current generated OpenAPI contract

## Local Verification Commands

```bash
.venv311/bin/python scripts/ci/generate_openapi_schema_snapshot.py --output config/openapi_schema_snapshot.json

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
  --output-json /tmp/workflow_inventory_report_ci2.json \
  --output-md /tmp/workflow_inventory_report_ci2.md
```

## Local Verification Results

### 1. OpenAPI contract tests

Result: passed

- `5 passed`

### 2. `validate-core-fast`

Result: passed locally after refreshing the snapshot baseline

Key previously failing OpenAPI section is now green.

### 3. `eval_with_history` regression command

Result: passed

- `4 passed`

This exactly matches the repaired workflow intent without pulling in unrelated `eval_trend` drift.

### 4. Workflow integrity

Result: passed

- `check_workflow_identity_invariants.py` passed
- workflow inventory audit passed
- required workflow mapping still reports:
  - `CI Enhanced: status=ok`
  - `CI Tiered Tests: status=ok`
  - `Evaluation Report: status=ok`
  - `Governance Gates: status=ok`

## Sidecar Review

`Claude Code CLI` was used as read-only sidecar inspection during failure triage.

It confirmed:

- `ci-enhanced.yml` referenced the deleted test file directly
- no Makefile wrapper was involved
- the failure surface was the workflow command itself, not the main unit-test shard

The main repair and validation path did not depend on Claude CLI.

## Conclusion

The PR repair is complete locally:

- stale OpenAPI snapshot baseline refreshed
- missing `eval_with_history` workflow test reference replaced with focused existing regression coverage
- workflow inventory and identity checks remain healthy

GitHub rerun verification is pending the push of this repair batch.
