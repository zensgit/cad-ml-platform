# Workflow Identity Invariants Validation

Date: 2026-03-17

## Goal

Add a dedicated guardrail for the most brittle workflow integration points:

- workflow file names must stay on `.yml`
- critical workflow display names must not drift
- wrapper workflows must keep their `workflow_dispatch.inputs`
- `Evaluation Report` must remain present in `CI_WATCH_REQUIRED_WORKFLOWS`

This reduces the chance that wrappers, Make targets, or CI watchers silently break after a rename.

## Added

- `scripts/ci/check_workflow_identity_invariants.py`
- `tests/unit/test_check_workflow_identity_invariants.py`
- `Makefile` targets:
  - `validate-workflow-identity`
  - `validate-workflow-identity-tests`

`validate-ci-watchers` now also invokes `validate-workflow-identity-tests`.

## Covered workflows

- `ci.yml`
- `code-quality.yml`
- `evaluation-report.yml`
- `evaluation-soft-mode-smoke.yml`
- `hybrid-superpass-e2e.yml`
- `hybrid-blind-strict-real-e2e.yml`
- `experiment-archive-dry-run.yml`
- `experiment-archive-apply.yml`
- `stress-tests.yml`

## Validation

```bash
pytest -q \
  tests/unit/test_check_workflow_identity_invariants.py \
  tests/unit/test_workflow_file_health_make_target.py
```

```bash
make validate-workflow-identity
```

```bash
make validate-workflow-identity-tests
```

```bash
make validate-ci-watchers
```

Results:

- `pytest -q tests/unit/test_check_workflow_identity_invariants.py tests/unit/test_workflow_file_health_make_target.py` -> `11 passed`
- `make validate-workflow-identity` -> `ok: workflow identity check passed for 9 workflow(s)`
- `make validate-workflow-identity-tests` -> `11 passed`
- `make validate-ci-watchers` -> all nested targets passed, including the new `validate-workflow-identity-tests` step

## Notes

- This guard is intentionally narrow and avoids changing existing watcher logic.
- Duplicate non-critical workflow display names, such as the current `Security Audit` collision, remain a separate cleanup topic.
