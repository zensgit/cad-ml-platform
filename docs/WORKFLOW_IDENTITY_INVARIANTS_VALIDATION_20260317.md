# Workflow Identity Invariants Validation

Date: 2026-03-17

## Goal

Add a dedicated guardrail for the most brittle workflow integration points:

- workflow file names must stay on `.yml`
- critical workflow display names must not drift
- wrapper workflows must keep their `workflow_dispatch.inputs`
- `Evaluation Report` must remain present in `CI_WATCH_REQUIRED_WORKFLOWS`
- every name in `CI_WATCH_REQUIRED_WORKFLOWS` must resolve to exactly one `.yml` workflow

This reduces the chance that wrappers, Make targets, or CI watchers silently break after a rename.

## Added

- `scripts/ci/check_workflow_identity_invariants.py`
- `tests/unit/test_check_workflow_identity_invariants.py`
- `Makefile` targets:
  - `validate-workflow-identity`
  - `validate-workflow-identity-tests`

`validate-ci-watchers` now also invokes `validate-workflow-identity-tests`.

## Security Audit conflict cleanup

The duplicate workflow display name was removed by keeping:

- `security-audit.yml` -> `Security Audit`

and renaming:

- `security-check.yml` -> `Security Check`

This keeps the watcher-required `Security Audit` identity unique without changing `CI_WATCH_REQUIRED_WORKFLOWS`.

## Covered workflows

- `ci.yml`
- `code-quality.yml`
- `security-audit.yml`
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

- `pytest -q tests/unit/test_check_workflow_identity_invariants.py tests/unit/test_workflow_file_health_make_target.py` -> `12 passed`
- `make validate-workflow-identity` -> `ok: workflow identity check passed for 11 checks`
- `make validate-workflow-identity-tests` -> `12 passed`
- `make validate-ci-watchers` -> all nested targets passed, including the new `validate-workflow-identity-tests` step

## Notes

- This guard is intentionally narrow and avoids changing existing watcher logic.
- This guard now covers both selected workflow file identities and the full watcher-required workflow name mapping.
