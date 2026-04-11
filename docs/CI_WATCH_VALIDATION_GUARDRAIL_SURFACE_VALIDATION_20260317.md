# CI Watch Validation Guardrail Surface Validation

## Scope

Extend `generate_ci_watcher_validation_report.py` so the generated watcher validation markdown
surfaces both higher-level guardrail artifacts:

- `workflow_guardrail_summary.json`
- `ci_workflow_guardrail_overview.json`

This keeps the watcher validation report aligned with the newer workflow guardrail reporting
chain and reduces reviewer context switching across separate artifacts.

## Files

- `scripts/ci/generate_ci_watcher_validation_report.py`
- `Makefile`
- `tests/unit/test_generate_ci_watcher_validation_report.py`
- `tests/unit/test_watch_commit_workflows_make_target.py`

## Changes

- Added optional CLI flags:
  - `--workflow-guardrail-summary-json`
  - `--ci-workflow-guardrail-overview-json`
- Added inferred-path fallback logic relative to the watcher summary directory.
- Added two new markdown sections:
  - `Workflow Guardrail Summary`
  - `CI Workflow Guardrail Overview`
- Extended the `generate-ci-watch-validation-report` make target to pass both artifact paths.
- Updated tests to cover:
  - successful rendering when both guardrail artifacts are present
  - missing-artifact messages when they are absent
  - make target flag passthrough

## Validation

Commands run:

```bash
pytest -q tests/unit/test_generate_ci_watcher_validation_report.py tests/unit/test_watch_commit_workflows_make_target.py
make validate-generate-ci-watch-validation-report
make generate-ci-watch-validation-report
TMPDIR=$PWD/.tmp_pytest PYTEST_ADDOPTS="--basetemp=$PWD/.tmp_pytest/basetemp" make validate-ci-watchers
```

Observed results:

- `pytest -q tests/unit/test_generate_ci_watcher_validation_report.py tests/unit/test_watch_commit_workflows_make_target.py` -> `19 passed`
- `make validate-generate-ci-watch-validation-report` -> passed
- `make generate-ci-watch-validation-report` -> passed
- `make validate-ci-watchers` -> passed

## Outcome

The CI watcher validation markdown now includes both guardrail layers directly, so the watcher
report can be used as a single review surface for:

- readiness
- soft-mode smoke
- workflow guardrail summary
- CI workflow guardrail overview
- watcher summary and failure details
