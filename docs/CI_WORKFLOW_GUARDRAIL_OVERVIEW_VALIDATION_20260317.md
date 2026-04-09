# CI Workflow Guardrail Overview Validation

Date: 2026-03-17
Branch: `feat/hybrid-blind-drift-autotune-e2e`

## Scope

This change adds a thin aggregate layer over:

- CI watcher commit summary
- workflow guardrail summary

Primary files:

- [generate_ci_workflow_guardrail_overview.py](/Users/huazhou/Downloads/Github/cad-ml-platform/scripts/ci/generate_ci_workflow_guardrail_overview.py)
- [Makefile](/Users/huazhou/Downloads/Github/cad-ml-platform/Makefile)
- [test_generate_ci_workflow_guardrail_overview.py](/Users/huazhou/Downloads/Github/cad-ml-platform/tests/unit/test_generate_ci_workflow_guardrail_overview.py)
- [test_watch_commit_workflows_make_target.py](/Users/huazhou/Downloads/Github/cad-ml-platform/tests/unit/test_watch_commit_workflows_make_target.py)

## Design

- Input A: latest or explicit `watch_commit_*_summary.json`
- Input B: explicit `workflow_guardrail_summary.json`
- Output:
  - `reports/ci/ci_workflow_guardrail_overview.json`
  - `reports/ci/ci_workflow_guardrail_overview.md`

The overview reduces reviewer jumps by collapsing watcher status and workflow guardrail status into one release-gate summary.

## Validation

Executed:

```bash
pytest -q tests/unit/test_generate_ci_workflow_guardrail_overview.py \
  tests/unit/test_watch_commit_workflows_make_target.py

make validate-generate-ci-workflow-guardrail-overview
make generate-ci-workflow-guardrail-overview
TMPDIR=$PWD/.tmp_pytest PYTEST_ADDOPTS="--basetemp=$PWD/.tmp_pytest/basetemp" make validate-ci-watchers
```

Observed results:

- `pytest -q tests/unit/test_generate_ci_workflow_guardrail_overview.py tests/unit/test_watch_commit_workflows_make_target.py`
  - `18 passed`
- `make validate-generate-ci-workflow-guardrail-overview`
  - `18 passed`
- `make generate-ci-workflow-guardrail-overview`
  - generated:
    - [ci_workflow_guardrail_overview.json](/Users/huazhou/Downloads/Github/cad-ml-platform/reports/ci/ci_workflow_guardrail_overview.json)
    - [ci_workflow_guardrail_overview.md](/Users/huazhou/Downloads/Github/cad-ml-platform/reports/ci/ci_workflow_guardrail_overview.md)
- `TMPDIR=$PWD/.tmp_pytest PYTEST_ADDOPTS="--basetemp=$PWD/.tmp_pytest/basetemp" make validate-ci-watchers`
  - passed end-to-end

## Notes

- The overview is additive. It does not replace the existing CI watcher validation report or workflow guardrail report.
- The generator can auto-pick the latest watcher summary from `reports/ci`.
