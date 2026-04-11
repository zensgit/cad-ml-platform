# CI Workflow Guardrail Overview Integration Validation

Date: 2026-03-17
Branch: `feat/hybrid-blind-drift-autotune-e2e`

## Scope

This change wires the existing CI workflow guardrail overview into the evaluation report flow:

- optional build step in [evaluation-report.yml](/Users/huazhou/Downloads/Github/cad-ml-platform/.github/workflows/evaluation-report.yml)
- step summary append in [evaluation-report.yml](/Users/huazhou/Downloads/Github/cad-ml-platform/.github/workflows/evaluation-report.yml)
- PR comment summarization in [comment_evaluation_report_pr.js](/Users/huazhou/Downloads/Github/cad-ml-platform/scripts/ci/comment_evaluation_report_pr.js)

Updated files:

- [evaluation-report.yml](/Users/huazhou/Downloads/Github/cad-ml-platform/.github/workflows/evaluation-report.yml)
- [comment_evaluation_report_pr.js](/Users/huazhou/Downloads/Github/cad-ml-platform/scripts/ci/comment_evaluation_report_pr.js)
- [test_evaluation_report_workflow_graph2d_extensions.py](/Users/huazhou/Downloads/Github/cad-ml-platform/tests/unit/test_evaluation_report_workflow_graph2d_extensions.py)
- [test_comment_evaluation_report_pr_js.py](/Users/huazhou/Downloads/Github/cad-ml-platform/tests/unit/test_comment_evaluation_report_pr_js.py)

## Design

- New env passthrough:
  - `CI_WORKFLOW_GUARDRAIL_OVERVIEW_JSON_FOR_COMMENT`
- New optional workflow step:
  - `Build CI workflow guardrail overview for PR comment (optional)`
- New step summary section:
  - `CI Workflow Guardrail Overview`
- New PR comment summary rows:
  - `CI Workflow Guardrail Overview`
  - `CI+Workflow Overview`

The integration is additive. It does not remove the existing `CI Watch Failure Details` or `Workflow Guardrail Summary` rows.

## Validation

Executed:

```bash
pytest -q tests/unit/test_comment_evaluation_report_pr_js.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

node --check scripts/ci/comment_evaluation_report_pr.js

TMPDIR=$PWD/.tmp_pytest \
PYTEST_ADDOPTS="--basetemp=$PWD/.tmp_pytest/basetemp" \
make validate-eval-with-history-ci-workflows
```

Observed results:

- `pytest -q tests/unit/test_comment_evaluation_report_pr_js.py tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
  - `21 passed`
- `node --check scripts/ci/comment_evaluation_report_pr.js`
  - passed
- `TMPDIR=$PWD/.tmp_pytest PYTEST_ADDOPTS="--basetemp=$PWD/.tmp_pytest/basetemp" make validate-eval-with-history-ci-workflows`
  - `37 passed`
- `TMPDIR=$PWD/.tmp_pytest PYTEST_ADDOPTS="--basetemp=$PWD/.tmp_pytest/basetemp" make validate-ci-watchers`
  - passed end-to-end

Key integrated signals:

- `CI Workflow Guardrail Overview`
- `CI+Workflow Overview`

The PR comment path still preserves:

- `CI Watch Failure Details`
- `Workflow Guardrail Summary`

The new overview is an additional aggregate signal, not a replacement.
