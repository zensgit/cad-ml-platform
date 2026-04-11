# Workflow Guardrail Summary Validation

Date: 2026-03-17
Branch: `feat/hybrid-blind-drift-autotune-e2e`

## Scope

This change aggregates three existing workflow guardrails into one summary:

- workflow file health
- workflow inventory
- workflow publish-helper adoption

The aggregate summary is produced by:

- [generate_workflow_guardrail_summary.py](/Users/huazhou/Downloads/Github/cad-ml-platform/scripts/ci/generate_workflow_guardrail_summary.py)

It is surfaced in:

- [stress-tests.yml](/Users/huazhou/Downloads/Github/cad-ml-platform/.github/workflows/stress-tests.yml)
- [evaluation-report.yml](/Users/huazhou/Downloads/Github/cad-ml-platform/.github/workflows/evaluation-report.yml)
- [comment_evaluation_report_pr.js](/Users/huazhou/Downloads/Github/cad-ml-platform/scripts/ci/comment_evaluation_report_pr.js)

## Design

- The aggregate JSON carries `overall_status`, `overall_light`, and a compact `summary`.
- `stress-tests` uploads both JSON and Markdown and appends the Markdown to `GITHUB_STEP_SUMMARY`.
- `evaluation-report` rebuilds the guardrail report for PR comment use from the three optional per-run JSON files.
- The PR comment renderer keeps the detailed rows and adds a new aggregated row:
  - `Workflow Guardrail Summary`
  - `Workflow Guardrails`

## Validation

Executed:

```bash
pytest -q tests/unit/test_generate_workflow_guardrail_summary.py \
  tests/unit/test_stress_workflow_workflow_file_health.py \
  tests/unit/test_comment_evaluation_report_pr_js.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  tests/unit/test_workflow_file_health_make_target.py

make validate-workflow-guardrail-summary-report
make validate-workflow-file-health-tests
TMPDIR=$PWD/.tmp_pytest PYTEST_ADDOPTS="--basetemp=$PWD/.tmp_pytest/basetemp" make validate-eval-with-history-ci-workflows
TMPDIR=$PWD/.tmp_pytest PYTEST_ADDOPTS="--basetemp=$PWD/.tmp_pytest/basetemp" make validate-ci-watchers
```

Observed results:

- `pytest -q tests/unit/test_generate_workflow_guardrail_summary.py tests/unit/test_stress_workflow_workflow_file_health.py tests/unit/test_comment_evaluation_report_pr_js.py tests/unit/test_evaluation_report_workflow_graph2d_extensions.py tests/unit/test_workflow_file_health_make_target.py`
  - `38 passed`
- `make validate-workflow-guardrail-summary-report`
  - `16 passed`
- `make validate-workflow-file-health-tests`
  - `23 passed`
- `TMPDIR=$PWD/.tmp_pytest PYTEST_ADDOPTS="--basetemp=$PWD/.tmp_pytest/basetemp" make validate-eval-with-history-ci-workflows`
  - `35 passed`
- `TMPDIR=$PWD/.tmp_pytest PYTEST_ADDOPTS="--basetemp=$PWD/.tmp_pytest/basetemp" make validate-ci-watchers`
  - passed end-to-end

Generated artifacts:

- [workflow_guardrail_summary.json](/Users/huazhou/Downloads/Github/cad-ml-platform/reports/ci/workflow_guardrail_summary.json)
- [workflow_guardrail_summary.md](/Users/huazhou/Downloads/Github/cad-ml-platform/reports/ci/workflow_guardrail_summary.md)

Expected assertions covered:

- guardrail report JSON/Markdown generation
- `stress-tests` artifact + step-summary wiring
- `evaluation-report` optional build + append wiring
- PR comment env passthrough
- JS summarizer happy path and parse-error handling
- aggregate signal rows rendered into the PR comment body

## Notes

- The new guardrail summary is additive. Existing workflow file health, inventory, and publish-helper sections remain intact.
- The optional `evaluation-report` guardrail build step skips cleanly when any prerequisite JSON is missing.
- `make validate-workflow-file-health` can still fail against GitHub default-branch visibility when a workflow exists on the current branch but is not yet present on the default branch. For real artifact generation in this validation pass, workflow file health was regenerated with:

```bash
python3 scripts/ci/check_workflow_file_issues.py \
  --glob ".github/workflows/*.yml" \
  --ref "HEAD" \
  --mode yaml \
  --summary-json-out reports/ci/workflow_file_health_summary.json
```
