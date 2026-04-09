# Evaluation Report Guardrail Detail Summary Validation

## Scope

Improve `comment_evaluation_report_pr.js` so `evaluation-report` PR comments keep top-level
guardrail summaries short when healthy, but append child-section details when the guardrail state
is degraded.

Covered summaries:

- `Workflow Guardrail Summary`
- `CI Workflow Guardrail Overview`

## Files

- `scripts/ci/comment_evaluation_report_pr.js`
- `tests/unit/test_comment_evaluation_report_pr_js.py`

## Change

- `summarizeWorkflowGuardrail(...)` now appends non-`ok` child section details from:
  - `workflow_file_health`
  - `workflow_inventory`
  - `workflow_publish_helper`
- `summarizeCiWorkflowGuardrailOverview(...)` now appends non-`ok` child section details from:
  - `ci_watch`
  - `workflow_guardrail`

Example output shape:

```text
status=error, workflow_health=ok, inventory=error, publish_helper=ok;
workflow_inventory=error:workflows=33, duplicate=1, missing_required=0, non_unique_required=0
```

## Validation

Commands run:

```bash
pytest -q tests/unit/test_comment_evaluation_report_pr_js.py
node --check scripts/ci/comment_evaluation_report_pr.js
make validate-eval-with-history-ci-workflows
```

Observed results:

- `pytest -q tests/unit/test_comment_evaluation_report_pr_js.py` -> passed
- `node --check scripts/ci/comment_evaluation_report_pr.js` -> passed
- `make validate-eval-with-history-ci-workflows` -> passed

## Outcome

When workflow guardrails are red or yellow, the PR comment now exposes the failing child summary
directly, reducing the need to open artifact JSON just to understand the first broken layer.
