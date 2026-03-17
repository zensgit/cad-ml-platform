# Workflow Inventory Audit Validation

Date: 2026-03-17

## Goal

Add a read-only workflow inventory audit report so the repository can render:

- all `.github/workflows/*.yml` files
- current workflow display names
- duplicate workflow names
- watcher-required workflow name mapping

This complements the blocking invariant checks with an operator-friendly inventory view.

## Added

- `scripts/ci/generate_workflow_inventory_report.py`
- `tests/unit/test_generate_workflow_inventory_report.py`
- `Makefile` targets:
  - `workflow-inventory-report`
  - `validate-workflow-inventory-report`

`validate-ci-watchers` now also invokes `validate-workflow-inventory-report`.

GitHub Actions integration:

- `stress-tests.yml` now triggers on `scripts/ci/generate_workflow_inventory_report.py`
- the `workflow-file-health` job now:
  - generates the inventory report
  - uploads `workflow_inventory_report.json` and `workflow_inventory_report.md`
  - appends the rendered Markdown to `GITHUB_STEP_SUMMARY`
- `evaluation-report.yml` now mirrors a compact inventory summary into:
  - PR comment input env via `WORKFLOW_INVENTORY_REPORT_JSON_FOR_COMMENT`
  - evaluation run step summary via `workflow_inventory_for_comment.md`

## Validation

```bash
pytest -q \
  tests/unit/test_generate_workflow_inventory_report.py \
  tests/unit/test_workflow_file_health_make_target.py
```

```bash
make validate-workflow-inventory-report
```

```bash
make workflow-inventory-report
```

```bash
pytest -q tests/unit/test_comment_evaluation_report_pr_js.py
```

Results:

- `pytest -q tests/unit/test_generate_workflow_inventory_report.py tests/unit/test_workflow_file_health_make_target.py` -> `10 passed`
- `pytest -q tests/unit/test_stress_workflow_workflow_file_health.py tests/unit/test_generate_workflow_inventory_report.py tests/unit/test_workflow_file_health_make_target.py` -> `13 passed`
- `pytest -q tests/unit/test_comment_evaluation_report_pr_js.py` -> `6 passed`
- `make validate-workflow-file-health-tests` -> `17 passed`
- `make validate-workflow-inventory-report` -> `11 passed`
- `make workflow-inventory-report` generated:
  - `reports/ci/workflow_inventory_report.json`
  - `reports/ci/workflow_inventory_report.md`
- current repository snapshot:
  - `workflow_count = 33`
  - `duplicate_name_count = 0`
  - `missing_required_count = 0`
  - `non_unique_required_count = 0`
- `pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py tests/unit/test_generate_workflow_inventory_report.py` -> `9 passed`
- `pytest -q tests/unit/test_graph2d_parallel_make_targets.py` -> `11 passed`
- `make validate-eval-with-history-ci-workflows` -> `25 passed`
- `evaluation-report.yml` now mirrors:
  - `workflow_inventory_for_comment.json`
  - `workflow_inventory_for_comment.md`
  - `WORKFLOW_INVENTORY_REPORT_JSON_FOR_COMMENT` into the PR comment script env
- `comment_evaluation_report_pr.js` now surfaces top workflow inventory names in PR comments when duplicates or missing required mappings exist
- `comment_evaluation_report_pr.js` now also has runtime parse-error coverage for `workflow inventory` / `CI watch` / `workflow file health` comment inputs
- `generate_workflow_inventory_report.py` now renders an `Issue Summary` section so artifact markdown and step summary show top duplicate/missing/non-unique workflow names directly
- `make validate-ci-watchers` also passed with the new inventory step included

## Notes

- The report is read-only and does not fail on duplicates by itself.
- Blocking behavior remains in `check_workflow_identity_invariants.py`.
- The generated outputs default to:
  - `reports/ci/workflow_inventory_report.json`
  - `reports/ci/workflow_inventory_report.md`
