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

Results:

- `pytest -q tests/unit/test_generate_workflow_inventory_report.py tests/unit/test_workflow_file_health_make_target.py` -> `10 passed`
- `pytest -q tests/unit/test_stress_workflow_workflow_file_health.py tests/unit/test_generate_workflow_inventory_report.py tests/unit/test_workflow_file_health_make_target.py` -> `13 passed`
- `make validate-workflow-file-health-tests` -> `17 passed`
- `make validate-workflow-inventory-report` -> `10 passed`
- `make workflow-inventory-report` generated:
  - `reports/ci/workflow_inventory_report.json`
  - `reports/ci/workflow_inventory_report.md`
- current repository snapshot:
  - `workflow_count = 33`
  - `duplicate_name_count = 0`
  - `missing_required_count = 0`
  - `non_unique_required_count = 0`
- `make validate-ci-watchers` also passed with the new inventory step included

## Notes

- The report is read-only and does not fail on duplicates by itself.
- Blocking behavior remains in `check_workflow_identity_invariants.py`.
- The generated outputs default to:
  - `reports/ci/workflow_inventory_report.json`
  - `reports/ci/workflow_inventory_report.md`
