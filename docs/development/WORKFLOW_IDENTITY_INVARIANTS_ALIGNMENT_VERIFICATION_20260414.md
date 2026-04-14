# Workflow Identity Invariants Alignment Verification

Date: 2026-04-14
Owner: Codex
Branch: `submit/local-main-20260414`

## Change Summary

Updated the workflow identity checker so its expected contracts match the current workflow YAMLs:

- `scripts/ci/check_workflow_identity_invariants.py`
  - `evaluation-report.yml` no longer requires legacy `workflow_dispatch` inputs
  - `hybrid-superpass-e2e.yml` now requires the current eight dispatch inputs
- `tests/unit/test_check_workflow_identity_invariants.py`
  - Updated the CI-watch CSV fixture to include `Governance Gates`

No workflow YAML files were changed.

## Verification Commands

```bash
.venv311/bin/python scripts/ci/check_workflow_identity_invariants.py
.venv311/bin/python -m pytest -q tests/unit/test_check_workflow_identity_invariants.py
.venv311/bin/python scripts/ci/generate_workflow_inventory_report.py --output-json /tmp/workflow_inventory_report.json --output-md /tmp/workflow_inventory_report.md
claude -p "Read scripts/ci/check_workflow_identity_invariants.py and tests/unit/test_check_workflow_identity_invariants.py in the current repo. Give a concise 3-bullet review of whether the identity spec now matches the real workflows, focusing on evaluation-report.yml and hybrid-superpass-e2e.yml. No code changes."
```

## Verification Results

### 1. Identity checker

Result: passed

Key output:

- `evaluation-report.yml - ok`
- `hybrid-superpass-e2e.yml - ok`
- `CI_WATCH_REQUIRED_WORKFLOWS - ok`
- `ok: workflow identity check passed for 12 check(s)`

### 2. Unit tests

Result: passed

Output summary:

- `7 passed, 7 warnings`

Warnings were existing `pyparsing` deprecation warnings from `ezdxf` test imports and are unrelated to this change.

### 3. Workflow inventory audit

Result: passed

Key output:

- `required_count: 11`
- `missing_required_count: 0`
- `non_unique_required_count: 0`
- `Evaluation Report: status=ok files=evaluation-report.yml`
- `Governance Gates: status=ok files=governance-gates.yml`

### 4. Claude Code CLI sidecar review

Result: passed

Observed review summary:

- `evaluation-report.yml` spec now correctly matches an empty `workflow_dispatch`
- `hybrid-superpass-e2e.yml` spec matches the real eight-input dispatch contract
- Test coverage is appropriate for name drift, missing input drift, YAML twin drift, and CI-watch membership

## Conclusion

`workflow identity` validation is now aligned with the actual repository workflows.

Historical failures against:

- `evaluation-report.yml`
- `hybrid-superpass-e2e.yml`

have been resolved without modifying the workflow YAML definitions themselves.
