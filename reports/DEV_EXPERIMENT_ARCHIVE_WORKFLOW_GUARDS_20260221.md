# Experiment Archive Workflow Guards (2026-02-21)

## Goal
Add regression coverage to prevent accidental weakening of archive workflow safety controls.

## Changes
- Added `tests/unit/test_experiment_archive_workflows.py`.
- Coverage includes:
  - Dry-run workflow trigger and script guard checks.
  - Apply workflow approval phrase gate checks.
  - Apply workflow environment gate check (`experiment-archive-approval`).
  - Script guard checks for `--delete-source` and `--require-exists`.
  - Artifact upload step presence checks.

## Validation
- Local tests:
  - `pytest -q tests/unit/test_experiment_archive_workflows.py tests/unit/test_archive_experiment_dirs.py`
  - Result: `5 passed`.
- GitHub Actions dry-run validation:
  - Workflow: `Experiment Archive Dry Run`
  - Run: `22248985990`
  - Conclusion: `success`

## Notes
- The apply workflow remains intentionally dispatch-only and approval-gated.
- No destructive apply run was triggered in this validation cycle.
