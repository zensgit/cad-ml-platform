# DEV Experiment Archive Workflows (2026-02-20)

## Scope
- Add scheduled archive dry-run workflow with artifact output.
- Add manual archive apply workflow with explicit confirmation and approval gate.

## Added Workflows

1. `.github/workflows/experiment-archive-dry-run.yml`
- Trigger:
  - `schedule`: daily at `02:30 UTC`
  - `workflow_dispatch`
- Behavior:
  - Runs `scripts/ci/archive_experiment_dirs.py --dry-run`
  - Emits run summary
  - Uploads manifest/log artifacts

2. `.github/workflows/experiment-archive-apply.yml`
- Trigger:
  - `workflow_dispatch`
- Safety:
  - Requires `approval_phrase == I_UNDERSTAND_DELETE_SOURCE`
  - Uses environment `experiment-archive-approval`
  - Supports `require_exists` and optional explicit `dirs_csv`
- Behavior:
  - Runs archive script with `--delete-source`
  - Uploads manifest/log/tar artifacts

## Related Docs Update
- `README.md` section `实验目录归档自动化` now includes:
  - Make/script usage
  - Workflow names and behaviors
  - Environment approval recommendation

## Verification
1. Workflow YAML structure parse:
   - `python3 ... yaml.BaseLoader` check passed (`on/jobs` sections present).
2. Regression tests:
   - `pytest -q tests/unit/test_archive_experiment_dirs.py -q` passed.

## Notes
- `PyYAML` `safe_load` interprets top-level key `on` as boolean `True` (YAML 1.1 behavior), so structure checks use `BaseLoader`.
