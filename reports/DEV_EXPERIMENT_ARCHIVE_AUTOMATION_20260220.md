# DEV Experiment Archive Automation (2026-02-20)

## Scope
1. Archive and clean local untracked experiment folders:
   - `reports/experiments/20260217`
   - `reports/experiments/20260219`
2. Add reusable automation for future cleanup.

## Local Cleanup Executed
- Archived to:
  - `/Users/huazhou/Downloads/cad-ml-platform-experiment-archives/20260217_20260220_215829.tar.gz`
  - `/Users/huazhou/Downloads/cad-ml-platform-experiment-archives/20260219_20260220_215829.tar.gz`
- Removed original directories from `reports/experiments/`.
- `git status --short` returned clean after cleanup.

## New Automation
- Added script: `scripts/ci/archive_experiment_dirs.py`
  - Supports explicit directory selection (`--dir` repeatable)
  - Supports age-based selection (`--keep-latest-days`, with `--today` override)
  - Supports `--dry-run`
  - Supports source deletion after archive (`--delete-source`)
  - Emits JSON manifest (`--manifest-json`)
  - Optional strict mode (`--require-exists`)

## Tests Added
- `tests/unit/test_archive_experiment_dirs.py`
  - `test_archive_experiment_dirs_explicit_and_delete`
  - `test_archive_experiment_dirs_dry_run_by_age`
  - `test_archive_experiment_dirs_require_exists_fails`

## Verification
- `pytest -q tests/unit/test_archive_experiment_dirs.py -q` passed.
- `python3 -m py_compile scripts/ci/archive_experiment_dirs.py` passed.

## Example Usage
```bash
python3 scripts/ci/archive_experiment_dirs.py \
  --experiments-root reports/experiments \
  --archive-root /Users/huazhou/Downloads/cad-ml-platform-experiment-archives \
  --keep-latest-days 7 \
  --delete-source \
  --manifest-json reports/DEV_EXPERIMENT_ARCHIVE_AUTOMATION_MANIFEST.json
```
