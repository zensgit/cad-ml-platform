# DEV Archive Experiments Make/README Integration (2026-02-20)

## Scope
- Wire experiment-archive automation into `Makefile`.
- Add usage guide into `README.md`.

## Changes
- `Makefile`
  - Added new target: `archive-experiments` (default dry-run).
  - Added configurable vars:
    - `ARCHIVE_EXPERIMENTS_ROOT`
    - `ARCHIVE_EXPERIMENTS_OUT`
    - `ARCHIVE_EXPERIMENTS_KEEP_DAYS`
    - `ARCHIVE_EXPERIMENTS_TODAY`
    - `ARCHIVE_EXPERIMENTS_MANIFEST`
    - `ARCHIVE_EXPERIMENTS_EXTRA_ARGS`
  - Added target into `.PHONY`.
- `README.md`
  - Added section `实验目录归档自动化`.
  - Included both `make` usage and direct script usage examples.

## Verification
1. Unit tests:
   - `pytest -q tests/unit/test_archive_experiment_dirs.py -q` passed.
2. Command-chain verification:
   - Executed `make archive-experiments` with temp experiment dirs.
   - Confirmed dry-run manifest output was generated and selection logic was correct.

## Notes
- `archive-experiments` defaults to `--dry-run` to avoid accidental data deletion.
- To apply archive + cleanup, pass:
  - `ARCHIVE_EXPERIMENTS_EXTRA_ARGS="--delete-source"`
