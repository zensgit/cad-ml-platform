# DEV_BRANCH_SYNC_TOLERANCE_KNOWLEDGE_AND_STANDARDS_LIBRARY_20260212

## Goal
Keep long-running feature branches up to date with `main` and confirm the fast validation
gate remains green after sync.

## Actions
### 1) Sync `feat/tolerance-knowledge`
- Merged `main` into `feat/tolerance-knowledge`.
- Resolved add/add conflicts in `src/core/knowledge/tolerance/` by preferring the `main`
  implementation (ISO286 deviation tables + helper APIs).
- Pushed the merge commit to `origin/feat/tolerance-knowledge`.

### 2) Create baseline `feat/standards-library`
- Created `feat/standards-library` from `main` (baseline branch).
- Pushed to `origin/feat/standards-library`.

## Validation
- `feat/tolerance-knowledge`: `make validate-core-fast` (passed)
- `feat/standards-library`: `make validate-core-fast` (passed)

## Notes
- The provider framework (`src/core/providers/`) and canonical local tiered runner
  (`scripts/test_with_local_api.sh`) are both first-class, referenced by `Makefile` and docs.
  Avoid introducing a repo-root `test_with_local_api.sh` to prevent drift.

