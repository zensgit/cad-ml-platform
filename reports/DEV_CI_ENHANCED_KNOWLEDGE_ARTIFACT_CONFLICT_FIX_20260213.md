# DEV_CI_ENHANCED_KNOWLEDGE_ARTIFACT_CONFLICT_FIX_20260213

## Problem
GitHub Actions workflow `CI Enhanced` (unit-tests matrix) was intermittently failing even when tests passed.

Observed failure:
- `actions/upload-artifact@v4` returned `(409) Conflict: an artifact with this name already exists on the workflow run`.

Root cause:
- The unit-tests job uploads the knowledge report artifact with a name that only included `python-version`:
  - `knowledge-test-report-${{ matrix.python-version }}`
- With matrix shards `[1..4]`, multiple jobs attempted to upload an artifact with the same name in the same workflow run.

## Fix
Made the artifact name unique per matrix entry by including `shard`:
- Updated `.github/workflows/ci-enhanced.yml`:
  - `knowledge-test-report-${{ matrix.python-version }}-${{ matrix.shard }}`

## Validation
- Local: no runtime code changes; this is a workflow-only change.
- Expected CI outcome:
  - `CI Enhanced` unit-test shards should no longer fail due to artifact upload conflicts.

