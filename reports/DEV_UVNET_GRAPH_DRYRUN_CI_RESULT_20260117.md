# DEV_UVNET_GRAPH_DRYRUN_CI_RESULT_20260117

## Summary
Ran the UV-Net graph dry-run workflow after fixing the micromamba version; the
workflow completed but pythonocc-core import failed due to a TopExp import error.

## Steps
- Observed run: `gh run view 21096593368 --log`.

## Results
- micromamba installed pythonocc-core 7.9.0 successfully.
- Import failed with: `cannot import name 'TopExp' from 'OCC.Core.TopExp'`.
- Dry-run skipped as `HAS_OCC` remained false.

## Fix
- Updated imports to use `from OCC.Core import TopExp` and keep
  `TopExp_Explorer` from `OCC.Core.TopExp`.

## Notes
- Re-run the workflow to confirm the import fix resolves the dry-run.
