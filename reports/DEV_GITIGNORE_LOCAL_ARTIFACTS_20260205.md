# DEV_GITIGNORE_LOCAL_ARTIFACTS_20260205

## Summary
Extended `.gitignore` to keep local experiment artifacts (model weights,
claudedocs samples, and ad-hoc scripts) out of the main repository history.

## Changes
- Ignored `models/*.pt` weight files.
- Ignored local experiment outputs in `claudedocs/` and ad-hoc scripts under
  `scripts/`.

## Notes
- If any of these artifacts should be tracked, remove the specific ignore
  patterns and consider Git LFS for large weights.
