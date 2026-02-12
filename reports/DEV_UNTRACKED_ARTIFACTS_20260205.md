# DEV_UNTRACKED_ARTIFACTS_20260205

## Summary
Captured the current set of untracked artifacts that are likely experiment outputs or local
model assets. None were modified during this review.

## Inventory (untracked)
- `claudedocs/suspicious_samples/` (~296K)
- `claudedocs/v14_errors.json` (~12K)
- `models/cad_classifier_v11.pt` (~2.4M)
- `models/cad_classifier_v12.pt` (~2.4M)
- `models/cad_classifier_v13.pt` (~644K)
- `models/cad_classifier_v14.pt` (~2.0M)
- `models/cad_classifier_v14_ensemble.pt` (~10M)
- `models/cad_classifier_v15_ensemble.pt` (~20M)
- `models/cad_classifier_v2.pt` (~128K)
- `models/cad_classifier_v6.pt` (~492K)
- `scripts/eval_v16_after_fix.py` (~16K)
- `scripts/synthesize_dxf_v2.py` (~28K)

## Recommendations
- If these artifacts are not meant for version control, add patterns to `.gitignore` (or move them
  under `reports/experiments/` and keep a reference in documentation).
- If the model weights are needed in-repo, consider Git LFS for `models/*.pt` to avoid bloating
  the main Git history.
- If `scripts/*.py` are reusable utilities, move them into `scripts/` and commit with README notes;
  otherwise keep them ignored as local experiments.

## Notes
- This report is informational only; no files were added, removed, or modified.
