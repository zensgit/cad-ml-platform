# DEV_UNTRACKED_ARTIFACTS_20260204

## Summary
Collected current untracked files to avoid accidental deletion or commit.

## Untracked files (by category)

### Docs
- `docs/TOLERANCE_KNOWLEDGE_ROADMAP.md` (new roadmap doc; should be tracked)

### Models (likely experimental)
- `models/cad_classifier_v11.pt`
- `models/cad_classifier_v12.pt`
- `models/cad_classifier_v13.pt`
- `models/cad_classifier_v14.pt`
- `models/cad_classifier_v14_ensemble.pt`

### Scripts (training/analysis)
- `scripts/analyze_v13_errors.py`
- `scripts/analyze_v14_errors.py`
- `scripts/finetune_v11_on_real.py`
- `scripts/synthesize_dxf_v2.py`
- `scripts/train_classifier_v11.py`
- `scripts/train_classifier_v13.py`
- `scripts/train_classifier_v14.py`
- `scripts/train_classifier_v15.py`

### Misc
- `claudedocs/v14_errors.json`

## Recommendations
- Track `docs/TOLERANCE_KNOWLEDGE_ROADMAP.md` with the roadmap updates.
- For model checkpoints and experimental scripts:
  - If they are for reference only, add patterns to `.gitignore`.
  - If they are intended for sharing or reproducibility, commit with a clear naming/versioning note.
