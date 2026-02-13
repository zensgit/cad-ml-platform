# DEV_GRAPH2D_DISTILLATION_TEACHER_LABEL_BUCKET_MAPPING_20260214

## Goal

When training Graph2D with knowledge distillation, the teacher (filename/titleblock/hybrid) may predict a fine-grained label that is not present in the student label-space after label normalization/cleaning.

This change adds a single, shared "fine -> bucket" mapping and applies it inside the distillation teacher before falling back to `"other"` or a uniform prior. Example: teacher predicts `"对接法兰"` while the student label-map contains only `"法兰"`.

## Changes

### Shared Mapping Module

File: `src/ml/label_normalization.py`

- Added `DXF_LABEL_BUCKET_MAP` (fine label -> bucket label)
- Added `normalize_dxf_label(label, default=None)` helper

### Distillation Teacher Uses Bucket Mapping

File: `src/ml/knowledge_distillation.py`

- `TeacherModel.generate_soft_labels(...)` now:
  - normalizes predicted labels into coarse buckets when possible
  - only then falls back to `"other"` (low-confidence) or uniform logits

### Manifest Normalization Reuses the Same Mapping

File: `scripts/normalize_dxf_label_manifest.py`

- Uses `src/ml/label_normalization.DXF_LABEL_BUCKET_MAP` as the single source of truth.

## Validation

### Unit Tests

- `.venv/bin/python -m pytest tests/unit/test_label_normalization.py tests/unit/test_knowledge_distillation_teacher.py -v` (passed)

Covered scenarios:

- `normalize_dxf_label("对接法兰") == "法兰"`
- Distillation teacher maps `"对接法兰"` into `"法兰"` when `"法兰"` exists in `label_to_idx`
- Unknown labels still fall back to `"other"` when available

### Lint

- `.venv/bin/flake8 src/ml/label_normalization.py src/ml/knowledge_distillation.py scripts/normalize_dxf_label_manifest.py tests/unit/test_knowledge_distillation_teacher.py tests/unit/test_label_normalization.py` (passed)

## Notes / Caveats

- The bucket mapping is intentionally conservative and only applies when the predicted fine label is not in the student `label_to_idx`.
- This improves teacher signal alignment for coarse-bucket training runs (e.g. `--normalize-labels` + `--clean-min-count ...`) without changing fine-label training behavior.

