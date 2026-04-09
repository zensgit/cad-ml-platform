# Active Learning Coarse Feedback Validation (2026-03-07)

## Goal

Keep `true_type` backward-compatible while letting active-learning feedback and exported training data carry stable fine/coarse correction labels.

## Changes

- Added additive API fields in `src/api/v1/active_learning.py`:
  - `true_fine_type`
  - `true_coarse_type`
- Extended `ActiveLearningSample` in `src/core/active_learning.py` with:
  - `true_fine_type`
  - `true_coarse_type`
  - `true_is_coarse_label`
- Updated `submit_feedback()` to normalize:
  - legacy `true_type`
  - fine/coarse labels
  - coarse-label boolean
- Updated export payloads to include:
  - `predicted_coarse_type`
  - `true_fine_type`
  - `true_coarse_type`
  - `true_is_coarse_label`

## Validation Commands

```bash
python3 -m py_compile src/api/v1/active_learning.py src/core/active_learning.py \
  tests/test_active_learning_api.py tests/unit/test_active_learning_export_context.py
flake8 src/api/v1/active_learning.py src/core/active_learning.py \
  tests/test_active_learning_api.py tests/unit/test_active_learning_export_context.py \
  --max-line-length=100
pytest -q tests/test_active_learning_api.py tests/unit/test_active_learning_export_context.py
```

## Expected Result

- Existing clients can still send only `true_type`.
- Samples persist normalized `true_fine_type` / `true_coarse_type`.
- Exported training data can be evaluated on both exact and coarse taxonomy.
