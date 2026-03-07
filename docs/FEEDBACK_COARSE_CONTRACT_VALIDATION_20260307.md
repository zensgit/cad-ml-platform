# Feedback Coarse Contract Validation (2026-03-07)

## Goal

Make `/api/v1/feedback/` backward-compatible while adding stable fine/coarse/source/review fields so human corrections can flow back into active-learning and later training jobs.

## Changes

- Added additive request fields in `src/api/v1/feedback.py`:
  - `corrected_fine_part_type`
  - `corrected_coarse_part_type`
  - `original_part_type`
  - `original_fine_part_type`
  - `original_coarse_part_type`
  - `original_decision_source`
  - `review_outcome`
  - `review_reasons`
- Added normalization helpers to derive:
  - `corrected_fine_part_type`
  - `corrected_coarse_part_type`
  - `corrected_is_coarse_label`
  - `original_fine_part_type`
  - `original_coarse_part_type`
  - `original_is_coarse_label`
- Preserved legacy `corrected_part_type` in stored JSONL entries.
- Sanitized `review_reasons` and wrote JSONL using UTF-8 encoding.

## Validation Commands

```bash
python3 -m py_compile src/api/v1/feedback.py tests/test_feedback.py
flake8 src/api/v1/feedback.py tests/test_feedback.py --max-line-length=100
pytest -q tests/test_feedback.py
```

## Expected Result

- Legacy payloads remain accepted.
- Stored log entry contains both legacy and normalized coarse/fine fields.
- Explicit fine/coarse payloads are preserved.
- Review metadata is sanitized and persisted.
