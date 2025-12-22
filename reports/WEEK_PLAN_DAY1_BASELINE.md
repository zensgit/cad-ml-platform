#!/usr/bin/env markdown
# Week Plan Day 1 - Baseline & Layout Verification

## Scope
- Vector layout/rehydrate unit tests.
- Vector store layout audit (FAISS/Redis snapshot).

## Tests
- Command:
  `.venv/bin/python -m pytest tests/unit/test_feature_vector_layout.py tests/unit/test_feature_rehydrate.py tests/unit/test_feature_slots.py -q`
- Result: `7 passed in 2.71s`

## Audit
- Command:
  `.venv/bin/python scripts/audit_vector_store_layout.py --output reports/WEEK_PLAN_DAY1_BASELINE_AUDIT.md`
- Output:
  `reports/WEEK_PLAN_DAY1_BASELINE_AUDIT.md`

## Notes
- Layout audit report contains backend detection and migration guidance.
