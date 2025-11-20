# Quality Baseline (Metrics & Lint)

Date: 2025-11-20

## Metrics Parity Added
- `ocr_input_rejected_total{reason}`
- Extended `ocr_errors_total{provider,code,stage}` coverage (validation, provider_down, rate_limit, circuit_open, internal, endpoint)

## Lint Snapshot (Pre-Cleanup Phase 2)
Collected via `flake8 src` (max-line-length=100):

| Category | Count | Notes |
|----------|-------|-------|
| F401 (unused imports) | 9 | assembly, ocr providers, vision api |
| F841 (unused variable) | 1 | `src/api/v1/ocr.py` local `e` removed in next phase |
| E501 (line too long) | 20+ | deferred (config allows 100; some >100) |
| F821 (undefined name) | 3 | `metrics_monitor.py` Tuple/List annotations missing imports |
| E114/E116 (indent) | 2 | comment indentation in `ocr/manager.py` |
| E722 (bare except) | 1 | `metrics_monitor.py` |
| W391/W292 (blank line / newline) | 5 | several modules end-of-file issues |
| E301 (expected blank line) | 1 | `confidence_calibrator.py` |
| E741 (ambiguous name) | 1 | `bbox_mapper.py` variable `l` |

## Planned Next Cleanup
1. Remove unused imports (F401) across OCR providers & vision api.
2. Fix undefined names in `metrics_monitor.py` (add typing imports or remove code paths).
3. Replace ambiguous variable name `l` in `bbox_mapper.py`.
4. Normalize indentation comment in `ocr/manager.py`.
5. Add trailing newlines & remove stray blank lines.
6. Gradually wrap >120 char lines; keep semantic grouping.

## Test Coverage Reference
All tests passing: 139 tests (vision + ocr error paths + metrics assertions). Warning: one script collection warning (`TestRunner` in `scripts/test_eval_system.py`).

## Notes
Large-scale formatting deferred to avoid churn before functional parity & metrics validation are fully adopted by clients.

