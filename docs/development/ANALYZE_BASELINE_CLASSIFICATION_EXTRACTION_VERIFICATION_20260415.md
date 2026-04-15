# Analyze Baseline Classification Extraction Verification

Date: 2026-04-15
Owner: Codex
Scope: Verification for baseline classification extraction from `analyze.py`

## Files Changed

- `src/core/classification/baseline_policy.py`
- `src/core/classification/__init__.py`
- `src/api/v1/analyze.py`
- `tests/unit/test_classification_baseline_policy.py`

## Verification Plan

1. Run the new unit tests for baseline policy
2. Re-run existing fusion integration tests
3. Re-run the existing Phase 3 classification helper tests
4. Run lightweight syntax/style validation on touched files
5. Reconfirm baseline parity for blank `entity.kind` aggregation
6. Add direct coverage for `build_baseline_classification_context()`

## Expected Behavior

- L3 fusion still wins when 3D features are present and fusion succeeds
- L3 fusion still falls back to L1 when fusion errors
- L2 fusion still upgrades 2D baseline predictions when it produces a valid label
- low-value or unknown L2 fusion outputs still keep the L1 result
- downstream Hybrid/Fusion override and finalization logic remain unchanged
- downstream context values used by `analyze.py` are explicitly asserted:
  `text_signals`, `entity_counts`, `doc_metadata`, `l2_features`, `l3_features`
- blank entity kinds keep baseline aggregation parity with the pre-extraction route logic

## Verification Results

Commands run:

```bash
.venv311/bin/python -m pytest -q \
  tests/unit/test_classification_baseline_policy.py \
  tests/unit/test_classification_override_policy.py \
  tests/unit/test_classification_finalization.py \
  tests/unit/test_classification_decision_contract.py

.venv311/bin/python -m pytest -q \
  tests/integration/test_analyze_dxf_fusion.py \
  tests/integration/test_analyze_json_fusion.py

.venv311/bin/flake8 \
  src/core/classification/baseline_policy.py \
  tests/unit/test_classification_baseline_policy.py

python3 -m py_compile \
  src/core/classification/baseline_policy.py \
  tests/unit/test_classification_baseline_policy.py
```

Observed results:

- `15 passed, 7 warnings`
- `10 passed, 7 warnings`
- `flake8` passed
- `py_compile` passed

Key hardening completed after initial extraction:

- added direct unit coverage for `build_baseline_classification_context()`
- verified downstream signal fields used by `analyze.py`
- restored `entity_counts` parity for blank `entity.kind` values to avoid silent behavior drift
