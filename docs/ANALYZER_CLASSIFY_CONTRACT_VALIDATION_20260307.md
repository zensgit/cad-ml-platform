# Analyzer Classify Contract Validation - 2026-03-07

## Goal
Add stable fine/coarse/source/review fields to legacy `CADAnalyzer.classify_part()` outputs without breaking existing `type`-based consumers.

## Scope
Updated files:
- `src/core/analyzer.py`
- `tests/unit/test_analyzer_rules.py`

## Contract Additions
All analyzer classification paths now add:
- `fine_type`
- `coarse_type`
- `is_coarse_type`
- `decision_source`
- `review_reasons`

Existing keys remain unchanged:
- `type`
- `confidence`
- `classifier`
- `rule_version`
- `review_reason`
- `top2_category`
- `top2_confidence`

## Validation Commands
```bash
python3 -m py_compile src/core/analyzer.py tests/unit/test_analyzer_rules.py
flake8 src/core/analyzer.py tests/unit/test_analyzer_rules.py --max-line-length=100
pytest -q tests/unit/test_analyzer_rules.py
```

## Results
- `py_compile`: passed
- `flake8`: passed
- `pytest`: `3 passed`

## Notes
- Rule-based outputs now expose the same additive contract used by ML paths.
- V16 outputs preserve top2/review fields and now add normalized coarse labels.
