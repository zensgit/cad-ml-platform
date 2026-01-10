# Vision CAD Feature Compare Report Missing Baseline Tests Design

## Overview
Extend compare report coverage to confirm the report renders a friendly message
when baseline data is missing.

## Tests
- `tests/unit/test_vision_cad_feature_compare_report.py`
  - Missing baseline combo renders "Baseline entry missing".

## Validation
- `pytest tests/unit/test_vision_cad_feature_compare_report.py -v`
