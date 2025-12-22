#!/usr/bin/env markdown
# Week 9 Report: Manufacturing Decision Output

## Summary
- Added unified manufacturing decision summary to analysis output.

## Changes
- `src/api/v1/analyze.py`: build `manufacturing_decision` from quality/process/cost
- Doc: `docs/MANUFACTURING_DECISION_OUTPUT.md`
- Linked in `README.md`

## Tests
- `python3 -m pytest tests/test_l4_cost.py -q`

## Verification
- Run analysis with DFM + process + cost enabled and check:
  - `results.manufacturing_decision` present
