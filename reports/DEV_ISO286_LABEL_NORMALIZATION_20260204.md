# DEV_ISO286_LABEL_NORMALIZATION_20260204

## Summary
Normalized ISO 286 deviation table labels to handle footnote prefixes, spaces, and mixed-case tokens
from `iso286_deviations.json`. Added tests to assert CD6 and y6 lookups resolve against the
normalized table.

## Changes
- Added ISO 286 label normalization helpers and symbol-kind inference in `src/core/knowledge/tolerance/fits.py`.
- Normalized keys during ISO 286 table parsing and retained the widest row set on collisions.
- Added regression tests for CD6 (holes) and y6 (shafts) normalization in
  `tests/unit/test_tolerance_limit_deviations.py`.

## Tests
- `python3 -m pytest tests/unit/test_tolerance_limit_deviations.py -q`

## Notes
- Normalization is conservative and preserves canonical labels after case/footnote cleanup.
