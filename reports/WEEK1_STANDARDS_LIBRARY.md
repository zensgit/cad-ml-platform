#!/usr/bin/env markdown
# Week 1 Report: Standards Library

## Summary
- Added a deterministic standards library manifest and documentation.

## Changes
- Generator now writes `data/standards_dxf/MANIFEST.json`.
- Added `docs/STANDARDS_LIBRARY.md` and linked in `README.md`.
- `scripts/learn_from_standards.py` now adds repo root to `sys.path`.

## Tests
- `python3 scripts/generate_dxf_standards.py`

## Verification
- Verified output: `data/standards_dxf/MANIFEST.json` generated.
- `python3 scripts/learn_from_standards.py` completed (0 new rules added).
