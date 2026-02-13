# DEV_CI_MYPY_UNUSED_IGNORE_FIX_20260213

## Summary

Fixed a GitHub Actions `mypy` failure caused by an unused `# type: ignore[...]` comment.

## Root Cause

- CI runs `make type-check` which executes `mypy src`.
- `src/utils/dxf_io.py` had `for block in doc.blocks:  # type: ignore[attr-defined]`.
- Because `doc` is typed as `Any`, mypy does not require (or honor) the ignore, and flags it as:
  `Unused "type: ignore" comment [unused-ignore]`.

## Change

- `src/utils/dxf_io.py`
  - Removed the unused `# type: ignore[attr-defined]` comment.

## Validation

Executed locally:

```bash
make type-check
.venv/bin/python -m pytest tests/unit/test_dxf_io.py -q
```

