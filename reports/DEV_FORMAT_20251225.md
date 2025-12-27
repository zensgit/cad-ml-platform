# DEV_FORMAT_20251225

## Scope

- Run code formatter (black + isort).

## Changes

- Imports reordered by isort in:
  - `src/adapters/factory.py`
  - `src/core/vision/__init__.py`

## Validation

- Command: `make format`
- Result: completed (black unchanged on 535 files; isort updated 2 files)

## Notes

- `make` printed warnings about duplicate `security-audit` target (pre-existing).
