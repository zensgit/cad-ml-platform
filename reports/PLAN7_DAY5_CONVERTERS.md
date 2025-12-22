# Day 5 - Multi-tool Compatibility

Date: 2025-12-21

## Scope

- Document multiple DWG -> DXF conversion strategies.
- Verify auto-selection prefers ODA when available.

## Changes

- Added converter guide:
  - `cad-ml-platform/docs/DWG_CONVERTERS.md`
- Production doc links converter guide:
  - `cad-ml-platform/docs/CAD_RENDER_PRODUCTION.md`

## Verification

- Command:
  - `PYTHONPATH=... python3 -c 'resolve_dwg_converter/resolve_oda_exe_from_env'`
- Result:
  - `DWG_CONVERTER=auto`
  - ODA detected at `/Applications/ODAFileConverter.app/Contents/MacOS/ODAFileConverter`
