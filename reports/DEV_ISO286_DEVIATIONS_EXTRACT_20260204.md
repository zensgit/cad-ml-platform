# DEV_ISO286_DEVIATIONS_EXTRACT_20260204

## Summary
- Added a PDF extraction script for ISO 286 / GB/T 1800.2 deviation tables.
- Generated `data/knowledge/iso286_deviations.json` from the provided PDF.

## Changes
- `scripts/extract_iso286_deviations.py`
  - Extracts deviation tables using pdfplumber and emits structured JSON.
- `data/knowledge/iso286_deviations.json`
  - Parsed tables (holes/shafts) with ES/EI (upper/lower) per symbol+grade.

## Validation
- `python3 scripts/extract_iso286_deviations.py --pdf <path> --out data/knowledge/iso286_deviations.json`

## Notes
- Output is best-effort; table headers are preserved in `tables` for manual audit.
