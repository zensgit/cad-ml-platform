# DEV_ISO286_DEVIATIONS_PDF_REVALIDATION_20260210

## Summary
- Re-ran the GB/T 1800.2-2020 (ISO 286 equivalent) hole+shaft limit deviation table extraction against the user-provided PDF to confirm the extractor still works and the current `iso286_deviations.json` remains consistent.

## Commands & Results
1) Extract deviation tables (holes + shafts) from PDF (non-tracked output)
- Command:
  - `python3 scripts/extract_iso286_deviations.py --pdf "<local GB/T_1800.2-2020 PDF>" --out /tmp/iso286_deviations_extracted_20260210.json`
- Result:
  - Extracted labels: holes=`203`, shafts=`129`
  - Extractor warnings: `4` (rows dropped due to sign-consistency filtering)

2) Validate extracted JSON structure + spot-check lookups
- Command:
  - `python3 scripts/validate_iso286_deviations.py --json /tmp/iso286_deviations_extracted_20260210.json --spot-check`
- Result:
  - `OK`

3) Validate repository JSON structure + spot-check lookups
- Command:
  - `python3 scripts/validate_iso286_deviations.py --spot-check`
- Result:
  - `ISO286 deviations: holes=203 shafts=129 path=data/knowledge/iso286_deviations.json`
  - `OK`

4) Compare extracted tables to repository data
- Result:
  - `tables_equal=True` for both `holes` and `shafts`

## Output Artifacts
- `/tmp/iso286_deviations_extracted_20260210.json` (local-only; not committed)
- `data/knowledge/iso286_deviations.json` (repository source of truth)

