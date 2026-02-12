# DEV_ISO286_HOLE_DEVIATIONS_PDF_REVALIDATION_20260210

## Summary
- Re-ran the GB/T 1800.2-2020 (ISO 286 equivalent) hole EI extraction against the user-provided PDF to confirm the extractor still works and the current `iso286_hole_deviations.json` remains consistent.

## Commands & Results
1) Extract Tables 2–16 (holes EI) from PDF
- Command:
  - `python3 scripts/extract_iso286_hole_deviations_from_pdf.py --pdf "<local GB/T_1800.2-2020 PDF>" --output data/knowledge/iso286_hole_deviations.json --prefer-grade 6 --report reports/experiments/20260210/iso286_hole_deviations_extract/extracted_ei_prefer_grade6.csv`
- Result:
  - Extracted symbols: `A, B, C, CD, D, E, EF, F, FG, G, H, J, JS, K, M, N, P, R, S, T, U, V, X, Y, Z, ZA, ZB, ZC`
  - No tracked changes were produced (the extracted values matched the current repository data).

2) Validate required ISO286 hole symbol coverage
- Command:
  - `python3 scripts/validate_iso286_hole_deviations.py`
- Result:
  - `All required hole symbols present: A, B, C, D, N, P`

3) Regression: tolerance knowledge unit tests
- Command:
  - `pytest -q tests/unit/knowledge/test_tolerance.py tests/unit/knowledge/test_iso286_hole_deviations_coverage.py tests/unit/knowledge/test_iso286_pdf_extract.py`
- Result:
  - pass (27 tests)

## Output Artifacts
- `reports/experiments/20260210/iso286_hole_deviations_extract/extracted_ei_prefer_grade6.csv`
- `data/knowledge/iso286_hole_deviations.json`

