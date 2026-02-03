# DEV_ISO286_HOLE_DEVIATIONS_PDF_VALIDATION_20260203

## Validation Summary
- Extracted A/B/C, CD/D/E, EF/F, FG/G, H/JS/J/K/M/N/P/R, and S/T/U/V/X/Y/Z/ZA/ZB/ZC hole deviations from GB/T 1800.2-2020 PDF (Tables 2–16).
- Re-validated ISO286 deviation coverage.
- Re-ran tolerance-fit unit tests.
- Confirmed extractor now fails fast when `pdfplumber` is missing (unless `--allow-partial` is used).
- Re-ran knowledge test suites for tolerance, GD&T, and design standards.

## Commands & Results
1) Extract Tables 2–16 from PDF (hardened extractor)
- Command:
  - `python3 scripts/extract_iso286_hole_deviations_from_pdf.py --pdf "/Users/huazhou/Downloads/GB-T 1800.2-2020 产品几何技术规范（GPS） 线性尺寸公差ISO代号体系 第2部分：标准公差带代号和孔、轴的极限偏差表在线预览.pdf" --report reports/DEV_ISO286_EI_REPORT_20260203.csv --compare-grade 7 --compare-report reports/DEV_ISO286_EI_DIFF_6_vs_7_20260203.csv`
- Result: Updated `data/knowledge/iso286_hole_deviations.json` with A/B/C, CD/D/E, EF/F, FG/G, H/JS/J/K/M/N/P/R, and S/T/U/V/X/Y/Z/ZA/ZB/ZC.
  - `preferred_grade` recorded in the JSON output for traceability.
  - Report: `reports/DEV_ISO286_EI_REPORT_20260203.csv`
  - Diff: `reports/DEV_ISO286_EI_DIFF_6_vs_7_20260203.csv`

2) Validate ISO286 symbol coverage
- Command:
  - `python3 scripts/validate_iso286_hole_deviations.py`
- Result: `All required hole symbols present: A, B, C, D, N, P`

3) Tolerance-fit unit tests (includes A/B/C non-H fit coverage and JS hole EI symmetry)
- Command:
  - `pytest tests/test_tolerance_fits.py -v`
- Result: `6 passed` (warning: `python_multipart` PendingDeprecationWarning)

4) Extraction guard test (missing pdfplumber)
- Command:
  - `pytest tests/unit/knowledge/test_iso286_pdf_extract.py -v`
- Result: `1 passed`

5) Knowledge suite regression
- Command:
  - `pytest tests/unit/knowledge/test_tolerance.py tests/unit/knowledge/test_design_standards.py tests/unit/knowledge/test_gdt.py -v`
- Result: `68 passed`

## Output Artifacts
- `data/knowledge/iso286_hole_deviations.json`
- `reports/DEV_ISO286_HOLE_DEVIATIONS_PDF_DESIGN_20260203.md`
