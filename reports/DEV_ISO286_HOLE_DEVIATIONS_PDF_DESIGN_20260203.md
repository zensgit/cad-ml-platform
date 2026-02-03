# DEV_ISO286_HOLE_DEVIATIONS_PDF_DESIGN_20260203

## Summary
Imported ISO 286 hole deviation data for A/B/C, CD/D/E, EF/F, FG/G, H/JS/J/K/M/N/P/R, and the interference series (S/T/U/V/X/Y/Z/ZA/ZB/ZC) directly from GB/T 1800.2-2020 PDF (Tables 2–16) to remove reliance on inferred/absent values.

## Scope
- Extract Tables 2–5 (A/B/C, CD/D/E, EF/F, FG/G), Tables 6–11 (H/JS/J/K/M/N/P/R), and Tables 12–16 (S/T/U/V/X/Y/Z/ZA/ZB/ZC) lower deviations (EI) from the GB/T 1800.2-2020 PDF.
- Merge extracted values into `data/knowledge/iso286_hole_deviations.json`.
- Preserve existing deviations for other symbols.

## Design Decisions
### 1) PDF extraction helper
- **Decision**: Add `scripts/extract_iso286_hole_deviations_from_pdf.py` targeting Tables 2–16.
- **Why**: These tables are structured and provide stable EI values across grades; automation reduces transcription risk.
- **Safeguard**: Use start/stop markers (`表 2`…`表 16`) to avoid mixing tables when they share pages.

### 2) pdfplumber extraction for multi-line cells
- **Decision**: Use `pdfplumber.extract_table()` for Tables 6–16 where cells contain stacked ES/EI values and columns shift by size range.
- **Why**: pdfplumber returns cell-wise values (e.g., `-14\\n-20`) that are easier to interpret for EI than raw text line parsing.

### 3) Data merge strategy
- **Decision**: Merge Tables 2–16 symbols into the existing JSON without overwriting other symbol tables.
- **Why**: This yields a single authoritative GB/T-backed override table while keeping any missing entries intact.

### 4) Metadata update
- **Decision**: Update `source`/`notes` to cite GB/T 1800.2-2020 Tables 2–16.
- **Why**: Keep provenance clear for later audits.

### 5) Tooling dependency
- **Decision**: Add `pdfplumber` to `requirements-dev.txt` and `requirements-dev-lite.txt`.
- **Why**: Ensures PDF table extraction is reproducible in dev environments.

### 6) Fail-fast on missing pdfplumber
- **Decision**: Require `pdfplumber` for Tables 6–16 by default; add `--allow-partial` to skip with a warning.
- **Why**: Prevents silently writing incomplete overrides when table extraction is unavailable.

### 7) Preferred grade tracking
- **Decision**: Persist `preferred_grade` in the JSON output to document which IT grade column was selected for EI extraction.
- **Why**: Provides traceability when changing grade selection (e.g., `--prefer-grade 7`).

### 8) Report & compare outputs
- **Decision**: Add `--report` (CSV dump of symbol/size/ei) and `--compare-grade`/`--compare-report` for EI diffing.
- **Why**: Enables quick human review and grade-to-grade sensitivity checks without manual spreadsheet work.

## Files Updated
- `scripts/extract_iso286_hole_deviations_from_pdf.py`
- `data/knowledge/iso286_hole_deviations.json`
