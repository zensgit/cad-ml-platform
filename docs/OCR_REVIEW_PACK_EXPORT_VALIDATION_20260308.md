# OCR Review Pack Export Validation 20260308

## Goal
- Add a low-conflict batch export script for OCR/drawing review work.
- Reuse `review_hints`, `field_coverage`, and `engineering_signals` without modifying OCR/drawing inference logic.

## Design
- Added `scripts/export_ocr_review_pack.py`.
- Input supports:
  - a single `.json` response file
  - a single `.jsonl` batch file
  - a directory containing `.json` / `.jsonl` responses
- Export outputs:
  - flattened CSV rows for spreadsheet-style manual review
  - structured JSON with nested OCR/drawing review data plus aggregate summary
- Default behavior exports only rows where `review_hints.review_recommended=true`.
- Optional flags:
  - `--include-ready` to keep automation-ready rows in the pack
  - `--top-k` to cap the ranked output size
- Ranking uses:
  - `review_priority`
  - `readiness_score`
  - missing critical fields
  - coverage ratio

## Files
- `scripts/export_ocr_review_pack.py`
- `tests/unit/test_export_ocr_review_pack.py`

## Validation
```bash
python3 -m py_compile \
  scripts/export_ocr_review_pack.py \
  tests/unit/test_export_ocr_review_pack.py

flake8 \
  scripts/export_ocr_review_pack.py \
  tests/unit/test_export_ocr_review_pack.py \
  --max-line-length=100

pytest -q tests/unit/test_export_ocr_review_pack.py
```

## Example
```bash
python3 scripts/export_ocr_review_pack.py \
  --input reports/ocr_batches \
  --output-csv reports/ocr_review_pack.csv \
  --output-json reports/ocr_review_pack.json
```

## Result
- OCR/drawing responses can now be turned into a ranked review pack without touching `src/api/v1/analyze.py` or `src/ml/hybrid_classifier.py`.
- CSV is optimized for bulk human review; JSON preserves nested review payloads for downstream automation.
