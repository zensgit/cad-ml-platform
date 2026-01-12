# Drawing Recognition Title Block Design

## Summary
This update expands title block support for drawing recognition by extending the OCR title block schema and adding a dedicated parser for common CAD title fields. The goal is to reliably extract key drawing metadata (number, revision, material, scale, etc.) from both OCR text and provider JSON output.

## Schema Updates
`TitleBlock` now includes the following optional fields:
- `drawing_number`
- `revision`
- `part_name`
- `material`
- `scale`
- `sheet`
- `date`
- `weight`
- `company`
- `projection`

## Parsing Approach
- New parser: `src/core/ocr/parsing/title_block_parser.py`.
- Regex patterns cover English and Chinese labels (e.g., drawing no, rev, material, scale, sheet, date).
- Values are trimmed when a subsequent label is detected to avoid capturing the rest of the line.
- `sheet` values normalize `"1 of 3"` to `"1/3"`.
- `parse_title_block_with_confidence` captures per-line confidence when OCR lines include scores.

## Pattern Expansion
- Added label aliases for common title block variations:
  - Drawing number: `DWG#`, `Drawing ID`, `Part No`, `图纸代号`, `零件号`.
  - Revision: `REV.`, `版本号`.
  - Part name: `DESC`, `Description`.
  - Material: `MATL`, `材质`.
  - Sheet: `SHT`, `Sheet No`.
  - Weight: `WT`, `Mass`.
  - Projection: `First/Third Angle` and Chinese variants.

## Provider Integration
- Fallback parser now calls the shared title block parser for text regex extraction.
- Paddle provider extracts title block fields from OCR lines (confidence-aware), then merges from full text.
- DeepSeek provider merges parsed JSON title block data with the text-based parser for coverage.

## Limitations
- Regex parsing is heuristic and may miss non-standard labels or layouts.
- Per-field confidence is only available when OCR line scores are provided.

## Follow-ups
- Add language-specific label aliases as new samples appear.
- Expand tests with more title block variants and multilingual layouts.
