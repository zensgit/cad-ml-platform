# MECH_4000_DWG_REVIEW_SHEET_INSTRUCTIONS_20260120

## Purpose
Manual verification of DXF drawing metadata (title block + labels) for the
4000CAD conversion set.

## Files
- Review CSV: `reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_20260120.csv`
- Preview images (20 samples): `reports/experiments/20260120/MECH_4000_DWG_REVIEW_PREVIEWS_20260120`

## Columns to Review/Fill
- `reviewer_label_cn`: final Chinese label for the drawing
- `reviewer_label_en`: optional English label
- `review_status`: `pending` / `confirmed` / `needs_followup`
- `review_notes`: free-form notes
- Title block overrides (fill when auto-extraction is missing/incorrect):
  - `review_drawing_number`, `review_part_name`, `review_material`,
    `review_scale`, `review_sheet`, `review_date`, `review_weight`,
    `review_company`, `review_projection`, `review_revision`

## Tips
- `text_sample` and `normalized_text_sample` are best-effort from DXF TEXT/MTEXT.
  Some CAD exports store glyphs as `\M+` codes; use the preview PNG when text
  is unreadable in the CSV.
- If a field is already correct under `title_block_*`, leave the `review_*`
  column empty.
