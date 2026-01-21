# DEV_MECH_KNOWLEDGE_4000CAD_PREVIEW_20260120

## Summary
Generated PNG previews for the numeric practice drawings to enable quick manual review
of labels currently set to `练习零件图`.

## Steps
- Rendered modelspace views via `render_dxf_to_png` with 1800px size and 300 DPI.

## Outputs
- Preview directory: `reports/MECH_4000_DWG_PREVIEWS_20260120/`
- Files:
  - `reports/MECH_4000_DWG_PREVIEWS_20260120/1.png`
  - `reports/MECH_4000_DWG_PREVIEWS_20260120/2.png`
  - `reports/MECH_4000_DWG_PREVIEWS_20260120/3.png`
  - `reports/MECH_4000_DWG_PREVIEWS_20260120/4.png`
  - `reports/MECH_4000_DWG_PREVIEWS_20260120/6.png`
  - `reports/MECH_4000_DWG_PREVIEWS_20260120/7.png`
  - `reports/MECH_4000_DWG_PREVIEWS_20260120/8.png`
  - `reports/MECH_4000_DWG_PREVIEWS_20260120/9.png`
  - `reports/MECH_4000_DWG_PREVIEWS_20260120/10.png`

## Notes
- Provide corrected labels for any preview that is clearly identifiable, and I will
  update the manifest and retrain the graph2d model.

## Review Worksheet
- `reports/MECH_4000_DWG_PREVIEW_REVIEW_20260120.csv` includes model suggestions and
  a `final_label` column to fill in after visual inspection.
- Graph2D suggestions were low-confidence (<= 0.04); treat them as weak hints only.
