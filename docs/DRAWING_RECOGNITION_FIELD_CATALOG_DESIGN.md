# DRAWING_RECOGNITION_FIELD_CATALOG_DESIGN

## Goal
- Expose a stable field catalog for drawing recognition clients.
- Provide a structured title block map alongside the existing field list for easier consumption.

## Changes
- Add `GET /api/v1/drawing/fields` to return the drawing field keys and labels.
- Extend drawing recognition response with `title_block` (key/value map) in addition to the field list.

## Approach
- Reuse `FIELD_LABELS` as the single source of truth for catalog output.
- Populate `title_block` from the OCR result model to keep parity with `/api/v1/ocr/extract`.
