# DRAWING_RECOGNITION_FIELD_CONFIDENCE_MAP_DESIGN

## Goal
- Provide a direct key-to-confidence mapping for title block fields in drawing recognition responses.

## Changes
- Add `field_confidence` map to `/api/v1/drawing/recognize` responses.

## Approach
- Reuse the existing per-field confidence logic and emit a flattened map keyed by field name.
