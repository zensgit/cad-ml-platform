# 2D Part Recognition JSON Input

## Overview
This document describes the JSON payload accepted by `/api/v1/analyze/` for 2D CAD part recognition.
The JSON payload represents a vector drawing (entities + optional text/dimension metadata). It is
parsed into a `CadDocument` via `JsonAdapter`, normalized with the DedupCAD v2 schema, and then
used for L2 fusion classification.

## Supported Uploads
- File extension: `.json`
- MIME type: `application/json`

## Top-Level Structure
- `entities` (list): Vector entities (LINE, CIRCLE, ARC, LWPOLYLINE, POLYLINE, TEXT, MTEXT, DIMENSION, etc.)
- `text_content` (list, optional): Plain text extracted from the drawing
- `meta` (dict, optional): Metadata such as drawing number or identifiers
- `layers` (dict, optional): Layer metadata

## Entity Fields (Common)
- `type` (string): Entity type, e.g. LINE, CIRCLE, LWPOLYLINE
- `layer` (string, optional): Layer name

## Entity Fields (Examples)
- LINE: `start` [x, y], `end` [x, y]
- CIRCLE: `center` [x, y], `radius`
- ARC: `center` [x, y], `radius`, `start_angle`, `end_angle`
- LWPOLYLINE/POLYLINE: `points` [[x, y], ...], `closed` (optional)
- TEXT/MTEXT: `text`
- DIMENSION: `text`, `value`, `tol`, `unit`

## Example Payload
See `examples/2d_part_sample.json` for a minimal example with a bolt keyword.

## API Usage
```bash
curl -X POST \
  -H "X-API-Key: test" \
  -F 'options={"extract_features": true, "classify_parts": true}' \
  -F 'file=@examples/2d_part_sample.json;type=application/json' \
  http://localhost:8000/api/v1/analyze/
```

## Classification Notes
- The L2 fusion path uses keyword matches from `data/knowledge/geometry_rules.json`.
- Text signals are built from filename, `text_content`, and `meta.drawing_number` (or similar keys).
- OCR is **not** required for JSON/DXF inputs. Use OCR only for raster inputs (PNG/PDF).
