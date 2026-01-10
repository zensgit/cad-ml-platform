# Vision CAD Feature API Tuning Design

## Overview
Expose CAD feature heuristic thresholds on the vision analyze request so callers
can tune lightweight extraction without code changes. The API accepts the
threshold overrides and forwards them through the request object.

## Request Field
- `cad_feature_thresholds`: optional dict of numeric overrides.

### Supported Keys
- `max_dim`
- `ink_threshold`
- `min_area`
- `line_aspect`
- `line_elongation`
- `circle_aspect`
- `circle_fill_min`
- `arc_aspect`
- `arc_fill_min`
- `arc_fill_max`

## Example
```json
{
  "image_base64": "iVBORw0KGgoAAAANS...",
  "include_description": true,
  "include_ocr": true,
  "cad_feature_thresholds": {
    "line_aspect": 5.0,
    "arc_fill_min": 0.08
  }
}
```

## Notes
- Thresholds are optional and do not change default behavior when omitted.
