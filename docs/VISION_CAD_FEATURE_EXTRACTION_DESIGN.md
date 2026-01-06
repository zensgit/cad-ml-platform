# Vision CAD Feature Extraction Design

## Overview
Provide lightweight CAD feature extraction when no specialized model is
available. The heuristic focuses on simple line and circle detection from
rasterized drawings and records basic drawing stats.

## Inputs
- PIL image from the Vision analyzer workflow.

## Processing
1. Convert the image to grayscale.
2. Capture the original width and height as overall dimensions.
3. Downscale the image to a max dimension of 256 pixels.
4. Threshold pixels (value < 200) to build an ink mask.
5. Run 4-connected component detection over the ink mask.
6. Drop tiny components (area < 12 pixels).
7. For each component, compute bbox, aspect ratio, fill ratio, and an
   elongation ratio from the covariance matrix.
8. Classify:
   - Line: aspect ratio >= 4.0 or elongation >= 6.0.
   - Circle: aspect ratio <= 1.3 and fill ratio >= 0.3.
   - Arc: aspect ratio <= 2.5 and fill ratio in [0.05, 0.3).

## Outputs
- `drawings.lines`: list of `{bbox, length, fill_ratio}`.
- `drawings.circles`: list of `{bbox, radius, fill_ratio}`.
- `drawings.arcs`: list of `{bbox, radius, fill_ratio}`.
- `dimensions.overall_width`, `dimensions.overall_height`.
- `stats.ink_ratio`, `stats.components`.

## Limitations
- Heuristic only; no arc detection or vector-level precision.
- 4-connected components may split diagonal strokes.
- Arc detection is heuristic and does not estimate sweep angle.

## Tests
- `tests/unit/test_vision_cad_feature_extraction.py`
