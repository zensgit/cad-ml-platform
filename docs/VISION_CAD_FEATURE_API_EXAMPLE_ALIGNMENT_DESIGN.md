# Vision CAD Feature API Example Alignment Design

## Overview
Align the FastAPI endpoint documentation examples with the current
`cad_feature_stats` shape so the OpenAPI description matches the response
payload.

## Changes
- Add `arc_sweep_bins` to the example response in the vision analyze endpoint
  docstring.
- Keep the example fields consistent with the typed `CadFeatureStats` model and
  README documentation.

## Tests
- `pytest tests/vision/test_vision_endpoint.py -v`
