# BREP_PYTHONOCC_API_UPDATE_DESIGN

## Goal
Replace deprecated pythonocc-core helper functions with the current module APIs to remove deprecation warnings during B-Rep analysis.

## Changes
- Swap `brepgprop_VolumeProperties` / `brepgprop_SurfaceProperties` for `brepgprop.VolumeProperties` and `brepgprop.SurfaceProperties`.
- Swap `brepbndlib_Add` for `brepbndlib.Add` in geometry and STEP/IGES adapter paths.

## Compatibility
- Targets pythonocc-core 7.7+ (the deprecated functions were removed from new guidance).
- No behavioral change to surface, volume, or bounding box calculations.

## Testing
- `pytest tests/integration/test_brep_features_v4.py -v` (linux/amd64 micromamba, pythonocc-core 7.9.0).
