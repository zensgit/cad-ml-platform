# DRAWING_RECOGNITION_TITLE_BLOCK_NORMALIZATION_DESIGN

## Goal
- Normalize title block parsing outputs for consistent downstream consumption.

## Changes
- Accept `NTS` / `Not to scale` values for the scale field.
- Normalize scale separators (e.g., `1 : 2` -> `1:2`).
- Canonicalize projection values to `first` / `third` across language variants.
- Keep the drawing recognition request dependency explicit for FastAPI injection.

## Approach
- Extend scale regex patterns and add normalization helpers for scale/projection.
- Update title block parser tests to cover the new normalization paths.
