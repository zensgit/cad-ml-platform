# Archived (superseded by stub-based CI on 2025-12-23)

# Week6 CI - DedupCAD Vision Real Image (2025-12-23)

## Change
- CI now starts a pinned dedupcad-vision image (`caddedup/vision:1.1.0`) with optional override via `DEDUPCAD_VISION_IMAGE`, and maps host `58001` -> container `8000`.
- Removed CI dependency on the stub for required contract checks; stub remains for local dev fallback.

## Test
- `make e2e-smoke`

## Result
- Passed: 4
- Failed: 0
