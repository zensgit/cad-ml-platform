# Week6 CI - DedupCAD Vision Pinned Image (2025-12-23)

## Change
- CI `e2e-smoke` job now uses a pinned dedupcad-vision image with override support via `DEDUPCAD_VISION_IMAGE` (default `caddedup/vision:1.1.0`).
- Host mapping remains `58001:8000` to avoid clashing with the API on 8000.
- E2E smoke now normalizes zero feature vectors to a deterministic non-zero probe for stable nearest-neighbor assertions.

## Test
- `make e2e-smoke`

## Result
- Passed: 4
- Failed: 0
