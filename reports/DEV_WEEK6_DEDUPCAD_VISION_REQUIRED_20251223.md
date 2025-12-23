# Week6 CI - DedupCAD Vision Required (2025-12-23)

## Change
- Added `scripts/dedupcad_vision_stub.py` for local contract testing.
- Contract and E2E dedup tests now honor `DEDUPCAD_VISION_REQUIRED=1` (fail instead of skip).
- CI workflow (`.github/workflows/ci.yml`) now uses the real dedupcad-vision image; stub remains a local fallback.

## Test
- `make e2e-smoke`

## Result
- Passed: 4
- Failed: 0
