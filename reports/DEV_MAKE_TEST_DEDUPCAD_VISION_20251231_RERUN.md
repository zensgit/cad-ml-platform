# DedupCAD Vision Required Test Run (Re-run) (2025-12-31)

## Scope

- Re-run full `make test-dedupcad-vision` against live dedupcad-vision + local cad-ml-platform.
- Validates integration tests + full pytest suite with coverage.

## Environment

- cad-ml-platform: http://localhost:8001
- dedupcad-vision: http://localhost:58001

## Command

```bash
API_BASE_URL=http://localhost:8001 DEDUPCAD_VISION_URL=http://localhost:58001 make test-dedupcad-vision
```

## Result

- Status: ✅ Passed
- Tests: 3979 passed, 22 skipped
- Coverage: 71%
- Duration: 102.09s

## Notes

- cad-ml-platform and dedupcad-vision were started locally for the run and stopped afterward.
