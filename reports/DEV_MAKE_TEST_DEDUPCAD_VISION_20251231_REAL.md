# DedupCAD Vision Required Test Run (Real Service) (2025-12-31)

## Scope

- Run `make test-dedupcad-vision` against a real dedupcad-vision service.

## Environment

- dedupcad-vision: Docker container `dedupcad-vision-api` (port 8100 -> 8000)
- cad-ml-platform API: `uvicorn src.main:app --port 8003`
- `DEDUPCAD_VISION_URL=http://localhost:8100`
- `API_BASE_URL=http://localhost:8003`

## Command

- `API_BASE_URL=http://localhost:8003 DEDUPCAD_VISION_URL=http://localhost:8100 make test-dedupcad-vision`

## Results

- Passed: 3991
- Skipped: 13
- Failed: 0
- Duration: 2m 35s
- Coverage: 71% (html report in `htmlcov/`)

## Notes

- dedupcad-vision container was already running and reused.
- cad-ml-platform API stopped after the run.
