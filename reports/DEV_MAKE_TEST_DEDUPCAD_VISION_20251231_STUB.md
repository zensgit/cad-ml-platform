# DedupCAD Vision Required Test Run (Stub) (2025-12-31)

## Scope

- Run `make test-dedupcad-vision` using the local dedupcad-vision stub and a local cad-ml-platform API.

## Command

- `API_BASE_URL=http://localhost:8001 DEDUPCAD_VISION_URL=http://localhost:58001 make test-dedupcad-vision`

## Environment

- dedupcad-vision stub: `scripts/dedupcad_vision_stub.py` on `http://localhost:58001`
- cad-ml-platform API: `uvicorn src.main:app --port 8001`

## Results

- Passed: 3982
- Skipped: 22
- Failed: 0
- Duration: 1m 49s
- Coverage: 71% (html report in `htmlcov/`)

## Notes

- Both stub and API were stopped after the run.
