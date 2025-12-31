# DedupCAD Vision Required Test Run (2025-12-31)

- Scope: `make test-dedupcad-vision` against live dedupcad-vision + local cad-ml-platform API.
- dedupcad-vision: started from `/Users/huazhou/Downloads/Github/dedupcad-vision` via `start_server.py --port 58001`.
- cad-ml-platform API: `uvicorn src.main:app --port 8001` with `DEDUPCAD_VISION_URL=http://localhost:58001`.

## Command

```bash
API_BASE_URL=http://localhost:8001 \
DEDUPCAD_VISION_URL=http://localhost:58001 \
make test-dedupcad-vision
```

## Result

- Passed: 3979
- Skipped: 22
- Failed: 0
- Coverage: 71% (HTML report in `htmlcov/`)

## Notes

- `tests/integration/test_dedupcad_vision_contract.py` and `tests/integration/test_e2e_api_smoke.py` passed.
- dedupcad-vision and cad-ml-platform API were stopped after the run.
