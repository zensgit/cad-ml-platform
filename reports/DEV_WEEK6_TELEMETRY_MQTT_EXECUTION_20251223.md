# Week6 Telemetry/MQTT Execution (2025-12-23)

## Changes
- `/api/v1/twin/*` routes are now registered in the API router.
- Documentation updated to reflect default route availability.

## Environment
- API: http://localhost:8000
- dedupcad-vision: http://localhost:58001
- MQTT broker: local Mosquitto (Docker) on 1883
- Python: .venv (3.11)

## Integration Suite
- Command: `API_BASE_URL=http://localhost:8000 DEDUPCAD_VISION_URL=http://localhost:58001 MQTT_HOST=localhost MQTT_PORT=1883 ./.venv/bin/python -m pytest tests/integration/ -q -rs`
- Result: 14 passed, 0 skipped

## E2E Smoke
- Command: `API_BASE_URL=http://localhost:8000 DEDUPCAD_VISION_URL=http://localhost:58001 ./.venv/bin/python -m pytest tests/integration/test_e2e_api_smoke.py tests/integration/test_dedupcad_vision_contract.py -v -rs`
- Result: 4 passed

## Notes
- MQTT broker was started for the run and stopped afterwards.
