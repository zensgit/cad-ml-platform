# DEV_GRAPH2D_MIN_CONF_RUN_TARGET_VALIDATION_20260123

## Checks
- Verified the `make run` target exports `GRAPH2D_MIN_CONF` with a default value.

## Runtime Output
- Command:
  - `make -n run`
- Result:
  - `GRAPH2D_MIN_CONF=${GRAPH2D_MIN_CONF:-0.6} uvicorn src.main:app --reload --host 0.0.0.0 --port 8000`
