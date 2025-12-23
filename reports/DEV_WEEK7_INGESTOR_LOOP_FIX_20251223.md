# Telemetry Ingestor Loop Safety Fix

## Summary
- Hardened `TelemetryIngestor.stop()` to avoid awaiting a worker task bound to a different event loop.
- Tracked the worker's loop at creation time and cleared the lock on stop to prevent reuse across loops.

## Code Changes
- `src/core/twin/ingest.py`
  - Added `_worker_loop` tracking.
  - Only `await` the worker when the current loop matches the worker loop.
  - Reset `_start_lock` on stop.

## Tests
- `pytest tests/unit/test_timeseries_store_factory.py -q`
  - Result: PASS (2 passed)
- `pytest tests/unit/test_twin_history_endpoint.py -q`
  - Result: PASS (1 passed)
- `pytest tests/integration/test_telemetry_mqtt_integration.py -q`
  - Result: SKIPPED (module skipped; pytest exit code 5)

## Notes
- This addresses the CI error: `RuntimeError: await wasn't used with future` from `TelemetryIngestor.stop()`.
- Full suite still pending; continue with remaining failing tests after this fix.
