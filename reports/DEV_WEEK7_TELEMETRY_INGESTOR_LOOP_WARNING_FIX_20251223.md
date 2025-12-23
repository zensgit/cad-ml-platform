# DEV_WEEK7_TELEMETRY_INGESTOR_LOOP_WARNING_FIX_20251223

## Context
- CI logs reported a PytestUnraisableExceptionWarning: `RuntimeError: Event loop is closed` from `TelemetryIngestor._run`.

## Changes
- `src/core/twin/ingest.py`: guard `_run()` against cancellation/loop teardown to exit cleanly when the event loop closes.

## Tests
- `pytest tests/unit/test_telemetry_ingestor.py tests/unit/test_twin_history_endpoint.py -q`

## Results
- Tests passed; no unraisable exception warnings observed in the run.
