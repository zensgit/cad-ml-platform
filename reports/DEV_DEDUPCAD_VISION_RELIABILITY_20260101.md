# DedupCAD Vision Reliability Tests (2026-01-01)

## Scope

- Validate dedupcad-vision client retry and circuit-breaker handling.

## Changes

- Added unit tests for retry-on-500, timeout retry exhaustion, and circuit-open short-circuit behavior.

## Tests

- `pytest tests/unit/test_dedupcad_vision_client.py -v`

## Results

- OK: 3 passed.
