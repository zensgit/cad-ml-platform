#!/usr/bin/env markdown
# Faiss Tests

## Summary
- Ran Faiss health and recovery scheduling tests.

## Tests
- `python3 -m pytest tests/unit/test_faiss_health_response.py -q`
- `python3 -m pytest tests/unit/test_faiss_eta_reset_on_recovery.py -q`
- `python3 -m pytest tests/unit/test_faiss_eta_schedules_on_failed_recovery.py -q`
- `python3 -m pytest tests/unit/test_faiss_degraded_batch.py -q`

## Results
- 12 passed
