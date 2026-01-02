# Health Alias Refactor Report

- Date: 2025-12-28
- Scope: Health payload sharing and OCR metrics registration tests

## Summary
- Extracted shared health payload builder to avoid circular imports between `/health` and `/api/v1/health`.
- Updated health/metrics tests to use a valid PNG payload so OCR provider metrics register.

## Tests
- `.venv/bin/python -m pytest tests/test_health_and_metrics.py -q`
