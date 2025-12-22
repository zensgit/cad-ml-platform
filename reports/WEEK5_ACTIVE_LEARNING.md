#!/usr/bin/env markdown
# Week 5 Report: Active Learning

## Summary
- Added Active Learning API endpoints and low-confidence sampling hook.

## Changes
- API router: `src/api/v1/active_learning.py`
- Registered router in `src/api/__init__.py`
- Sampling hook in `src/api/v1/analyze.py`
- Doc: `docs/ACTIVE_LEARNING_OPERATIONS.md` + README link

## Tests
- `python3 -m pytest tests/unit/test_active_learning_loop.py -q`

## Verification
- Enable with env:
  - `ACTIVE_LEARNING_ENABLED=true`
  - `ACTIVE_LEARNING_CONFIDENCE_THRESHOLD=0.6`
- Trigger low-confidence analysis and call:
  - `GET /api/v1/active-learning/pending`
