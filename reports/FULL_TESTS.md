# Full Test Suite Report

## Scope
- Full test suite with coverage after fixes to feature extraction and dedup2d Redis config.

## Fixes Applied
- src/core/dedupcad_2d_jobs_redis.py: add default for `render_queue_name` and reorder dataclass fields for compatibility.
- src/core/feature_extractor.py: restore v1–v4 feature vector layout (entity count/bbox/semantic + v2/v3/v4 extensions).

## Test Run
- Command: `.venv/bin/python -m pytest tests -v --cov=src --cov-report=term-missing`
- Result: `3628 passed, 42 skipped, 6 warnings in 98.53s`

## Notes
- Coverage summary printed by pytest; no failures after the above fixes.
