# DEV_CLASSIFIER_API_LIFESPAN_20260205

## Summary
Migrated the classifier API startup hook to FastAPI lifespan handlers to remove
deprecation warnings and keep model warmup in a supported lifecycle.

## Changes
- Replaced `@app.on_event("startup")` with a lifespan context manager in
  `src/inference/classifier_api.py`.

## Tests
- `python3 -m pytest tests/unit/test_classifier_api_cache.py -q`
