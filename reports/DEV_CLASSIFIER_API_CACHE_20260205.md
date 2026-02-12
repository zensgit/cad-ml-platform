# DEV_CLASSIFIER_API_CACHE_20260205

## Summary
Enabled LRU caching for the DXF classification API to avoid repeated inference
on identical uploads and added cache management endpoints.

## Changes
- Added cache usage to `POST /classify` and `POST /classify/batch` in
  `src/inference/classifier_api.py`.
- Added cache stats (`GET /cache/stats`) and cache clear (`POST /cache/clear`).

## Notes
- Cache key is based on file-content hash, so different filenames with identical
  contents reuse the same prediction.
