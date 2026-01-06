# Analysis History Retrieval and Logging Design

## Overview
This design documents how analysis results are retrieved after processing and how
OCR/vision logging is structured for error visibility. The goal is to provide a
stable, cache-friendly history lookup while emitting consistent error logs.

## Analysis History Retrieval
Retrieval order for `GET /api/v1/analyze/{analysis_id}`:
1. Cache lookup: `analysis_result:{analysis_id}` in Redis/in-memory cache.
2. Disk fallback: JSON file in `ANALYSIS_RESULT_STORE_DIR/{analysis_id}.json`.
3. If found on disk, rehydrate cache with a standard TTL.
4. If still missing, return 404.

### Storage on Write
- After successful analysis, results are cached and also persisted to disk when
  `ANALYSIS_RESULT_STORE_DIR` is configured.
- Writes are atomic: JSON is written to a temporary file and replaced.

### Configuration
- `ANALYSIS_RESULT_STORE_DIR`: enable disk persistence.
- `ANALYSIS_RESULT_STORE_TTL_SECONDS`: optional cleanup policy.
- `ANALYSIS_RESULT_STORE_MAX_FILES`: optional cleanup policy.
- `ANALYSIS_RESULT_CLEANUP_INTERVAL_SECONDS`: optional periodic cleanup.

## Logging Improvements
- Vision OCR extraction failures emit structured fields:
  - `provider`, `stage`, `error_code`, `error`.
- Logging format is JSON (see `src/utils/logging.py`) to keep telemetry fields
  consistent across components.

## Limitations
- Disk persistence is local-only; no distributed storage or replication.
- Logging is best-effort and does not replace error responses or metrics.
