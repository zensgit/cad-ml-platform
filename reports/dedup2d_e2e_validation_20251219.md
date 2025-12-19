# Dedup2D E2E Validation Report

Date: 2025-12-19

## Scope
- cad-ml-platform (API + ARQ worker)
- dedupcad-vision (vision API)

## Environment
- cad-ml-platform API: http://localhost:18000
- dedupcad-vision API: http://localhost:58001
- Redis (cad-ml-platform): cad-ml-redis container

## Pre-checks
- `GET http://localhost:18000/health` -> 200
- `GET http://localhost:58001/health` -> 200

## Test Case: 2D Async Search
### Request
- Endpoint: `POST /api/v1/dedup/2d/search?async=true&mode=balanced&max_results=5`
- Header: `X-API-Key: test`
- File: `reports/eval_history/plots/combined_trend.png`

### Response (submit)
- `job_id`: `300b9751-ac42-4c75-97d7-90e5d43a0d3b`
- `status`: `pending`

### Response (poll)
- Endpoint: `GET /api/v1/dedup/2d/jobs/300b9751-ac42-4c75-97d7-90e5d43a0d3b`
- Final status: `completed`
- Result summary:
  - `success`: true
  - `total_matches`: 0
  - `final_level`: 1
  - `timing.total_ms`: 423.85
  - `error`: null

## Notes
- A synthetic 1x1 PNG caused `broken data stream when reading image file` in dedupcad-vision.
  Using a real PNG asset resolved the issue.

## Conclusion
- Async job submission, Redis queueing, worker execution, and vision call succeeded end-to-end.
