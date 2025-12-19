# Dedup2D Callback Validation Report

Date: 2025-12-19

## Goal
Verify `callback_url` delivery for async dedup2d jobs.

## Environment
- cad-ml-platform: http://localhost:18000
- dedupcad-vision: http://localhost:58001
- Callback target: https://postman-echo.com/post

## Pre-checks
- `GET http://localhost:18000/health` -> 200
- `GET http://localhost:58001/health` -> 200

## Test Case: Async search with callback_url
### Request
- Endpoint: `POST /api/v1/dedup/2d/search?async=true&callback_url=https://postman-echo.com/post&mode=balanced&max_results=5`
- Header: `X-API-Key: test`
- File: `reports/eval_history/plots/combined_trend.png`

### Submit Response
- `job_id`: `b8f5ef0e-51be-41de-95f1-6020c3b7c977`
- `status`: `pending`

### Poll Response (final)
- `status`: `completed`
- `callback_status`: `success`
- `callback_attempts`: `1`
- `callback_http_status`: `200`
- `callback_last_error`: `null`

### Result Summary
- `success`: true
- `total_matches`: 0
- `final_level`: 1
- `timing.total_ms`: 334.86

## Conclusion
Callback URL delivery succeeded (HTTP 200) on first attempt.
