# DEV_DEDUP2D_PRECISION_SMOKE_20260101

## Scope
- Validate geom_json + precision path for dedup2d async jobs.

## Request
```bash
cat <<'JSON' > /tmp/geom.json
{"entities": []}
JSON

curl -sS -H 'X-API-Key: test' \
  -F 'file=@reports/eval_history/plots/combined_trend.png' \
  -F 'geom_json=@/tmp/geom.json;type=application/json' \
  'http://localhost:18000/api/v1/dedup/2d/search?async=true&enable_precision=true'
```
Response (summary): `job_id=7573fdf9-2e8a-4e88-9792-6567367105eb`.

## Result
```bash
curl -sS -H 'X-API-Key: test' \
  http://localhost:18000/api/v1/dedup/2d/jobs/7573fdf9-2e8a-4e88-9792-6567367105eb | jq '.'
```
Observed:
- `status=completed`
- `timing.l4_ms` and `timing.precision_ms` present (precision overlay executed).
- No duplicates/similar matches (empty test data).

## Result
- geom_json upload accepted and precision path executed successfully.
