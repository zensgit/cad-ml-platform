# DEV_DEDUP2D_PROMETHEUS_TARGETS_20260101

## Scope
- Verify Prometheus is scraping cad-ml-api metrics in staging compose.
- Confirm `dedup2d_*` metrics are queryable.

## Targets
```bash
curl -sS http://127.0.0.1:19091/api/v1/targets \
  | jq '.data.activeTargets[] | {scrapePool, health, lastError, lastScrape, scrapeUrl}'
```
Result (summary):
- `cad-ml-api` target `health=up`, scrape URL `http://cad-ml-api:8000/metrics`.
- `prometheus` target `health=up`.

## Queries
```bash
curl -sS 'http://127.0.0.1:19091/api/v1/query?query=up%7Bjob%3D%22cad-ml-api%22%7D' | jq '.data.result'
```
Result: `up{job="cad-ml-api"}=1`.

```bash
curl -sS 'http://127.0.0.1:19091/api/v1/query?query=dedup2d_jobs_total' | jq '.data.result'
```
Result: `dedup2d_jobs_total{status="pending"}=1`, `dedup2d_jobs_total{status="completed"}=1`.

## Result
- Prometheus scraping is healthy and dedup2d metrics are available via query API.
