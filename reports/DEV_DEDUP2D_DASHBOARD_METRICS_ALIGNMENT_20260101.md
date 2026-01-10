# DEV_DEDUP2D_DASHBOARD_METRICS_ALIGNMENT_20260101

## Design Summary
- Aligned Dedup2D Grafana dashboard queries with metrics actually emitted by cad-ml-platform.
- Replaced deprecated/unused job metrics with `dedup2d_jobs_total` and `dedup2d_job_queue_depth`.
- Updated job overview stats and the jobs time series panel to reflect new query sources.

## Files Updated
- `grafana/dashboards/dedup2d.json`

## Verification
```bash
jq -e . grafana/dashboards/dedup2d.json > /dev/null
```
Result: JSON valid.
