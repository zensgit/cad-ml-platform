# Metrics Server Smoke Check

- Date: 2025-12-27
- Scope: metrics HTTP server (local scrape)

## Command
- python3 scripts/export_eval_metrics.py --serve --port 9100 (background)
- curl http://localhost:9100/metrics
- curl http://localhost:9100/health

## Result
- PASS (metrics endpoint returned Prometheus exposition; health returned OK)
- Server stopped cleanly after validation
