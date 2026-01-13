# DEV_GHCR_PREPROD_COMPOSE_RUN_20260113

## Commands
- CAD_ML_IMAGE=ghcr.io/zensgit/cad-ml-platform:main docker compose -f deployments/docker/docker-compose.yml -f deployments/docker/docker-compose.ghcr.yml -f deployments/docker/docker-compose.external-network.yml up -d --no-build --pull=never --force-recreate
- curl -fsS http://localhost:8000/health
- curl -fsS http://localhost:8000/ready
- curl -fsS http://localhost:8000/metrics/ | head -n 20

## Results
- Docker compose started successfully with GHCR image override and external network.
- /health status: healthy (metrics_enabled: true)
- /ready status: ready
- /metrics/ responded with Prometheus payload (see sample below).

## Metrics Sample (first 5 lines)
```
# HELP python_gc_objects_collected_total Objects collected during gc
# TYPE python_gc_objects_collected_total counter
python_gc_objects_collected_total{generation="0"} 90242.0
python_gc_objects_collected_total{generation="1"} 20952.0
python_gc_objects_collected_total{generation="2"} 4411.0
```

## Notes
- Using `--pull=never` avoided Docker proxy pull timeouts; local images were present.
- /metrics/ responds at `/metrics/` (trailing slash), not `/metrics`.
