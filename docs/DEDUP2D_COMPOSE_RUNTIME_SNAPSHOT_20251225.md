# Dedup2D Compose Runtime Snapshot (2025-12-25)

Scope: capture the running local staging stack for cad-ml-platform + dedup2d worker
using the compose overrides, with MinIO host ports enabled.

## Compose command

```bash
CAD_ML_MINIO_PORT=19000 CAD_ML_MINIO_CONSOLE_PORT=19001 \
  docker compose -f deployments/docker/docker-compose.yml \
  -f deployments/docker/docker-compose.minio.yml \
  -f deployments/docker/docker-compose.dedup2d-staging.yml up -d
```

## Container snapshot (docker ps)

```text
cad-ml-dedup2d-worker        Up 7 seconds              8000/tcp, 9090/tcp                                                                             cad-ml-platform:latest
cad-ml-minio                 Up 13 seconds (healthy)   0.0.0.0:19000->9000/tcp, [::]:19000->9000/tcp, 0.0.0.0:19001->9001/tcp, [::]:19001->9001/tcp   minio/minio:latest
cad-ml-api                   Up 18 hours (healthy)     0.0.0.0:8000->8000/tcp, [::]:8000->8000/tcp, 0.0.0.0:9090->9090/tcp, [::]:9090->9090/tcp       cad-ml-platform:latest
cad-ml-grafana               Up 19 hours               0.0.0.0:3001->3000/tcp, [::]:3001->3000/tcp                                                    grafana/grafana:latest
cad-ml-redis                 Up 19 hours               0.0.0.0:16379->6379/tcp, [::]:16379->6379/tcp                                                  redis:7-alpine
cad-ml-prometheus            Up 19 hours               0.0.0.0:9091->9090/tcp, [::]:9091->9090/tcp                                                    prom/prometheus:latest
```

## Network membership (cad-ml-network)

```text
NAME	CONTAINER_ID	IPV4	IPV6
cad-ml-api	79bb2b44dc4f	172.19.0.7	
cad-ml-dedup2d-worker	b4289c1f98ed	172.19.0.6	
cad-ml-grafana	d0d3eaac170e	172.19.0.2	
cad-ml-minio	8d0b664b1386	172.19.0.5	
cad-ml-prometheus	590f168d5870	172.19.0.4	
cad-ml-redis	1d6409ec870a	172.19.0.3	
```

## Notes

- MinIO is exposed on host ports 19000 (S3 API) and 19001 (console) to avoid conflicts
  with other local MinIO stacks using 9000/9001.
- The compose files pin the network name to `cad-ml-network` for consistent service
  discovery across stack components.
