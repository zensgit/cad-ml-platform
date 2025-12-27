# Dedup2D Compose Runtime Snapshot (2025-12-24)

Scope: capture the running local staging stack for cad-ml-platform + dedup2d worker
using the compose overrides.

## Compose command

```bash
docker compose -f deployments/docker/docker-compose.yml \
  -f deployments/docker/docker-compose.minio.yml \
  -f deployments/docker/docker-compose.dedup2d-staging.yml up -d
```

## Container snapshot (docker ps)

```text
cad-ml-api                   Up 19 minutes (healthy)     0.0.0.0:8000->8000/tcp, [::]:8000->8000/tcp, 0.0.0.0:9090->9090/tcp, [::]:9090->9090/tcp       cad-ml-platform:latest
cad-ml-dedup2d-worker        Up 19 minutes (unhealthy)   8000/tcp, 9090/tcp                                                                             cad-ml-platform:latest
cad-ml-minio                 Up 34 minutes (healthy)     9000-9001/tcp                                                                                  minio/minio:latest
cad-ml-grafana               Up 41 minutes               0.0.0.0:3001->3000/tcp, [::]:3001->3000/tcp                                                    grafana/grafana:latest
cad-ml-redis                 Up 41 minutes               0.0.0.0:16379->6379/tcp, [::]:16379->6379/tcp                                                  redis:7-alpine
cad-ml-prometheus            Up 41 minutes               0.0.0.0:9091->9090/tcp, [::]:9091->9090/tcp                                                    prom/prometheus:latest
```

## Network membership (cad-ml-network)

```text
NAME	CONTAINER_ID	IPV4	IPV6
cad-ml-api	79bb2b44dc4f	172.19.0.7	
cad-ml-dedup2d-worker	9ac731876727	172.19.0.6	
cad-ml-grafana	d0d3eaac170e	172.19.0.2	
cad-ml-minio	87b45e51881b	172.19.0.5	
cad-ml-prometheus	590f168d5870	172.19.0.4	
cad-ml-redis	1d6409ec870a	172.19.0.3	
```

## Notes

- MinIO shows only container ports (9000-9001/tcp) in this snapshot; host port mapping
  is disabled here and S3 access is internal to the docker network.
- The compose files pin the network name to `cad-ml-network` for consistent service
  discovery across stack components.
