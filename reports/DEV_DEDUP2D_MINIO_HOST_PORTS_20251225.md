# DEV_DEDUP2D_MINIO_HOST_PORTS_20251225

## Scope

- Enable MinIO host port overrides (19000/19001) in the local staging stack.
- Resolve dedup2d-worker health status to avoid API healthcheck false negatives.
- Capture an updated compose runtime snapshot.

## Changes

- Disabled the image healthcheck for `dedup2d-worker` in `deployments/docker/docker-compose.yml`
  (worker does not expose `/health`).
- Added `docs/DEDUP2D_COMPOSE_RUNTIME_SNAPSHOT_20251225.md` with the current runtime snapshot.

## Validation

- `docker ps` shows `cad-ml-minio` mapped to host ports 19000/19001 and
  `cad-ml-dedup2d-worker` running without unhealthy status.
- `curl -sSf http://localhost:19000/minio/health/ready` returned HTTP 200.

## Notes

- MinIO container was recreated to apply host port mappings.
