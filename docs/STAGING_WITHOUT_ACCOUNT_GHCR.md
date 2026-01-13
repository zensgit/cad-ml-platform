# Staging Without an Account (GHCR + Compose)

## Goal
Provide a lightweight staging-like workflow using prebuilt GHCR images when no staging account is available.

## Prerequisites
- GHCR image published via `.github/workflows/ghcr-publish.yml` (tags: `main`, `sha-<commit>`).
- Docker Compose v2.

## Pull and Run (Preprod)
```bash
# Optional: authenticate if the GHCR package is private
# echo "$GHCR_TOKEN" | docker login ghcr.io -u <github-username> --password-stdin

# Use the prebuilt image
export CAD_ML_IMAGE=ghcr.io/zensgit/cad-ml-platform:main

docker compose -f deployments/docker/docker-compose.yml \
  -f deployments/docker/docker-compose.ghcr.yml pull

docker compose -f deployments/docker/docker-compose.yml \
  -f deployments/docker/docker-compose.ghcr.yml up -d --no-build
```

## Optional: Dedup2D staging overrides
```bash
export CAD_ML_IMAGE=ghcr.io/zensgit/cad-ml-platform:main

docker compose -f deployments/docker/docker-compose.yml \
  -f deployments/docker/docker-compose.minio.yml \
  -f deployments/docker/docker-compose.dedup2d-staging.yml \
  -f deployments/docker/docker-compose.ghcr.yml up -d --no-build
```

## Validation
```bash
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/ready
curl -fsS http://localhost:9090/metrics | head -n 20
```

## Notes
- Set `DEDUPCAD_VISION_URL` to point at the running dedupcad-vision service.
- Use `CAD_ML_IMAGE=ghcr.io/<org>/<repo>:sha-<commit>` for pinned rollbacks.
- Keep GHCR images public or grant `packages:read` to the CI/service account.
