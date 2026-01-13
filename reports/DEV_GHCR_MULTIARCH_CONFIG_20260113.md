# DEV_GHCR_MULTIARCH_CONFIG_20260113

## Commands
- CAD_ML_IMAGE=ghcr.io/zensgit/cad-ml-platform:main CAD_ML_PLATFORM=linux/arm64 docker compose -f deployments/docker/docker-compose.yml -f deployments/docker/docker-compose.ghcr.yml config > /tmp/cadml-ghcr-multiarch-config.yml

## Results
- Docker compose config rendered successfully with the arm64 platform override.
