# EXTERNAL_DOCKER_NETWORK_GITIGNORE_DESIGN

## Context
- A local-only Docker compose file declares an external network used by the developer environment.

## Decision
- Ignore `deployments/docker/docker-compose.external-network.yml` to avoid committing environment-specific configuration.

## Rationale
- Keeps repository portable and prevents accidental leakage of local infra details.
