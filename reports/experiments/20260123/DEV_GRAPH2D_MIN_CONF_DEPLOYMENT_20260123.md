# DEV_GRAPH2D_MIN_CONF_DEPLOYMENT_20260123

## Summary
- Set the default Graph2D confidence gate in deployment templates.
- Ensures production environments align with the documented 0.6 threshold.

## Updates
- `deployments/docker/docker-compose.yml`: added `GRAPH2D_MIN_CONF` env default.
- `k8s/app/deployment.yaml`: added `GRAPH2D_MIN_CONF` env.
