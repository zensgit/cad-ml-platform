# DEV_MECH_KNOWLEDGE_4000CAD_GRAPH2D_OVERRIDE_DEPLOY_SYNC_20260123

## Summary
- Synced Graph2D override threshold defaults into deployment templates.
- Kept override disabled by default; only the threshold value is explicit.

## Changes
- `deployments/docker/docker-compose.yml` (FUSION_GRAPH2D_OVERRIDE_MIN_CONF default 0.6)
- `k8s/app/deployment.yaml` (FUSION_GRAPH2D_OVERRIDE_MIN_CONF=0.6)

## Notes
- Override remains opt-in via feature flags; this only standardizes the threshold when enabled.
