# Helm Validation Report

Date: 2025-12-19

## Scope
- cad-ml-platform: `charts/cad-ml-platform/values.yaml` + `charts/cad-ml-platform/values-prod.yaml`
- dedupcad-vision: `deploy/helm/caddedup-vision/values.yaml` + `deploy/helm/caddedup-vision/values-prod.yaml`

## Environment
- Helm runtime: `alpine/helm:3.13.2` via Docker (local `helm` binary not installed).

## Results
### cad-ml-platform
- `helm lint`: PASS
  - Note: Chart.yaml recommends adding `icon` (non-blocking).
- `helm template`: PASS
  - Rendered line count: 347

### dedupcad-vision
- `helm lint`: PASS
  - Note: Chart.yaml recommends adding `icon` (non-blocking).
- `helm template`: PASS
  - Rendered line count: 2112
  - Dependency: required `helm dependency build` for redis subchart.

## Fixes Applied During Validation
- Escaped Grafana legend format variables that conflicted with Helm templating:
  - `{{service}}` and `{{pod}}` are now rendered as literal Grafana legend templates.
  - File: `deploy/helm/caddedup-vision/templates/grafana-dashboard.yaml`
  - Commit: `9a827ce` (dedupcad-vision)

## Commands Executed
```bash
# cad-ml-platform
 docker run --rm -v /Users/huazhou/Downloads/Github/cad-ml-platform:/work -w /work alpine/helm:3.13.2 lint charts/cad-ml-platform -f charts/cad-ml-platform/values.yaml -f charts/cad-ml-platform/values-prod.yaml
 docker run --rm --entrypoint /bin/sh -v /Users/huazhou/Downloads/Github/cad-ml-platform:/work -w /work alpine/helm:3.13.2 -c "helm template cad-ml-platform charts/cad-ml-platform -f charts/cad-ml-platform/values.yaml -f charts/cad-ml-platform/values-prod.yaml | wc -l"

# dedupcad-vision
 docker run --rm --entrypoint /bin/sh -v /Users/huazhou/Downloads/Github/dedupcad-vision:/work -w /work alpine/helm:3.13.2 -c "helm dependency build deploy/helm/caddedup-vision"
 docker run --rm -v /Users/huazhou/Downloads/Github/dedupcad-vision:/work -w /work alpine/helm:3.13.2 lint deploy/helm/caddedup-vision -f deploy/helm/caddedup-vision/values.yaml -f deploy/helm/caddedup-vision/values-prod.yaml
 docker run --rm --entrypoint /bin/sh -v /Users/huazhou/Downloads/Github/dedupcad-vision:/work -w /work alpine/helm:3.13.2 -c "helm template caddedup-vision deploy/helm/caddedup-vision -f deploy/helm/caddedup-vision/values.yaml -f deploy/helm/caddedup-vision/values-prod.yaml | wc -l"
```
