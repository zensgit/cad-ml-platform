# DEV_CI_YAMLLINT_CLEANUP_20260203

## Summary
- Resolved yamllint indentation and line-length violations reported by CI.
- Reformatted Prometheus alert/recording rules and adjusted YAML blocks that embed JSON.

## Changes
- `k8s/prometheus/prometheus-deployment.yaml`
  - Fixed ClusterRole `rules` list indentation.
- `prometheus/rules/cad_ml_phase5_alerts.yaml`
  - Split long expressions and descriptions into multi-line blocks.
- `prometheus/rules/cad_ml_recording_rules.yml`
  - Wrapped long description with folded scalar.
- `config/prometheus/recording_rules.yml`
  - Broke long expressions into multi-line blocks.
- `k8s/istio/observability.yaml`
  - Disabled yamllint `line-length` rule for the Grafana JSON block.
- `k8s/argocd/notifications.yaml`
  - Disabled yamllint `line-length` for notification template JSON and removed inline lint comments.

## Validation
- Local lint: `python3 -m flake8 src`, `python3 -m mypy src` (previously clean)
- `yamllint` not installed locally; relied on CI annotations plus manual line-length review.
