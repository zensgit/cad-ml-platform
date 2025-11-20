---
name: Observability Incident
about: Report health/metrics anomalies for CAD ML Platform
title: "[Observability] <brief summary>"
labels: [observability, metrics]
assignees: []
---

## Summary
- What happened? When did it start? Any recent changes/deploys?

## Environment
- Service version (commit/tag):
- Runtime: Python {{ python --version }}
- Env/Cluster: (dev/staging/prod)

## Impact
- Affected endpoints (e.g., /api/v1/vision/analyze, /api/v1/ocr/extract):
- Severity: (P1/P2/P3) and user impact description

## Health Snapshot (/health)
Paste JSON payload, especially `runtime.error_rate_ema` and `runtime.config`:

```
<paste /health response>
```

## Metrics Evidence (/metrics)
If Prometheus is enabled, paste relevant lines or screenshots:
- vision_input_rejected_total{reason=...}
- ocr_input_rejected_total{reason=...}
- vision_error_rate_ema / ocr_error_rate_ema
- vision_image_size_bytes / ocr_image_size_bytes

## Reproduction Steps
1. Request payload(s) and sequence
2. Expected vs actual results
3. Frequency (always/often/rare) and time window

## Logs & Screenshots
- Relevant logs (redact secrets)
- Grafana screenshots if available (panel JSON in docs/grafana/)

## Suspected Causes / Notes
- Configuration, upstream provider, network, or recent code changes

## Runbooks & References
- Alert rules: docs/ALERT_RULES.md
- Runbooks: docs/runbooks/
- Grafana dashboard: docs/grafana/observability_dashboard.json

