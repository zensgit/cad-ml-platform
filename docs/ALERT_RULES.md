# Alert Rules (Prometheus) — CAD ML Platform

This document includes sample Prometheus alerting rules for OCR/Vision services. Tune thresholds to your SLOs and traffic.

## Groups

```
- name: cad-ml-platform-alerts
  rules:
    # 1) Error rate EMA elevated (Vision)
    - alert: VisionErrorRateEmaHigh
      expr: vision_error_rate_ema > 0.4
      for: 5m
      labels:
        severity: warning
        team: vision
      annotations:
        summary: "Vision error rate EMA elevated"
        description: "vision_error_rate_ema={{ $value }} (>0.4 for 5m). Investigate input errors and provider status."
        runbook_url: https://example.org/runbooks/ocr_vision_error_rate_ema

    # 2) Error rate EMA elevated (OCR)
    - alert: OcrErrorRateEmaHigh
      expr: ocr_error_rate_ema > 0.3
      for: 10m
      labels:
        severity: warning
        team: ocr
      annotations:
        summary: "OCR error rate EMA elevated"
        description: "ocr_error_rate_ema={{ $value }} (>0.3 for 10m). Check provider availability and recent changes."
        runbook_url: https://example.org/runbooks/ocr_vision_error_rate_ema

    # 3) Input rejections spike (Vision)
    - alert: VisionInputRejectionsSpike
      expr: rate(vision_input_rejected_total[5m]) > 3
      for: 5m
      labels:
        severity: critical
        team: vision
      annotations:
        summary: "Vision input rejections spike"
        description: "vision_input_rejected_total rate is high. Check reasons (base64_decode_error, base64_too_large)."
        runbook_url: https://example.org/runbooks/input_rejections_spike

    # 4) Input rejection ratio (Vision)
    - alert: VisionInputRejectionRatioHigh
      expr: (sum by() (rate(vision_input_rejected_total[5m])))
            /
            (sum by() (rate(vision_requests_total[5m]))) > 0.2
      for: 10m
      labels:
        severity: warning
        team: vision
      annotations:
        summary: "Vision input rejection ratio high"
        description: "More than 20% of Vision requests rejected by validation in the last 10m."
        runbook_url: https://example.org/runbooks/input_rejections_spike

    # 5) Provider down (OCR)
    - alert: OcrProviderDown
      expr: sum by(provider) (rate(ocr_errors_total{code="provider_down"}[5m])) > 0
      for: 2m
      labels:
        severity: critical
        team: ocr
      annotations:
        summary: "OCR provider down"
        description: "Provider reports provider_down errors. Check credentials, rate limits, or provider status."
        runbook_url: https://example.org/runbooks/provider_down

    # 6) Circuit breaker open (OCR)
    - alert: OcrCircuitOpen
      expr: ocr_circuit_state == 2
      for: 2m
      labels:
        severity: warning
        team: ocr
      annotations:
        summary: "OCR circuit breaker open"
        description: "Circuit state open (2). Investigate upstream failures and error rates."
        runbook_url: https://example.org/runbooks/provider_down

    # 7) Oversized images surge (Vision)
    - alert: VisionOversizedImagesSurge
      expr: sum by() (rate(vision_input_rejected_total{reason="base64_too_large"}[5m])) > 1
      for: 10m
      labels:
        severity: warning
        team: vision
      annotations:
        summary: "Vision oversized base64 images surge"
        description: "Clients are sending images larger than allowed. Consider communicating limits or adjusting VISION_MAX_BASE64_BYTES."
        runbook_url: https://example.org/runbooks/input_rejections_spike

    # 8) Large payloads distribution shift (Vision)
    - alert: VisionImageSizeP99High
      expr: histogram_quantile(0.99, rate(vision_image_size_bytes_bucket[5m])) > 2000000
      for: 15m
      labels:
        severity: info
        team: vision
      annotations:
        summary: "Vision image size P99 high"
        description: "P99 input size >2MB. Monitor downstream latency and costs."
        runbook_url: https://example.org/runbooks/input_rejections_spike

    # 9) Compare failure rate elevated
    - alert: CompareRequestFailureRateHigh
      expr: |
        (
          sum(rate(compare_requests_total{status!="success"}[5m]))
          / sum(rate(compare_requests_total[5m]))
        ) > 0.3
        and sum(rate(compare_requests_total[5m])) > 0.05
      for: 10m
      labels:
        severity: warning
        team: compare
      annotations:
        summary: "High /api/compare failure rate"
        description: "More than 30% of /api/compare requests are failing over 10m."
        runbook_url: https://example.org/runbooks/compare_failure_rate

    # 10) Compare not_found dominates
    - alert: CompareNotFoundDominant
      expr: |
        (
          sum(rate(compare_requests_total{status="not_found"}[5m]))
          / sum(rate(compare_requests_total[5m]))
        ) > 0.5
        and sum(rate(compare_requests_total[5m])) > 0.05
      for: 10m
      labels:
        severity: warning
        team: compare
      annotations:
        summary: "Compare not_found dominates"
        description: "More than 50% of /api/compare requests are not_found over 10m."
        runbook_url: https://example.org/runbooks/compare_not_found
```

Notes:
- Tune thresholds for your traffic. For small volumes, prefer longer `for:` windows.
- Replace `example.org` with your doc host or Git URL for runbooks.
- Consider SLO-based alerting (error budget burn) when traffic is large enough.

## Integrations

- GitHub Actions: use `curl` in post-failure hooks to send alerts to Slack/PagerDuty.
- Slack: route Vision alerts to `#vision-alerts`, OCR alerts to `#ocr-alerts`.
- PagerDuty: map severity critical to P1, warning to P2/P3.
