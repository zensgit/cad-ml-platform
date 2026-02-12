# Analyze Classification Fields

This document describes the `results.classification` payload returned by `/api/v1/analyze/`.

## Primary Label (Stable)

These fields represent the primary, stable decision path (rules/Fusion). They are the only fields that should be treated as "production truth" today.

- `part_type`: canonical part type (e.g. `bolt`)
- `confidence`: confidence for `part_type` (0..1)
- `confidence_source`: where `part_type` came from (e.g. `rules`, `fusion`)
- `rule_version`: ruleset / decision version (e.g. `v1`, `L2-Fusion-v1`)

## Fine Label (Additive)

When HybridClassifier is enabled, it can emit a fine-grained label without overriding `part_type`:

- `fine_part_type`
- `fine_confidence`
- `fine_source`
- `fine_rule_version`

These fields are intended for UI/analysis, not as a hard override of `part_type`.

## Shadow Part Family (Provider, Additive)

When enabled, `/api/v1/analyze/` can also run the in-process PartClassifier via the provider framework in **shadow-only** mode:

- Raw provider payload:
  - `part_classifier_prediction`: dict (provider-specific fields)
- Normalized coarse fields (stable shape):
  - `part_family`: coarse label (string) or `null`
  - `part_family_confidence`: 0..1 or `null`
  - `part_family_source`: string, currently `provider:<name>`
  - `part_family_model_version`: optional string
  - `part_family_needs_review`: optional bool
  - `part_family_review_reason`: optional string
  - `part_family_top2`: optional dict `{label, confidence}`
  - `part_family_error`: present only when normalization fails or provider status is not ok:
    - `{code, message}`

Important behavior:
- This does **not** override `part_type`.
- It is gated behind env flags and format checks (DXF/DWG by default).

### Enablement / Ops Flags

Environment variables controlling the shadow invocation:

- `PART_CLASSIFIER_PROVIDER_ENABLED` (default `false`)
- `PART_CLASSIFIER_PROVIDER_NAME` (default `v16`)
- `PART_CLASSIFIER_PROVIDER_SHADOW_FORMATS` (default `dxf,dwg`)
- `PART_CLASSIFIER_PROVIDER_TIMEOUT_SECONDS` (default `2.0`)
- `PART_CLASSIFIER_PROVIDER_MAX_MB` (default `10.0`)
- `PART_CLASSIFIER_PROVIDER_INCLUDE_IN_CACHE_KEY` (default `true`)

### Expected Provider Status Values

`part_classifier_prediction.status` may be one of:
- `ok`
- `timeout`
- `error`
- `file_too_large`
- `unavailable` / `no_prediction` / `model_unavailable` (provider-specific)

When `status != "ok"`, `part_family` remains `null` and `part_family_error` is emitted.

## Metrics

Prometheus metrics emitted by the shadow integration:

- `analysis_part_classifier_requests_total{status,provider}`
- `analysis_part_classifier_seconds{provider}`
- `analysis_part_classifier_skipped_total{reason}`

