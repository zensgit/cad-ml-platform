# Batch J — Instrumentation Enrichment (Logging + Metrics fields)

Status: completed

Scope
- Enrich structured logging fields emitted by `/api/v1/ocr/extract` and `OcrManager`.
- Keep metrics as-is; ensure logs include enough context for debugging, correlation, and SLI/SLO analysis.

Changes
- `src/utils/logging.py`: Json formatter now forwards extra fields:
  - provider, image_hash, latency_ms, fallback_level, error_code, stage
  - trace_id, extraction_mode, completeness, calibrated_confidence
  - dimensions_count, symbols_count, stages_latency_ms
- `src/api/v1/ocr.py`: emit `ocr.extract` log with the new fields.
- `src/core/ocr/manager.py`: emit `ocr.manager.extract` with the same field set.

Rationale
- Trace-level visibility: With `trace_id` and `image_hash` we can correlate requests across API, manager, and providers without logging sensitive content.
- Debuggability: `extraction_mode` and per-stage latencies help isolate bottlenecks or fallback reasons.
- SLOs: `latency_ms` aligns with `ocr_processing_duration_seconds` histogram for dual-view analysis.

Acceptance
- Logs are valid JSON and include the enriched fields.
- No sensitive content (raw text or images) is logged; only hashes and counts.
- All tests pass (no behavior change).

Next
- Expand provider-level logs (per stage) if needed and wire sampling for high-QPS scenarios.

