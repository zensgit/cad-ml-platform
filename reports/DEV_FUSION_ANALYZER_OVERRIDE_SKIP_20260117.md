# DEV_FUSION_ANALYZER_OVERRIDE_SKIP_20260117

## Summary
Validated the FusionAnalyzer override skip path when the fused confidence is below the
minimum threshold.

## Steps
- Started API with override enabled:
  `FUSION_ANALYZER_ENABLED=true FUSION_ANALYZER_OVERRIDE=true FUSION_ANALYZER_OVERRIDE_MIN_CONF=0.8 uvicorn src.main:app --port 8001`
- Posted analysis request:
  `curl -fsS -X POST http://127.0.0.1:8001/api/v1/analyze/ -H 'X-API-Key: test' \
    -F 'file=@examples/sample_part.step' \
    -F 'options={"extract_features": false, "classify_parts": true, "quality_check": false, "process_recommendation": false, "calculate_similarity": false, "estimate_cost": false}'`

## Results
- Classification was NOT overridden; `fusion_override_skipped` reported:
  - `min_confidence`: `0.8`
  - `decision_confidence`: `0.5`

## Notes
- `fusion_decision` still returned in the response for inspection.
