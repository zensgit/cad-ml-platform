# FUSION_ANALYZER_DESIGN

## Goal
Provide a single fusion entry point that combines L1/L2/L3 signals with L4 model output, while
preserving explainability and deterministic fallback behavior.

## Inputs
- L1: `doc_metadata` (format, basic metadata, validity flags)
- L2: `l2_features` (2D projection statistics such as aspect_ratio)
- L3: `l3_features` (B-Rep/physics signals)
- L4: `l4_prediction` (model label + confidence)

## Output
`FusionDecision` (see `src/core/knowledge/fusion_contracts.py`) includes:
- `primary_label`
- `confidence`
- `source` (rule_based, ai_model, hybrid)
- `reasons`, `rule_hits`
- `consistency_check` + `consistency_notes`
- `schema_version`

## MVP Logic (v1)
1) L1 guardrails: invalid format -> rule-based fallback.
2) Consistency check: if AI label conflicts with geometric indicators (e.g., Slot with low
   aspect ratio), mark conflict.
3) Decision:
   - AI dominates when confidence >= threshold and no conflict.
   - Otherwise fall back to rule-based heuristics.

## Notes
- This MVP does not yet apply feature normalization; `DEFAULT_NORM_SCHEMA` is reserved for the
  next iteration.
- `FusionAnalyzer` is wired behind feature flags in `/api/v1/analyze`.

## Integration Flags
- `FUSION_ANALYZER_ENABLED=true`: compute a fusion decision and attach it to classification output.
- `FUSION_ANALYZER_OVERRIDE=true`: replace the classification label/confidence with the fusion
  decision (requires `FUSION_ANALYZER_ENABLED=true`).
