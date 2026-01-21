# DEV_MECH_KNOWLEDGE_4000CAD_FUSION_API_UPDATE_20260121

## Summary
- Wired FusionAnalyzer to reuse L1/L2/L3 signals already computed during classification.
- Added `fusion_inputs` to the classification payload for explainability.
- Documented fusion toggles in `.env.example`.

## Files Updated
- `src/api/v1/analyze.py`
- `.env.example`

## Implementation Notes
- L3 features passed to fusion inputs omit `embedding_vector` to keep payloads lightweight.
- Fusion inputs include `l1` (doc metadata), `l2` (2D feature stats), `l3` (3D stats), and `l4` (Graph2D or ML prediction when enabled).

## Config Toggles
- `FUSION_ANALYZER_ENABLED`
- `GRAPH2D_FUSION_ENABLED`
- `FUSION_ANALYZER_OVERRIDE`
- `FUSION_ANALYZER_OVERRIDE_MIN_CONF`
