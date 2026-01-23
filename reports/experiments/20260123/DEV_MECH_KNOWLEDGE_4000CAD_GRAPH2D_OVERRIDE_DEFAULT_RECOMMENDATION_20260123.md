# DEV_MECH_KNOWLEDGE_4000CAD_GRAPH2D_OVERRIDE_DEFAULT_RECOMMENDATION_20260123

## Summary
- Chose Graph2D override default threshold 0.6 based on the 50-sample comparison.
- Documented 0.5 as an experimental option for higher coverage.

## Decision
- Default: `FUSION_GRAPH2D_OVERRIDE_MIN_CONF=0.6`
- Experimental: `FUSION_GRAPH2D_OVERRIDE_MIN_CONF=0.5`

## Config Update
- `.env.example`
