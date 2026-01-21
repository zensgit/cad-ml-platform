# DEV_MECH_KNOWLEDGE_4000CAD_DEFAULT_MODEL_20260120

## Summary
Set the recommended default graph2d model to the full checkpoint based on A/B
results and documented the environment variables for enabling Graph2D.

## Decision
- Default model: `models/graph2d_latest.pth`
- Rationale: full checkpoint edges out freq2 on Top-1 accuracy while matching Top-3.

## Updates
- Added Graph2D env settings to `.env.example`.

## Reference
- A/B report: `reports/DEV_MECH_KNOWLEDGE_4000CAD_AB_EVAL_20260120.md`
