# DEV_MECH_KNOWLEDGE_4000CAD_GRAPH2D_DEFAULT_MODEL_SWITCH_20260122

## Summary
- Switched default Graph2D model path to the parts-upsampled checkpoint.
- Updated env defaults to point to `graph2d_parts_upsampled_20260122.pth`.

## Code Changes
- `.env.example` (GRAPH2D_MODEL_PATH)
- `src/ml/vision_2d.py` (default fallback path)
- `src/main.py` (startup validation default path)

## Notes
- Ensure the model file exists in deployment environments.
