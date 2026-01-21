# DEV_MECH_KNOWLEDGE_4000CAD_VALIDATION_20260119

## Summary
Validated the graph2d integration path after applying a high-confidence
auto-label filter for the 4000CAD dataset slice.
OCR was configured with PaddleOCR (paperspace + model render), but no text was
extracted from the rendered DXFs.

## Tests
- `GRAPH2D_ENABLED=true GRAPH2D_MODEL_PATH=models/graph2d_latest.pth pytest tests/integration/test_analyze_dxf_fusion.py -v`
  - Result: 1 passed

## Conversion Validation
- Previous log: `reports/MECH_4000_DWG_TO_DXF_LOG_20260119.csv` (50/50 conversions). Not re-run.

## Graph2D Weak-Label Baseline
- Not re-run after the high-confidence filter (previous weak-label metrics remain).

## Manual-Eval Baseline (Auto-Filled)
- Not re-run (auto-filled reviewer labels remain as placeholders).

## Auto-Label Attempt
- Unlabeled rows: 27
- Auto-labeled rows (>= 0.7 confidence): 15
- Cleared for review: 12
- Report: `reports/DEV_MECH_KNOWLEDGE_4000CAD_AUTO_LABEL_ATTEMPT_20260119.md`
