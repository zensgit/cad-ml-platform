# DEV_MECH_KNOWLEDGE_4000CAD_SYNTHETIC_DXF_AUGMENTATION_20260121

## Summary
- Generated a synthetic DXF dataset for 2D graph augmentation.
- Captured feature distribution statistics for plates, holes, and slots.

## Outputs
- Synthetic dataset: `data/synthetic_dxf/`
- Labels manifest: `data/synthetic_dxf/labels.json`

## Key Stats
- Files generated: 20
- Feature counts: plate=20, hole=64, slot=7
- Avg holes per file: 3.2
- Files containing slots: 7

## Commands
- `python3 scripts/generate_synthetic_dxf_dataset.py`

## Notes
- Output dataset remains local-only (not tracked in git).
