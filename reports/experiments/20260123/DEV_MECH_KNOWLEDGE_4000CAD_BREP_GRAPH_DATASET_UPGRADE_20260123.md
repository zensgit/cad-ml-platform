# DEV_MECH_KNOWLEDGE_4000CAD_BREP_GRAPH_DATASET_UPGRADE_20260123

## Summary
- Expanded B-Rep graph extraction to emit adjacency edges for all face pairs on shared edges.
- Ensured graph datasets attach labels directly to PyG Data objects when PyG is enabled.
- Documented label handling behavior for graph backends.

## Changes
- `src/core/geometry/engine.py` (multi-face edge adjacency handling)
- `src/ml/train/dataset.py` (PyG label attachment)
- `docs/BREP_GRAPH_DATASET_DESIGN.md` (label handling notes)

## Notes
- Graph schema remains v1; node and edge feature ordering unchanged.
- Dict backend continues to return `(sample, label)` tuples for non-PyG workflows.
