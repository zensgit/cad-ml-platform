# DEV_MECH_KNOWLEDGE_4000CAD_GRAPH2D_DEFAULTS_20260120

## Summary
Aligned Graph2D defaults to the merged-label manifest and merged checkpoint for
training, knowledge-rule generation, and runtime fallback.

## Updates
- `scripts/train_2d_graph.py`: default manifest -> merged manifest; default output -> `models/graph2d_merged_latest.pth`.
- `scripts/build_geometry_rules_from_manifest.py`: default manifest -> merged manifest.
- `scripts/build_label_synonyms_template.py`: default manifest -> merged manifest.
- `src/ml/vision_2d.py`: fallback model path -> `models/graph2d_merged_latest.pth`.

## Context
The merged-label manifest is the baseline for the 4000CAD Graph2D experiments
and aligns the training defaults with the current checkpoint inventory.
