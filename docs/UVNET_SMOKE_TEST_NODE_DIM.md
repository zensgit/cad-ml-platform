#!/usr/bin/env markdown
# UV-Net Smoke Test Node Dimension Alignment

## Goal
Keep the UV-Net smoke test aligned with the graph node feature schema so the
resulting checkpoint matches production data.

## Approach
- Import `BREP_GRAPH_NODE_FEATURES` from `src.core.geometry.engine`.
- Derive `DEFAULT_NODE_DIM = len(BREP_GRAPH_NODE_FEATURES)` with a fallback.
- Use the derived dimension for both the mock dataset and model initialization.

## Rationale
This prevents mismatched input dimensions between training scripts and inference
pipelines, avoiding shape errors during checkpoint loads.
