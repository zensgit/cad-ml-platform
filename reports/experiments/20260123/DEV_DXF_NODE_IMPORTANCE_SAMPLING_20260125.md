# DEV_DXF_NODE_IMPORTANCE_SAMPLING_20260125

## Objective
Replace the hard-coded DXF node truncation with configurable importance sampling to preserve critical entities (text, borders, key geometry) in complex drawings.

## Implementation
- `src/ml/train/dataset_2d.py`
  - Replaced fixed `max_nodes=50` truncation with `ImportanceSampler` usage.
  - Sampling respects environment flags:
    - `DXF_MAX_NODES`
    - `DXF_SAMPLING_STRATEGY` (importance|random|hybrid)
    - `DXF_SAMPLING_SEED`
    - `DXF_TEXT_PRIORITY_RATIO`

## Sampling Priority (from `src/ml/importance_sampler.py`)
1. TEXT/MTEXT/DIMENSION
2. Title-block region
3. Border entities
4. CIRCLE / ARC
5. Long lines
6. Other entities

## Notes
- Importance sampling is deterministic under a fixed seed.
- Sampling logic is applied both in training and inference when using `_dxf_to_graph`.
