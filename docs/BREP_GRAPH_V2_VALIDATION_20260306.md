# B-Rep Graph V2

Date: 2026-03-06

## Goal

Upgrade the existing `extract_brep_graph()` path from a minimal adjacency graph to
a richer graph payload that is closer to the `UV-Net` / `AAGNet` direction without
pulling in their full training stack.

## Changes

### Node Schema

Existing surface-type one-hot and area/bbox/normal features are retained.

Added:

- `bbox_diag`
- `center_x`
- `center_y`
- `center_z`
- `normal_defined`

### Edge Schema

Existing:

- `dihedral_angle`
- `convexity`

Added:

- `shared_edge_length`
- `same_surface_type`
- `angle_defined`

### Graph Metadata

Added `graph_metadata` with:

- `undirected_edge_count`
- `directed_edge_count`
- `undirected_edge_index`
- `surface_type_histogram`
- `bbox`

## Compatibility

- The dataset still returns `x`, `edge_index`, and `edge_attr`.
- Schema-aware models can infer dimensions from `node_schema` and `edge_schema`.
- Existing trained checkpoints tied to the previous schema should be treated as
  incompatible with `graph_schema_version=v2`.

## Validation

- `tests/unit/test_graph_dataset_output.py`
- `tests/integration/test_brep_graph_extraction.py`

## Next Steps

1. Add optional feature standardization statistics for node and edge attributes.
2. Expose a graph extraction mode that emits UV-grids for faces and coedge-aware
   adjacency for research experiments.
3. Add offline graph caching for STEP training corpora to avoid reparsing on
   every epoch.
