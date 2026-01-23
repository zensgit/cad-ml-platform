# BREP_GRAPH_DATASET_DESIGN

## Goal
Extend the ABC dataset loader to emit graph-structured samples for GNN training while keeping
legacy numeric features for the existing PointNet scaffold.

## Dataset Output Modes
- `output_format="numeric"` (default): returns the existing 12x1024 tensor.
- `output_format="graph"`: returns a graph sample built from `GeometryEngine.extract_brep_graph`.

## Graph Backends
- `graph_backend="auto"` (default): use PyTorch Geometric if available, otherwise return a dict.
- `graph_backend="pyg"`: require PyG; fall back to dict with a warning if unavailable.
- `graph_backend="dict"`: always return a plain dict with tensors.

## Label Handling
- PyG backend: attaches `y` to the Data object and returns the Data instance directly.
- Dict backend: returns `(sample, label)` tuples.

## Graph Tensor Layout
- `x`: node features, shape `(num_nodes, node_dim)`
- `edge_index`: edge indices, shape `(2, num_edges)`
- `edge_attr`: edge features, shape `(num_edges, edge_dim)`
- `graph_schema_version`, `node_schema`, `edge_schema`: metadata for alignment

## Failure Handling
- Missing or invalid shapes return empty graph tensors with the correct feature dimensions.
- Errors are logged but do not crash the dataset loader.
