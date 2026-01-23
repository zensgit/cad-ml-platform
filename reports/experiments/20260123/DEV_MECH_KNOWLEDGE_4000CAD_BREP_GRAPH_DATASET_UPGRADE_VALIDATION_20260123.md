# DEV_MECH_KNOWLEDGE_4000CAD_BREP_GRAPH_DATASET_UPGRADE_VALIDATION_20260123

## Checks
- Confirmed graph dataset output shapes and schema metadata using STEP fixtures.
- Verified dict backend returns empty graph tensors with correct dimensions when OCC is unavailable.

## Runtime Output
- Command:
  - `python3 - <<'PY'
from src.ml.train.dataset import ABCDataset

dataset = ABCDataset("tests/fixtures", output_format="graph", graph_backend="dict")
print("files", len(dataset))
sample, label = dataset[0]
print("x_shape", tuple(sample["x"].shape))
print("edge_index_shape", tuple(sample["edge_index"].shape))
print("edge_attr_shape", tuple(sample["edge_attr"].shape))
print("graph_schema_version", sample.get("graph_schema_version"))
print("label", label)
PY`
- Result:
  - `files 3`
  - `x_shape (0, 15)`
  - `edge_index_shape (2, 0)`
  - `edge_attr_shape (0, 2)`
  - `graph_schema_version v1`
  - `label 5`

## Notes
- `pythonocc-core` is not available in this environment, so graph extraction returned empty tensors.
- PyG backend validation is deferred until a PyTorch Geometric environment is available.
