# DEV_BREP_GRAPH_DATASET_20260117

## Summary
Extended the ABC dataset loader to emit graph-structured samples for B-Rep GNN training while
preserving the legacy numeric tensor output.

## Design
- Doc: `docs/BREP_GRAPH_DATASET_DESIGN.md`

## Steps
- Added `output_format` and `graph_backend` options to `ABCDataset`.
- Converted `extract_brep_graph` output to tensors and optional PyG `Data` objects.
- Added unit coverage for dict-based graph output.
- Ran: `pytest tests/unit/test_graph_dataset_output.py -v`.

## Results
- Test passed in a local Python 3.11 virtualenv after installing `torch`, `numpy`, and
  `pydantic-settings` (required by repo test fixtures).

## Notes
- Graph output defaults to a dict unless `torch_geometric` is available.
- Empty graphs are returned for invalid shapes to keep batch collation stable.
- Virtualenv used: `.venv-graph`.
