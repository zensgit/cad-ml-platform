# DEV_MECH_KNOWLEDGE_2D_GRAPH_VALIDATION_20260119

## Summary
Validated dataset-derived geometry rules and verified DXF graph extraction using the
converted training drawings.

## Environment
- Python: /opt/homebrew/opt/python@3.13/bin/python3.13
- Graph runtime: `./.venv-graph/bin/python`

## Tests
- `pytest tests/unit/test_geometry_rules_dataset.py -v`

## Validation Steps
- DXF graph extraction dry-run (first converted drawing):
  `./.venv-graph/bin/python - <<'PY' ... DXFDataset._dxf_to_graph ...`

## Results
- Geometry rules test suite: 2 passed.
- DXF graph extraction produced a non-empty graph:
  - file: `BTJ01230901522-00汽水分离器v1.dxf`
  - nodes: (50, 7)
  - edges: (2, 92)
- Note: system Python without torch cannot execute the graph extraction; use
  `.venv-graph` or install torch to run the dry-run locally.
