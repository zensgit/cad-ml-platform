# DEV_BREP_GRAPH_EXTRACTION_20260117

## Summary
Implemented a face-adjacency graph extraction API for B-Rep shapes, including node and edge
features for GNN pipelines.

## Design
- Doc: `docs/BREP_GRAPH_EXTRACTION_DESIGN.md`

## Steps
- Added graph schema constants and a new `extract_brep_graph` API in the geometry engine.
- Added integration coverage for graph extraction using pythonocc-generated primitives.
- Ran: `pytest tests/integration/test_brep_graph_extraction.py -v`.

## Results
- Test skipped locally because `pythonocc-core` is not installed.

## Notes
- Graph schema v1 includes surface type one-hot, area, bbox extents, and planar normals.
- Edge features include approximate dihedral angle and convexity based on planar normals.
- Linux/amd64 validation attempt recorded in `reports/DEV_BREP_GRAPH_LINUX_AMD64_VALIDATION_20260117.md`.
