# BREP_GRAPH_EXTRACTION_DESIGN

## Goal
Provide a face-adjacency graph representation of B-Rep shapes for GNN-based learning.
The graph treats faces as nodes and shared edges as connections.

## Graph Schema (v1)
Node features (order matches `BREP_GRAPH_NODE_FEATURES`):
- Surface type one-hot: plane, cylinder, cone, sphere, torus, bezier, bspline, other
- Face area
- Face bounding box extents (x, y, z)
- Face normal (x, y, z) when planar; zeros otherwise

Edge features (order matches `BREP_GRAPH_EDGE_FEATURES`):
- Dihedral angle (radians) based on planar face normals
- Convexity flag derived from the normal dot product (1.0 convex, -1.0 concave)

## Extraction Approach
- Enumerate faces via `TopExp.MapShapes` and build a stable index.
- Compute per-face features from `BRepAdaptor_Surface`, `brepgprop.SurfaceProperties`, and
  `Bnd_Box` bounds.
- Map edges to adjacent faces using `TopExp.MapShapesAndAncestors`, then emit directed edges
  for each face pair.

## Limitations
- Normals and dihedral angles are approximated for planar faces only; non-planar faces return
  zero normals and edge features.
- Convexity is a coarse heuristic derived from the normal dot product, not a full topological
  concave/convex analysis.

## Follow-ups
- Compute normals for curved surfaces via `GeomLProp_SLProps` or face param sampling.
- Improve convexity detection using edge tangents and oriented face normals.
- Extend edge features with curvature or edge length for richer GNN inputs.
