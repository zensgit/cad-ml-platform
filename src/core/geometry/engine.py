"""
Geometry Engine.

Handles 3D CAD file parsing, B-Rep analysis, and topological feature extraction.
Uses pythonocc-core (OpenCascade) as the kernel.
"""

import io
import logging
import math
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Conditional import to allow running in environments without OCC
try:
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.GeomAbs import (
        GeomAbs_BezierSurface,
        GeomAbs_BSplineSurface,
        GeomAbs_Cone,
        GeomAbs_Cylinder,
        GeomAbs_Plane,
        GeomAbs_Sphere,
        GeomAbs_Torus,
    )
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID, TopAbs_VERTEX
    from OCC.Core import TopExp
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopTools import (
        TopTools_IndexedDataMapOfShapeListOfShape,
        TopTools_IndexedMapOfShape,
        TopTools_ListIteratorOfListOfShape,
    )
    from OCC.Core.TopoDS import TopoDS_Shape

    HAS_OCC = True
except ImportError as exc:
    HAS_OCC = False
    logger.warning(
        "pythonocc-core not found. 3D analysis capabilities will be limited to mock/fallback. (%s)",
        exc,
    )


BREP_GRAPH_NODE_FEATURES = (
    "plane",
    "cylinder",
    "cone",
    "sphere",
    "torus",
    "bezier",
    "bspline",
    "other",
    "area",
    "bbox_x",
    "bbox_y",
    "bbox_z",
    "normal_x",
    "normal_y",
    "normal_z",
)

BREP_GRAPH_EDGE_FEATURES = ("dihedral_angle", "convexity")


class GeometryEngine:
    """
    3D Geometry Analysis Engine.
    Wraps OpenCascade functionality to extract semantic features from STEP/IGES files.
    """

    def __init__(self):
        if not HAS_OCC:
            logger.warning("GeometryEngine initialized without OCC kernel.")

    def load_step(self, content: bytes, file_name: str = "temp.step") -> Optional[Any]:
        """
        Load a STEP file from bytes.
        Returns a TopoDS_Shape or None on failure.
        """
        if not HAS_OCC:
            return None

        # OCC requires a file path usually, so we write to temp
        suffix = os.path.splitext(file_name)[1]
        if not suffix:
            suffix = ".step"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            reader = STEPControl_Reader()
            status = reader.ReadFile(tmp_path)

            if status != IFSelect_RetDone:
                logger.error(f"Error reading STEP file {file_name}: status {status}")
                return None

            reader.TransferRoots()
            shape = reader.OneShape()
            return shape
        except Exception as e:
            logger.error(f"Exception loading STEP file: {e}")
            return None
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def extract_brep_features(self, shape: Any) -> Dict[str, Any]:
        """
        Extract B-Rep statistics and topological features from a shape.
        """
        features = {
            "valid_3d": False,
            "volume": 0.0,
            "surface_area": 0.0,
            "faces": 0,
            "edges": 0,
            "vertices": 0,
            "solids": 0,
            "shells": 0,
            "surface_types": {},
            "bbox": {"x": 0, "y": 0, "z": 0},
            "is_assembly": False,
        }

        if not HAS_OCC or shape is None:
            return features

        try:
            features["valid_3d"] = True

            # Topological counts
            features["faces"] = self._count_subshapes(shape, TopAbs_FACE)
            features["edges"] = self._count_subshapes(shape, TopAbs_EDGE)
            features["vertices"] = self._count_subshapes(shape, TopAbs_VERTEX)
            features["solids"] = self._count_subshapes(shape, TopAbs_SOLID)
            features["shells"] = self._count_subshapes(shape, TopAbs_SHELL)

            if features["solids"] > 1:
                features["is_assembly"] = True

            # Physical properties
            gprops = GProp_GProps()
            brepgprop.VolumeProperties(shape, gprops)
            features["volume"] = gprops.Mass()

            brepgprop.SurfaceProperties(shape, gprops)
            features["surface_area"] = gprops.Mass()

            # Surface type histogram (Semantic feature)
            features["surface_types"] = self._analyze_surface_types(shape)

            # Bounding box
            bbox = Bnd_Box()
            brepbndlib.Add(shape, bbox)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
            features["bbox"] = {
                "x": xmax - xmin,
                "y": ymax - ymin,
                "z": zmax - zmin,
                "diag": ((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2) ** 0.5,
            }

        except Exception as e:
            logger.error(f"Error extracting BREP features: {e}")
            features["error"] = str(e)

        return features

    def extract_brep_graph(self, shape: Any) -> Dict[str, Any]:
        """Extract a face adjacency graph with node and edge features."""
        graph = {
            "valid_3d": False,
            "graph_schema_version": "v1",
            "node_schema": BREP_GRAPH_NODE_FEATURES,
            "edge_schema": BREP_GRAPH_EDGE_FEATURES,
            "node_count": 0,
            "edge_count": 0,
            "node_features": [],
            "edge_index": [],
            "edge_features": [],
        }

        if not HAS_OCC or shape is None:
            return graph

        try:
            face_map = TopTools_IndexedMapOfShape()
            TopExp.MapShapes(shape, TopAbs_FACE, face_map)
            faces = []
            node_features = []
            for i in range(1, face_map.Extent() + 1):
                face = face_map(i)
                faces.append(face)
                node_features.append(self._face_feature_vector(face))

            edge_index = []
            edge_features = []
            edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
            TopExp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)
            for i in range(1, edge_face_map.Extent() + 1):
                face_indices = []
                face_list = edge_face_map.FindFromIndex(i)
                it = TopTools_ListIteratorOfListOfShape(face_list)
                while it.More():
                    face = it.Value()
                    face_idx = face_map.FindIndex(face)
                    if face_idx > 0:
                        face_indices.append(face_idx - 1)
                    it.Next()

                if len(face_indices) < 2:
                    continue

                a_idx, b_idx = face_indices[0], face_indices[1]
                edge_feat = self._edge_feature_vector(faces[a_idx], faces[b_idx])
                edge_index.extend([[a_idx, b_idx], [b_idx, a_idx]])
                edge_features.extend([edge_feat, edge_feat])

            graph["valid_3d"] = True
            graph["node_features"] = node_features
            graph["edge_index"] = edge_index
            graph["edge_features"] = edge_features
            graph["node_count"] = len(node_features)
            graph["edge_count"] = len(edge_features)

        except Exception as e:
            logger.error(f"Error extracting BREP graph: {e}")
            graph["error"] = str(e)

        return graph

    def extract_dfm_features(self, shape: Any) -> Dict[str, Any]:
        """
        Extract features specifically for Design for Manufacturability (DFM) analysis.
        (L4 Capability)
        """
        dfm_features = {
            "thin_walls_detected": False,
            "min_thickness_estimate": 0.0,
            "undercuts_detected": False,  # Requires ray tracing, placeholder
            "sharp_edges_count": 0,
            "machinability_score": 1.0,  # 0.0 - 1.0
        }

        if not HAS_OCC or shape is None:
            return dfm_features

        try:
            # 1. Bounding Box Aspect Ratio (Stock size estimation)
            bbox = Bnd_Box()
            brepbndlib.Add(shape, bbox)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
            dims = sorted([xmax - xmin, ymax - ymin, zmax - zmin])
            if dims[0] > 0:
                dfm_features["aspect_ratio_max_min"] = dims[2] / dims[0]
            else:
                dfm_features["aspect_ratio_max_min"] = 0.0

            # 2. Volume to BBox Volume Ratio (Material removal rate)
            bbox_vol = dims[0] * dims[1] * dims[2]
            gprops = GProp_GProps()
            brepgprop.VolumeProperties(shape, gprops)
            actual_vol = gprops.Mass()

            if bbox_vol > 0:
                dfm_features["stock_removal_ratio"] = 1.0 - (actual_vol / bbox_vol)

            # 3. Heuristic Thin Wall Detection
            # Real implementation uses ray casting or medial axis transform.
            # Here we use a heuristic: Surface Area / Volume ratio.
            # Very high ratio implies thin sheets or complex lattices.
            if actual_vol > 0:
                sa_vol_ratio = self.extract_brep_features(shape)["surface_area"] / actual_vol
                if sa_vol_ratio > 2.0:
                    # Threshold depends on unit system (assuming mm)
                    dfm_features["thin_walls_detected"] = True
                    dfm_features["min_thickness_estimate"] = 2.0 / sa_vol_ratio

            # 4. Sharp Edge Detection (Stress risers)
            # Iterate edges, check continuity (C0 vs G1)
            # Placeholder for complex topology traversal
            dfm_features["sharp_edges_count"] = self._count_subshapes(
                shape,
                TopAbs_EDGE,
            )

        except Exception as e:
            logger.error(f"Error extracting DFM features: {e}")

        return dfm_features

    def _face_feature_vector(self, face: Any) -> List[float]:
        surf = BRepAdaptor_Surface(face, True)
        one_hot = self._surface_type_one_hot(surf.GetType())

        gprops = GProp_GProps()
        brepgprop.SurfaceProperties(face, gprops)
        area = gprops.Mass()

        bbox = Bnd_Box()
        brepbndlib.Add(face, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        dims = [xmax - xmin, ymax - ymin, zmax - zmin]

        normal = self._face_normal_vector(surf)

        return one_hot + [area] + dims + list(normal)

    def _surface_type_one_hot(self, surface_type: Any) -> List[float]:
        mapping = {
            GeomAbs_Plane: 0,
            GeomAbs_Cylinder: 1,
            GeomAbs_Cone: 2,
            GeomAbs_Sphere: 3,
            GeomAbs_Torus: 4,
            GeomAbs_BezierSurface: 5,
            GeomAbs_BSplineSurface: 6,
        }
        vector = [0.0] * 8
        index = mapping.get(surface_type, 7)
        vector[index] = 1.0
        return vector

    def _face_normal_vector(self, surf: Any) -> Tuple[float, float, float]:
        if surf.GetType() != GeomAbs_Plane:
            return 0.0, 0.0, 0.0

        plane = surf.Plane()
        direction = plane.Axis().Direction()
        return direction.X(), direction.Y(), direction.Z()

    def _edge_feature_vector(self, face_a: Any, face_b: Any) -> List[float]:
        normal_a = self._face_normal_vector(BRepAdaptor_Surface(face_a, True))
        normal_b = self._face_normal_vector(BRepAdaptor_Surface(face_b, True))

        if normal_a == (0.0, 0.0, 0.0) or normal_b == (0.0, 0.0, 0.0):
            return [0.0, 0.0]

        dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(normal_a, normal_b))))
        angle = math.acos(dot)
        convexity = 1.0 if dot >= 0.0 else -1.0
        return [angle, convexity]

    def _count_subshapes(self, shape: Any, shape_type: Any) -> int:
        count = 0
        exp = TopExp_Explorer(shape, shape_type)
        while exp.More():
            count += 1
            exp.Next()
        return count

    def _analyze_surface_types(self, shape: Any) -> Dict[str, int]:
        """Classify faces by surface geometry (Plane, Cylinder, Spline, etc.)"""
        types = {
            "plane": 0,
            "cylinder": 0,
            "cone": 0,
            "sphere": 0,
            "torus": 0,
            "bezier": 0,
            "bspline": 0,
            "other": 0,
        }

        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            face = exp.Current()
            surf = BRepAdaptor_Surface(face, True)  # True = restriction
            st = surf.GetType()

            if st == GeomAbs_Plane:
                types["plane"] += 1
            elif st == GeomAbs_Cylinder:
                types["cylinder"] += 1
            elif st == GeomAbs_Cone:
                types["cone"] += 1
            elif st == GeomAbs_Sphere:
                types["sphere"] += 1
            elif st == GeomAbs_Torus:
                types["torus"] += 1
            elif st == GeomAbs_BezierSurface:
                types["bezier"] += 1
            elif st == GeomAbs_BSplineSurface:
                types["bspline"] += 1
            else:
                types["other"] += 1

            exp.Next()

        return types

    def tessellate(
        self,
        shape: Any,
        deflection: float = 0.1,
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """
        Tessellate shape to mesh (vertices, faces) for visualization or UV-Net processing if needed.
        """
        if not HAS_OCC or shape is None:
            return [], []

        BRepMesh_IncrementalMesh(shape, deflection)

        vertices = []
        triangles = []

        # This is a simplified extraction, a full one would iterate faces and triangulation
        # Placeholder for complex mesh extraction logic needed for PointNet/UV-Net inputs

        return vertices, triangles


# Singleton
_engine = GeometryEngine()


def get_geometry_engine():
    return _engine
