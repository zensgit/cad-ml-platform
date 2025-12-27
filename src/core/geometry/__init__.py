"""Advanced geometric feature extraction using Open3D.

This module provides rotation-invariant feature extraction for CAD meshes,
enhancing the basic features in src/core/feature_extractor.py.

Benefits over current implementation:
- FPFH descriptors (rotation-invariant)
- PCA-based canonical alignment
- Consistent point cloud sampling
- Industry-standard algorithms

Example:
    >>> from src.core.geometry import GeometricFeatureExtractor
    >>> extractor = GeometricFeatureExtractor()
    >>> features = extractor.extract_features(mesh)
    >>> print(features["fpfh_descriptor"])  # 33-dim rotation-invariant descriptor
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Conditional import for Open3D
try:
    import open3d as o3d

    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    o3d = None


__all__ = [
    "GeometricFeatureExtractor",
    "GeometricFeatures",
    "OPEN3D_AVAILABLE",
]


@dataclass
class GeometricFeatures:
    """Container for extracted geometric features."""

    # FPFH descriptor (33-dimensional, rotation-invariant)
    fpfh_descriptor: list[float] = field(default_factory=list)

    # Basic geometry
    point_count: int = 0
    surface_area: float = 0.0
    volume: float = 0.0

    # Bounding box (after canonical alignment)
    bbox_extent: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    bbox_volume: float = 0.0

    # Centroid
    centroid: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # Shape descriptors
    compactness: float = 0.0  # How sphere-like
    elongation: float = 0.0  # Ratio of largest to smallest extent

    # Curvature statistics
    mean_curvature: float = 0.0
    gaussian_curvature: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fpfh_descriptor": self.fpfh_descriptor,
            "point_count": self.point_count,
            "surface_area": self.surface_area,
            "volume": self.volume,
            "bbox_extent": self.bbox_extent,
            "bbox_volume": self.bbox_volume,
            "centroid": self.centroid,
            "compactness": self.compactness,
            "elongation": self.elongation,
            "mean_curvature": self.mean_curvature,
            "gaussian_curvature": self.gaussian_curvature,
        }

    def to_vector(self) -> list[float]:
        """Convert to flat feature vector for similarity search."""
        return (
            self.fpfh_descriptor
            + self.bbox_extent
            + self.centroid
            + [
                self.surface_area,
                self.volume,
                self.compactness,
                self.elongation,
            ]
        )


class GeometricFeatureExtractor:
    """Advanced geometric feature extractor using Open3D.

    This extractor computes rotation-invariant features suitable for
    CAD similarity search, replacing the basic features in feature_extractor.py.

    Attributes:
        num_points: Number of points for point cloud sampling.
        fpfh_radius: Radius for FPFH feature computation.
        normal_radius: Radius for normal estimation.
    """

    def __init__(
        self,
        num_points: int = 2048,
        fpfh_radius: float = 0.25,
        normal_radius: float = 0.1,
    ):
        """Initialize the extractor.

        Args:
            num_points: Target number of points after downsampling.
            fpfh_radius: Search radius for FPFH computation.
            normal_radius: Search radius for normal estimation.

        Raises:
            ImportError: If Open3D is not installed.
        """
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3D is not installed. Install with: pip install open3d")

        self.num_points = num_points
        self.fpfh_radius = fpfh_radius
        self.normal_radius = normal_radius

    def mesh_to_pointcloud(self, mesh: Any) -> "o3d.geometry.PointCloud":
        """Convert a mesh to Open3D point cloud.

        Args:
            mesh: Input mesh (trimesh or vertices array).

        Returns:
            Open3D PointCloud with estimated normals.
        """
        # Extract vertices
        if hasattr(mesh, "vertices"):
            vertices = np.asarray(mesh.vertices)
        elif isinstance(mesh, np.ndarray):
            vertices = mesh
        else:
            raise ValueError(f"Unsupported mesh type: {type(mesh)}")

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)

        # Downsample if too many points
        if len(vertices) > self.num_points:
            pcd = pcd.farthest_point_down_sample(self.num_points)

        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.normal_radius,
                max_nn=30,
            )
        )

        # Orient normals consistently
        pcd.orient_normals_consistent_tangent_plane(k=15)

        return pcd

    def align_to_canonical(self, pcd: "o3d.geometry.PointCloud") -> "o3d.geometry.PointCloud":
        """Align point cloud to canonical pose using PCA.

        This makes features rotation-invariant by aligning all
        shapes to a consistent orientation.

        Args:
            pcd: Input point cloud.

        Returns:
            Aligned point cloud.
        """
        points = np.asarray(pcd.points)

        # Center the points
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid

        # PCA: compute covariance matrix
        cov = np.cov(points_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalues (largest first)
        idx = np.argsort(eigenvalues)[::-1]
        rotation = eigenvectors[:, idx]

        # Ensure right-handed coordinate system
        if np.linalg.det(rotation) < 0:
            rotation[:, 2] *= -1

        # Apply rotation
        aligned_points = points_centered @ rotation

        # Create new point cloud
        aligned_pcd = o3d.geometry.PointCloud()
        aligned_pcd.points = o3d.utility.Vector3dVector(aligned_points)

        # Transform normals if available
        if pcd.has_normals():
            normals = np.asarray(pcd.normals)
            aligned_normals = normals @ rotation
            aligned_pcd.normals = o3d.utility.Vector3dVector(aligned_normals)

        return aligned_pcd

    def compute_fpfh(self, pcd: "o3d.geometry.PointCloud") -> np.ndarray:
        """Compute FPFH (Fast Point Feature Histogram) descriptor.

        FPFH is a 33-dimensional rotation-invariant descriptor that
        captures local geometry around each point.

        Args:
            pcd: Input point cloud with normals.

        Returns:
            33-dimensional FPFH descriptor (mean-pooled).
        """
        # Ensure normals exist
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.normal_radius,
                    max_nn=30,
                )
            )

        # Compute FPFH features
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.fpfh_radius,
                max_nn=100,
            ),
        )

        # Mean pooling to get global descriptor
        return np.mean(fpfh.data, axis=1)

    def compute_shape_descriptors(
        self, pcd: "o3d.geometry.PointCloud", mesh: Any = None
    ) -> dict[str, float]:
        """Compute shape descriptors.

        Args:
            pcd: Point cloud.
            mesh: Original mesh (for volume/area if available).

        Returns:
            Dictionary of shape descriptors.
        """
        points = np.asarray(pcd.points)
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()

        # Bounding box volume
        bbox_volume = extent[0] * extent[1] * extent[2]

        # Elongation (ratio of largest to smallest extent)
        sorted_extent = sorted(extent)
        elongation = sorted_extent[2] / max(sorted_extent[0], 1e-10)

        # Surface area and volume from mesh
        surface_area = 0.0
        volume = 0.0
        if mesh is not None:
            if hasattr(mesh, "area"):
                surface_area = float(mesh.area)
            if hasattr(mesh, "volume"):
                try:
                    volume = float(abs(mesh.volume))
                except Exception:
                    pass

        # Compactness (how sphere-like)
        # For a sphere: compactness = 1
        if surface_area > 0:
            compactness = (36 * np.pi * volume**2) / (surface_area**3)
        else:
            compactness = 0.0

        return {
            "bbox_extent": extent.tolist(),
            "bbox_volume": bbox_volume,
            "elongation": elongation,
            "surface_area": surface_area,
            "volume": volume,
            "compactness": min(compactness, 1.0),  # Clamp to [0, 1]
        }

    def extract_features(self, mesh: Any) -> GeometricFeatures:
        """Extract complete geometric features from a mesh.

        This is the main entry point for feature extraction.

        Args:
            mesh: Input mesh (trimesh or similar).

        Returns:
            GeometricFeatures object with all computed features.
        """
        # Convert to point cloud
        pcd = self.mesh_to_pointcloud(mesh)

        # Align to canonical pose
        aligned_pcd = self.align_to_canonical(pcd)

        # Compute FPFH descriptor
        fpfh = self.compute_fpfh(aligned_pcd)

        # Compute shape descriptors
        shape_desc = self.compute_shape_descriptors(aligned_pcd, mesh)

        # Centroid (should be near origin after alignment)
        points = np.asarray(aligned_pcd.points)
        centroid = np.mean(points, axis=0)

        return GeometricFeatures(
            fpfh_descriptor=fpfh.tolist(),
            point_count=len(points),
            surface_area=shape_desc["surface_area"],
            volume=shape_desc["volume"],
            bbox_extent=shape_desc["bbox_extent"],
            bbox_volume=shape_desc["bbox_volume"],
            centroid=centroid.tolist(),
            compactness=shape_desc["compactness"],
            elongation=shape_desc["elongation"],
        )

    def extract_features_dict(self, mesh: Any) -> dict[str, Any]:
        """Extract features as a dictionary.

        Convenience method for JSON serialization.

        Args:
            mesh: Input mesh.

        Returns:
            Dictionary of features.
        """
        return self.extract_features(mesh).to_dict()


# Fallback extractor when Open3D is not available
class _FallbackFeatureExtractor:
    """Fallback extractor using basic numpy operations."""

    def __init__(self, **kwargs):
        logger.warning("Open3D not available. Using fallback extractor with limited features.")

    def extract_features(self, mesh: Any) -> GeometricFeatures:
        """Extract basic features without Open3D."""
        features = GeometricFeatures()

        if hasattr(mesh, "vertices"):
            vertices = np.asarray(mesh.vertices)
            features.point_count = len(vertices)
            features.centroid = np.mean(vertices, axis=0).tolist()

            # Basic bounding box
            min_pt = np.min(vertices, axis=0)
            max_pt = np.max(vertices, axis=0)
            extent = max_pt - min_pt
            features.bbox_extent = extent.tolist()
            features.bbox_volume = float(np.prod(extent))

            # Elongation
            sorted_ext = sorted(extent)
            features.elongation = sorted_ext[2] / max(sorted_ext[0], 1e-10)

        if hasattr(mesh, "area"):
            features.surface_area = float(mesh.area)

        if hasattr(mesh, "volume"):
            try:
                features.volume = float(abs(mesh.volume))
            except Exception:
                pass

        # Empty FPFH (will be zeros)
        features.fpfh_descriptor = [0.0] * 33

        return features

    def extract_features_dict(self, mesh: Any) -> dict[str, Any]:
        return self.extract_features(mesh).to_dict()


def get_feature_extractor(**kwargs) -> GeometricFeatureExtractor:
    """Get the appropriate feature extractor.

    Returns Open3D extractor if available, otherwise fallback.

    Returns:
        Feature extractor instance.
    """
    if OPEN3D_AVAILABLE:
        return GeometricFeatureExtractor(**kwargs)
    else:
        return _FallbackFeatureExtractor(**kwargs)
