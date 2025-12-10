"""
Unit tests for Open3D geometry feature extraction module.

Tests cover:
- GeometricFeatures dataclass
- GeometricFeatureExtractor (when Open3D available)
- Fallback extractor (when Open3D unavailable)
- Feature extraction pipeline
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.core.geometry import (
    GeometricFeatures,
    OPEN3D_AVAILABLE,
)


class TestGeometricFeatures:
    """Test GeometricFeatures dataclass."""

    def test_default_features(self):
        """Test default feature values."""
        features = GeometricFeatures()

        assert features.fpfh_descriptor == []
        assert features.point_count == 0
        assert features.surface_area == 0.0
        assert features.volume == 0.0
        assert features.bbox_extent == [0.0, 0.0, 0.0]
        assert features.bbox_volume == 0.0
        assert features.centroid == [0.0, 0.0, 0.0]
        assert features.compactness == 0.0
        assert features.elongation == 0.0

    def test_custom_features(self):
        """Test custom feature values."""
        features = GeometricFeatures(
            fpfh_descriptor=[0.1] * 33,
            point_count=1000,
            surface_area=50.5,
            volume=100.0,
            bbox_extent=[10.0, 5.0, 2.0],
            bbox_volume=100.0,
            centroid=[1.0, 2.0, 3.0],
            compactness=0.8,
            elongation=5.0,
        )

        assert len(features.fpfh_descriptor) == 33
        assert features.point_count == 1000
        assert features.surface_area == 50.5
        assert features.volume == 100.0
        assert features.bbox_extent == [10.0, 5.0, 2.0]
        assert features.compactness == 0.8

    def test_to_dict(self):
        """Test conversion to dictionary."""
        features = GeometricFeatures(
            fpfh_descriptor=[0.1, 0.2],
            point_count=100,
            surface_area=10.0,
        )

        result = features.to_dict()

        assert isinstance(result, dict)
        assert result["fpfh_descriptor"] == [0.1, 0.2]
        assert result["point_count"] == 100
        assert result["surface_area"] == 10.0

    def test_to_vector(self):
        """Test conversion to flat feature vector."""
        features = GeometricFeatures(
            fpfh_descriptor=[0.1, 0.2, 0.3],
            bbox_extent=[1.0, 2.0, 3.0],
            centroid=[0.1, 0.2, 0.3],
            surface_area=10.0,
            volume=20.0,
            compactness=0.5,
            elongation=2.0,
        )

        vector = features.to_vector()

        assert isinstance(vector, list)
        # FPFH (3) + bbox (3) + centroid (3) + 4 scalars = 13
        assert len(vector) == 13
        assert vector[:3] == [0.1, 0.2, 0.3]  # FPFH
        assert vector[3:6] == [1.0, 2.0, 3.0]  # bbox_extent


class TestFallbackExtractor:
    """Test fallback extractor when Open3D is unavailable."""

    def test_fallback_extractor_init(self):
        """Test fallback extractor initialization."""
        from src.core.geometry import _FallbackFeatureExtractor

        extractor = _FallbackFeatureExtractor()
        assert extractor is not None

    def test_fallback_extractor_with_mock_mesh(self):
        """Test fallback extractor with mock mesh."""
        from src.core.geometry import _FallbackFeatureExtractor

        # Create mock mesh
        mock_mesh = MagicMock()
        mock_mesh.vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        mock_mesh.area = 6.0
        mock_mesh.volume = 1.0

        extractor = _FallbackFeatureExtractor()
        features = extractor.extract_features(mock_mesh)

        assert features.point_count == 4
        assert features.surface_area == 6.0
        assert features.volume == 1.0
        assert len(features.fpfh_descriptor) == 33  # Zeros

    def test_fallback_extractor_dict(self):
        """Test fallback extractor dict output."""
        from src.core.geometry import _FallbackFeatureExtractor

        mock_mesh = MagicMock()
        mock_mesh.vertices = np.array([[0, 0, 0], [1, 1, 1]])
        mock_mesh.area = 10.0
        mock_mesh.volume = 5.0

        extractor = _FallbackFeatureExtractor()
        result = extractor.extract_features_dict(mock_mesh)

        assert isinstance(result, dict)
        assert "fpfh_descriptor" in result
        assert "point_count" in result


class TestGetFeatureExtractor:
    """Test get_feature_extractor factory function."""

    def test_get_extractor_returns_instance(self):
        """Test that factory returns an extractor."""
        from src.core.geometry import get_feature_extractor

        extractor = get_feature_extractor()
        assert extractor is not None
        assert hasattr(extractor, "extract_features")

    def test_get_extractor_with_params(self):
        """Test factory with custom parameters."""
        from src.core.geometry import get_feature_extractor

        extractor = get_feature_extractor(num_points=1024)
        assert extractor is not None


@pytest.mark.skipif(not OPEN3D_AVAILABLE, reason="Open3D not installed")
class TestGeometricFeatureExtractor:
    """Test GeometricFeatureExtractor when Open3D is available."""

    @pytest.fixture
    def extractor(self):
        """Create a GeometricFeatureExtractor instance."""
        from src.core.geometry import GeometricFeatureExtractor
        return GeometricFeatureExtractor(num_points=512)

    @pytest.fixture
    def sample_mesh(self):
        """Create a sample mesh for testing."""
        # Simple tetrahedron
        mock_mesh = MagicMock()
        mock_mesh.vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
            [0.5, 0.5, 1],
        ])
        mock_mesh.area = 2.5
        mock_mesh.volume = 0.16
        return mock_mesh

    def test_mesh_to_pointcloud(self, extractor, sample_mesh):
        """Test mesh to point cloud conversion."""
        pcd = extractor.mesh_to_pointcloud(sample_mesh)

        assert pcd is not None
        assert len(pcd.points) <= 512  # Downsampled
        assert pcd.has_normals()

    def test_align_to_canonical(self, extractor, sample_mesh):
        """Test PCA-based canonical alignment."""
        pcd = extractor.mesh_to_pointcloud(sample_mesh)
        aligned = extractor.align_to_canonical(pcd)

        assert aligned is not None
        # Centroid should be near origin after alignment
        points = np.asarray(aligned.points)
        centroid = np.mean(points, axis=0)
        assert np.allclose(centroid, [0, 0, 0], atol=1e-10)

    def test_compute_fpfh(self, extractor, sample_mesh):
        """Test FPFH descriptor computation."""
        pcd = extractor.mesh_to_pointcloud(sample_mesh)
        fpfh = extractor.compute_fpfh(pcd)

        # FPFH is 33-dimensional
        assert len(fpfh) == 33
        assert all(isinstance(v, (int, float)) for v in fpfh)

    def test_extract_features(self, extractor, sample_mesh):
        """Test full feature extraction pipeline."""
        features = extractor.extract_features(sample_mesh)

        assert isinstance(features, GeometricFeatures)
        assert len(features.fpfh_descriptor) == 33
        assert features.point_count > 0
        assert features.surface_area == 2.5
        assert features.volume == 0.16

    def test_extract_features_dict(self, extractor, sample_mesh):
        """Test feature extraction returning dict."""
        result = extractor.extract_features_dict(sample_mesh)

        assert isinstance(result, dict)
        assert "fpfh_descriptor" in result
        assert "bbox_extent" in result
        assert "compactness" in result


class TestModuleExports:
    """Test module exports and availability."""

    def test_all_exports_available(self):
        """Test that all expected exports are available."""
        from src.core import geometry

        assert hasattr(geometry, "GeometricFeatureExtractor")
        assert hasattr(geometry, "GeometricFeatures")
        assert hasattr(geometry, "OPEN3D_AVAILABLE")
        assert hasattr(geometry, "get_feature_extractor")

    def test_open3d_availability_flag(self):
        """Test OPEN3D_AVAILABLE flag is set."""
        from src.core.geometry import OPEN3D_AVAILABLE

        assert isinstance(OPEN3D_AVAILABLE, bool)
