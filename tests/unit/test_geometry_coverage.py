"""Tests for src/core/geometry/__init__.py to improve coverage.

Covers:
- GeometricFeatures dataclass
- GeometricFeatureExtractor class
- _FallbackFeatureExtractor class
- get_feature_extractor factory
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestGeometricFeaturesDataclass:
    """Tests for GeometricFeatures dataclass."""

    def test_default_values(self):
        """Test default values for GeometricFeatures."""
        from src.core.geometry import GeometricFeatures

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
        assert features.mean_curvature == 0.0
        assert features.gaussian_curvature == 0.0

    def test_custom_values(self):
        """Test custom values for GeometricFeatures."""
        from src.core.geometry import GeometricFeatures

        features = GeometricFeatures(
            fpfh_descriptor=[1.0, 2.0, 3.0],
            point_count=1000,
            surface_area=100.5,
            volume=50.25,
            bbox_extent=[10.0, 20.0, 30.0],
            bbox_volume=6000.0,
            centroid=[5.0, 10.0, 15.0],
            compactness=0.8,
            elongation=3.0,
        )

        assert features.fpfh_descriptor == [1.0, 2.0, 3.0]
        assert features.point_count == 1000
        assert features.surface_area == 100.5
        assert features.volume == 50.25

    def test_to_dict(self):
        """Test to_dict method."""
        from src.core.geometry import GeometricFeatures

        features = GeometricFeatures(fpfh_descriptor=[1.0, 2.0], point_count=100, surface_area=50.0)

        result = features.to_dict()

        assert isinstance(result, dict)
        assert result["fpfh_descriptor"] == [1.0, 2.0]
        assert result["point_count"] == 100
        assert result["surface_area"] == 50.0
        assert "volume" in result
        assert "bbox_extent" in result
        assert "centroid" in result
        assert "compactness" in result
        assert "elongation" in result

    def test_to_vector(self):
        """Test to_vector method."""
        from src.core.geometry import GeometricFeatures

        features = GeometricFeatures(
            fpfh_descriptor=[1.0, 2.0, 3.0],
            bbox_extent=[10.0, 20.0, 30.0],
            centroid=[5.0, 10.0, 15.0],
            surface_area=100.0,
            volume=50.0,
            compactness=0.8,
            elongation=2.0,
        )

        vector = features.to_vector()

        assert isinstance(vector, list)
        # fpfh (3) + bbox (3) + centroid (3) + [surface, volume, compactness, elongation] (4) = 13
        assert len(vector) == 13
        # First elements are fpfh_descriptor
        assert vector[0] == 1.0
        assert vector[1] == 2.0
        assert vector[2] == 3.0


class TestOPEN3DAvailable:
    """Tests for OPEN3D_AVAILABLE flag."""

    def test_open3d_available_is_bool(self):
        """Test OPEN3D_AVAILABLE is boolean."""
        from src.core.geometry import OPEN3D_AVAILABLE

        assert isinstance(OPEN3D_AVAILABLE, bool)


class TestFallbackFeatureExtractor:
    """Tests for _FallbackFeatureExtractor class."""

    def test_fallback_init_logs_warning(self):
        """Test fallback extractor logs warning on init."""
        from src.core.geometry import _FallbackFeatureExtractor

        with patch("src.core.geometry.logger") as mock_logger:
            extractor = _FallbackFeatureExtractor()
            mock_logger.warning.assert_called_once()

    def test_fallback_extract_features_basic(self):
        """Test fallback extract_features returns basic features."""
        from src.core.geometry import GeometricFeatures, _FallbackFeatureExtractor

        extractor = _FallbackFeatureExtractor()

        # Create mock mesh with vertices
        mock_mesh = MagicMock()
        mock_mesh.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        mock_mesh.area = 10.0
        mock_mesh.volume = 0.166

        features = extractor.extract_features(mock_mesh)

        assert isinstance(features, GeometricFeatures)
        assert features.point_count == 4
        assert features.surface_area == 10.0

    def test_fallback_extract_features_no_area(self):
        """Test fallback handles mesh without area attribute."""
        from src.core.geometry import _FallbackFeatureExtractor

        extractor = _FallbackFeatureExtractor()

        mock_mesh = MagicMock()
        mock_mesh.vertices = np.array([[0, 0, 0], [1, 1, 1]])
        del mock_mesh.area

        features = extractor.extract_features(mock_mesh)

        assert features.surface_area == 0.0

    def test_fallback_extract_features_no_volume(self):
        """Test fallback handles mesh without volume attribute."""
        from src.core.geometry import _FallbackFeatureExtractor

        extractor = _FallbackFeatureExtractor()

        mock_mesh = MagicMock()
        mock_mesh.vertices = np.array([[0, 0, 0], [1, 1, 1]])
        del mock_mesh.volume

        features = extractor.extract_features(mock_mesh)

        assert features.volume == 0.0

    def test_fallback_extract_features_negative_volume(self):
        """Test fallback handles negative volume (uses abs)."""
        from src.core.geometry import _FallbackFeatureExtractor

        extractor = _FallbackFeatureExtractor()

        mock_mesh = MagicMock()
        mock_mesh.vertices = np.array([[0, 0, 0], [1, 1, 1]])
        mock_mesh.volume = -10.5  # Negative volume

        features = extractor.extract_features(mock_mesh)

        # Volume should be absolute value
        assert features.volume == 10.5

    def test_fallback_fpfh_is_zeros(self):
        """Test fallback FPFH descriptor is zeros."""
        from src.core.geometry import _FallbackFeatureExtractor

        extractor = _FallbackFeatureExtractor()

        mock_mesh = MagicMock()
        mock_mesh.vertices = np.array([[0, 0, 0], [1, 1, 1]])

        features = extractor.extract_features(mock_mesh)

        assert features.fpfh_descriptor == [0.0] * 33

    def test_fallback_extract_features_dict(self):
        """Test fallback extract_features_dict method."""
        from src.core.geometry import _FallbackFeatureExtractor

        extractor = _FallbackFeatureExtractor()

        mock_mesh = MagicMock()
        mock_mesh.vertices = np.array([[0, 0, 0], [1, 1, 1]])

        result = extractor.extract_features_dict(mock_mesh)

        assert isinstance(result, dict)
        assert "fpfh_descriptor" in result
        assert "point_count" in result

    def test_fallback_bbox_calculation(self):
        """Test fallback bounding box calculation."""
        from src.core.geometry import _FallbackFeatureExtractor

        extractor = _FallbackFeatureExtractor()

        mock_mesh = MagicMock()
        mock_mesh.vertices = np.array([[0, 0, 0], [10, 20, 30]])

        features = extractor.extract_features(mock_mesh)

        assert features.bbox_extent == [10.0, 20.0, 30.0]
        assert features.bbox_volume == 6000.0

    def test_fallback_elongation_calculation(self):
        """Test fallback elongation calculation."""
        from src.core.geometry import _FallbackFeatureExtractor

        extractor = _FallbackFeatureExtractor()

        mock_mesh = MagicMock()
        mock_mesh.vertices = np.array([[0, 0, 0], [100, 10, 10]])  # Elongated in x direction

        features = extractor.extract_features(mock_mesh)

        # Elongation = largest / smallest = 100 / 10 = 10
        assert features.elongation == 10.0

    def test_fallback_centroid_calculation(self):
        """Test fallback centroid calculation."""
        from src.core.geometry import _FallbackFeatureExtractor

        extractor = _FallbackFeatureExtractor()

        mock_mesh = MagicMock()
        mock_mesh.vertices = np.array([[0, 0, 0], [10, 10, 10]])

        features = extractor.extract_features(mock_mesh)

        # Centroid = mean of vertices
        assert features.centroid == [5.0, 5.0, 5.0]


class TestGetFeatureExtractor:
    """Tests for get_feature_extractor factory function."""

    def test_returns_extractor(self):
        """Test get_feature_extractor returns an extractor."""
        from src.core.geometry import get_feature_extractor

        extractor = get_feature_extractor()

        assert extractor is not None
        assert hasattr(extractor, "extract_features")

    def test_with_custom_params(self):
        """Test get_feature_extractor with custom parameters."""
        from src.core.geometry import get_feature_extractor

        extractor = get_feature_extractor(num_points=1024)

        assert extractor is not None


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports(self):
        """Test module exports expected names."""
        from src.core import geometry

        assert "GeometricFeatureExtractor" in geometry.__all__
        assert "GeometricFeatures" in geometry.__all__
        assert "OPEN3D_AVAILABLE" in geometry.__all__


class TestGeometricFeaturesEdgeCases:
    """Tests for edge cases in GeometricFeatures."""

    def test_empty_fpfh_to_vector(self):
        """Test to_vector with empty fpfh."""
        from src.core.geometry import GeometricFeatures

        features = GeometricFeatures()
        vector = features.to_vector()

        # Should still work with empty fpfh
        assert isinstance(vector, list)

    def test_to_dict_all_fields(self):
        """Test to_dict includes all fields."""
        from src.core.geometry import GeometricFeatures

        features = GeometricFeatures()
        result = features.to_dict()

        expected_fields = [
            "fpfh_descriptor",
            "point_count",
            "surface_area",
            "volume",
            "bbox_extent",
            "bbox_volume",
            "centroid",
            "compactness",
            "elongation",
            "mean_curvature",
            "gaussian_curvature",
        ]

        for field in expected_fields:
            assert field in result


class TestFallbackExtractorNoVertices:
    """Tests for fallback extractor with mesh without vertices."""

    def test_mesh_without_vertices(self):
        """Test fallback handles mesh without vertices attribute."""
        from src.core.geometry import GeometricFeatures, _FallbackFeatureExtractor

        extractor = _FallbackFeatureExtractor()

        mock_mesh = MagicMock()
        del mock_mesh.vertices

        features = extractor.extract_features(mock_mesh)

        # Should return default features
        assert isinstance(features, GeometricFeatures)
        assert features.point_count == 0


class TestFallbackElongationDivisionByZero:
    """Tests for elongation calculation division by zero."""

    def test_elongation_small_extent(self):
        """Test elongation with very small extent."""
        from src.core.geometry import _FallbackFeatureExtractor

        extractor = _FallbackFeatureExtractor()

        mock_mesh = MagicMock()
        # Nearly zero extent in one dimension
        mock_mesh.vertices = np.array([[0, 0, 0], [100, 0.0000001, 10]])

        features = extractor.extract_features(mock_mesh)

        # Should not raise, should use 1e-10 minimum
        assert features.elongation > 0
