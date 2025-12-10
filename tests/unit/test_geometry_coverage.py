"""Tests for src/core/geometry/__init__.py to improve coverage.

Covers:
- GeometricFeatures dataclass
- GeometricFeatureExtractor class (with mocked Open3D)
- _FallbackFeatureExtractor class
- get_feature_extractor function
- OPEN3D_AVAILABLE flag
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np

import pytest


class TestGeometricFeatures:
    """Tests for GeometricFeatures dataclass."""

    def test_default_values(self):
        """Test GeometricFeatures has correct default values."""
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
        """Test GeometricFeatures with custom values."""
        from src.core.geometry import GeometricFeatures

        features = GeometricFeatures(
            fpfh_descriptor=[0.1, 0.2, 0.3],
            point_count=1000,
            surface_area=100.5,
            volume=50.25,
            bbox_extent=[10.0, 20.0, 30.0],
            bbox_volume=6000.0,
            centroid=[1.0, 2.0, 3.0],
            compactness=0.8,
            elongation=3.0,
            mean_curvature=0.5,
            gaussian_curvature=0.1,
        )

        assert features.fpfh_descriptor == [0.1, 0.2, 0.3]
        assert features.point_count == 1000
        assert features.surface_area == 100.5
        assert features.volume == 50.25
        assert features.bbox_extent == [10.0, 20.0, 30.0]
        assert features.bbox_volume == 6000.0
        assert features.centroid == [1.0, 2.0, 3.0]
        assert features.compactness == 0.8
        assert features.elongation == 3.0

    def test_to_dict(self):
        """Test GeometricFeatures.to_dict method."""
        from src.core.geometry import GeometricFeatures

        features = GeometricFeatures(
            fpfh_descriptor=[0.1, 0.2],
            point_count=100,
            surface_area=50.0,
            volume=25.0,
            bbox_extent=[5.0, 10.0, 15.0],
            bbox_volume=750.0,
            centroid=[1.0, 2.0, 3.0],
            compactness=0.5,
            elongation=2.0,
            mean_curvature=0.3,
            gaussian_curvature=0.2,
        )

        result = features.to_dict()

        assert isinstance(result, dict)
        assert result["fpfh_descriptor"] == [0.1, 0.2]
        assert result["point_count"] == 100
        assert result["surface_area"] == 50.0
        assert result["volume"] == 25.0
        assert result["bbox_extent"] == [5.0, 10.0, 15.0]
        assert result["bbox_volume"] == 750.0
        assert result["centroid"] == [1.0, 2.0, 3.0]
        assert result["compactness"] == 0.5
        assert result["elongation"] == 2.0
        assert result["mean_curvature"] == 0.3
        assert result["gaussian_curvature"] == 0.2

    def test_to_vector(self):
        """Test GeometricFeatures.to_vector method."""
        from src.core.geometry import GeometricFeatures

        features = GeometricFeatures(
            fpfh_descriptor=[0.1, 0.2, 0.3],
            bbox_extent=[1.0, 2.0, 3.0],
            centroid=[4.0, 5.0, 6.0],
            surface_area=100.0,
            volume=50.0,
            compactness=0.5,
            elongation=2.0,
        )

        result = features.to_vector()

        # fpfh (3) + bbox_extent (3) + centroid (3) + [area, volume, compactness, elongation] (4) = 13
        assert isinstance(result, list)
        assert result[:3] == [0.1, 0.2, 0.3]  # fpfh
        assert result[3:6] == [1.0, 2.0, 3.0]  # bbox_extent
        assert result[6:9] == [4.0, 5.0, 6.0]  # centroid
        assert result[9:] == [100.0, 50.0, 0.5, 2.0]  # area, volume, compactness, elongation


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports(self):
        """Test __all__ contains expected exports."""
        from src.core import geometry

        assert "GeometricFeatureExtractor" in geometry.__all__
        assert "GeometricFeatures" in geometry.__all__
        assert "OPEN3D_AVAILABLE" in geometry.__all__

    def test_open3d_available_flag(self):
        """Test OPEN3D_AVAILABLE flag exists."""
        from src.core.geometry import OPEN3D_AVAILABLE

        assert isinstance(OPEN3D_AVAILABLE, bool)


class TestGetFeatureExtractor:
    """Tests for get_feature_extractor function."""

    def test_get_feature_extractor_import(self):
        """Test get_feature_extractor can be imported."""
        from src.core.geometry import get_feature_extractor

        assert callable(get_feature_extractor)

    def test_returns_fallback_when_no_open3d(self):
        """Test returns fallback extractor when Open3D not available."""
        from src.core.geometry import get_feature_extractor

        extractor = get_feature_extractor()

        # Should return either Open3D or fallback based on availability
        assert extractor is not None


class TestFallbackFeatureExtractor:
    """Tests for _FallbackFeatureExtractor class."""

    def test_fallback_extract_features_with_mesh(self):
        """Test fallback extractor with mesh-like object."""
        from src.core.geometry import _FallbackFeatureExtractor, GeometricFeatures

        extractor = _FallbackFeatureExtractor()

        # Create mock mesh
        mock_mesh = MagicMock()
        mock_mesh.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        mock_mesh.area = 10.0
        mock_mesh.volume = 5.0

        result = extractor.extract_features(mock_mesh)

        assert isinstance(result, GeometricFeatures)
        assert result.point_count == 4
        assert len(result.centroid) == 3
        assert result.surface_area == 10.0
        assert result.volume == 5.0
        assert len(result.fpfh_descriptor) == 33  # Empty FPFH placeholder

    def test_fallback_extract_features_dict(self):
        """Test fallback extractor extract_features_dict method."""
        from src.core.geometry import _FallbackFeatureExtractor

        extractor = _FallbackFeatureExtractor()

        mock_mesh = MagicMock()
        mock_mesh.vertices = np.array([[0, 0, 0], [1, 1, 1]])
        mock_mesh.area = 5.0
        mock_mesh.volume = 2.0

        result = extractor.extract_features_dict(mock_mesh)

        assert isinstance(result, dict)
        assert "fpfh_descriptor" in result
        assert "point_count" in result

    def test_fallback_handles_negative_volume(self):
        """Test fallback handles negative volume (takes absolute value)."""
        from src.core.geometry import _FallbackFeatureExtractor

        extractor = _FallbackFeatureExtractor()

        mock_mesh = MagicMock()
        mock_mesh.vertices = np.array([[0, 0, 0], [1, 1, 1]])
        mock_mesh.area = 5.0
        mock_mesh.volume = -10.5  # Negative volume

        result = extractor.extract_features(mock_mesh)

        assert result.volume == 10.5  # Should be absolute value

    def test_fallback_bbox_calculation(self):
        """Test fallback correctly calculates bounding box."""
        from src.core.geometry import _FallbackFeatureExtractor

        extractor = _FallbackFeatureExtractor()

        mock_mesh = MagicMock()
        # 3D box vertices
        mock_mesh.vertices = np.array([
            [0, 0, 0], [10, 0, 0], [0, 20, 0], [0, 0, 30],
            [10, 20, 0], [10, 0, 30], [0, 20, 30], [10, 20, 30]
        ])

        result = extractor.extract_features(mock_mesh)

        assert result.bbox_extent == [10.0, 20.0, 30.0]
        assert result.bbox_volume == 6000.0

    def test_fallback_elongation_calculation(self):
        """Test fallback correctly calculates elongation."""
        from src.core.geometry import _FallbackFeatureExtractor

        extractor = _FallbackFeatureExtractor()

        mock_mesh = MagicMock()
        # Long thin box
        mock_mesh.vertices = np.array([
            [0, 0, 0], [100, 0, 0], [0, 10, 0], [0, 0, 10]
        ])

        result = extractor.extract_features(mock_mesh)

        # Elongation = largest / smallest = 100 / 10 = 10
        assert result.elongation == 10.0

    def test_fallback_centroid_calculation(self):
        """Test fallback correctly calculates centroid."""
        from src.core.geometry import _FallbackFeatureExtractor

        extractor = _FallbackFeatureExtractor()

        mock_mesh = MagicMock()
        mock_mesh.vertices = np.array([
            [0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]
        ])

        result = extractor.extract_features(mock_mesh)

        # Centroid should be at [5, 5, 0]
        assert result.centroid[0] == 5.0
        assert result.centroid[1] == 5.0
        assert result.centroid[2] == 0.0


class TestGeometricFeatureExtractorInit:
    """Tests for GeometricFeatureExtractor initialization."""

    def test_init_raises_without_open3d(self):
        """Test init raises ImportError when Open3D not available."""
        with patch.dict("sys.modules", {"open3d": None}):
            with patch("src.core.geometry.OPEN3D_AVAILABLE", False):
                import src.core.geometry as geometry_module
                
                # Reset the OPEN3D_AVAILABLE flag
                original_flag = geometry_module.OPEN3D_AVAILABLE
                geometry_module.OPEN3D_AVAILABLE = False
                
                try:
                    with pytest.raises(ImportError) as exc_info:
                        geometry_module.GeometricFeatureExtractor()
                    
                    assert "Open3D is not installed" in str(exc_info.value)
                finally:
                    geometry_module.OPEN3D_AVAILABLE = original_flag

    def test_init_default_parameters(self):
        """Test init with default parameters (if Open3D available)."""
        from src.core.geometry import OPEN3D_AVAILABLE

        if OPEN3D_AVAILABLE:
            from src.core.geometry import GeometricFeatureExtractor

            extractor = GeometricFeatureExtractor()

            assert extractor.num_points == 2048
            assert extractor.fpfh_radius == 0.25
            assert extractor.normal_radius == 0.1

    def test_init_custom_parameters(self):
        """Test init with custom parameters (if Open3D available)."""
        from src.core.geometry import OPEN3D_AVAILABLE

        if OPEN3D_AVAILABLE:
            from src.core.geometry import GeometricFeatureExtractor

            extractor = GeometricFeatureExtractor(
                num_points=1024,
                fpfh_radius=0.5,
                normal_radius=0.2,
            )

            assert extractor.num_points == 1024
            assert extractor.fpfh_radius == 0.5
            assert extractor.normal_radius == 0.2


class TestOpen3DIntegration:
    """Tests for Open3D integration (skipped if not available)."""

    @pytest.fixture
    def extractor(self):
        """Create extractor if Open3D available."""
        from src.core.geometry import OPEN3D_AVAILABLE

        if not OPEN3D_AVAILABLE:
            pytest.skip("Open3D not available")

        from src.core.geometry import GeometricFeatureExtractor
        return GeometricFeatureExtractor(num_points=100)

    def test_mesh_to_pointcloud_with_vertices_attr(self, extractor):
        """Test mesh_to_pointcloud with vertices attribute."""
        mock_mesh = MagicMock()
        mock_mesh.vertices = np.random.rand(200, 3)

        pcd = extractor.mesh_to_pointcloud(mock_mesh)

        assert pcd is not None
        assert len(pcd.points) <= extractor.num_points

    def test_mesh_to_pointcloud_with_ndarray(self, extractor):
        """Test mesh_to_pointcloud with numpy array."""
        vertices = np.random.rand(200, 3)

        pcd = extractor.mesh_to_pointcloud(vertices)

        assert pcd is not None
        assert len(pcd.points) <= extractor.num_points

    def test_mesh_to_pointcloud_unsupported_type(self, extractor):
        """Test mesh_to_pointcloud raises for unsupported type."""
        with pytest.raises(ValueError) as exc_info:
            extractor.mesh_to_pointcloud("not a mesh")

        assert "Unsupported mesh type" in str(exc_info.value)

    def test_align_to_canonical(self, extractor):
        """Test align_to_canonical method."""
        import open3d as o3d

        # Create simple point cloud
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals()

        aligned_pcd = extractor.align_to_canonical(pcd)

        assert aligned_pcd is not None
        assert len(aligned_pcd.points) == len(points)

    def test_compute_fpfh(self, extractor):
        """Test compute_fpfh method."""
        import open3d as o3d

        # Create simple point cloud with normals
        points = np.random.rand(50, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals()

        fpfh = extractor.compute_fpfh(pcd)

        assert isinstance(fpfh, np.ndarray)
        assert len(fpfh) == 33  # FPFH is 33-dimensional

    def test_compute_shape_descriptors(self, extractor):
        """Test compute_shape_descriptors method."""
        import open3d as o3d

        # Create simple point cloud
        points = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
        ], dtype=np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        result = extractor.compute_shape_descriptors(pcd)

        assert "bbox_extent" in result
        assert "bbox_volume" in result
        assert "elongation" in result
        assert "surface_area" in result
        assert "volume" in result
        assert "compactness" in result

    def test_extract_features(self, extractor):
        """Test extract_features method."""
        from src.core.geometry import GeometricFeatures

        mock_mesh = MagicMock()
        mock_mesh.vertices = np.random.rand(200, 3)
        mock_mesh.area = 100.0
        mock_mesh.volume = 50.0

        result = extractor.extract_features(mock_mesh)

        assert isinstance(result, GeometricFeatures)
        assert len(result.fpfh_descriptor) == 33
        assert result.point_count > 0
        assert len(result.bbox_extent) == 3
        assert len(result.centroid) == 3

    def test_extract_features_dict(self, extractor):
        """Test extract_features_dict method."""
        mock_mesh = MagicMock()
        mock_mesh.vertices = np.random.rand(200, 3)
        mock_mesh.area = 100.0
        mock_mesh.volume = 50.0

        result = extractor.extract_features_dict(mock_mesh)

        assert isinstance(result, dict)
        assert "fpfh_descriptor" in result


class TestEdgeCases:
    """Tests for edge cases."""

    def test_fallback_without_area_attribute(self):
        """Test fallback handles mesh without area attribute."""
        from src.core.geometry import _FallbackFeatureExtractor

        extractor = _FallbackFeatureExtractor()

        mock_mesh = MagicMock(spec=["vertices"])
        mock_mesh.vertices = np.array([[0, 0, 0], [1, 1, 1]])

        result = extractor.extract_features(mock_mesh)

        assert result.surface_area == 0.0

    def test_fallback_without_volume_attribute(self):
        """Test fallback handles mesh without volume attribute."""
        from src.core.geometry import _FallbackFeatureExtractor

        extractor = _FallbackFeatureExtractor()

        mock_mesh = MagicMock(spec=["vertices"])
        mock_mesh.vertices = np.array([[0, 0, 0], [1, 1, 1]])

        result = extractor.extract_features(mock_mesh)

        assert result.volume == 0.0

    def test_geometric_features_empty_fpfh(self):
        """Test GeometricFeatures with empty FPFH."""
        from src.core.geometry import GeometricFeatures

        features = GeometricFeatures(fpfh_descriptor=[])

        vector = features.to_vector()

        # Should still work with empty fpfh
        assert isinstance(vector, list)

    def test_fallback_very_small_extent(self):
        """Test fallback handles very small extent for elongation."""
        from src.core.geometry import _FallbackFeatureExtractor

        extractor = _FallbackFeatureExtractor()

        mock_mesh = MagicMock()
        # Points very close together in one dimension
        mock_mesh.vertices = np.array([
            [0, 0, 0], [100, 0.0001, 0.0001]
        ])

        result = extractor.extract_features(mock_mesh)

        # Should handle division without errors
        assert result.elongation > 0
