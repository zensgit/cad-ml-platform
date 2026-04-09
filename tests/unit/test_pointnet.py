"""
Unit tests for the PointNet module.

Tests cover model shapes, preprocessing utilities, and the high-level
analyser fallback behaviour. Torch-dependent tests are skipped
automatically when PyTorch is not installed.
"""

import os
import tempfile
from unittest import mock

import numpy as np
import pytest


# ------------------------------------------------------------------
# Preprocessor tests (numpy-only, no torch required)
# ------------------------------------------------------------------


class TestPreprocessorNormalize:
    """PointCloudPreprocessor.normalize should centre and unit-scale."""

    def test_centered_at_origin(self):
        from src.ml.pointnet.preprocessor import PointCloudPreprocessor

        rng = np.random.RandomState(42)
        points = rng.randn(256, 3) * 10 + np.array([5, -3, 7])
        normed = PointCloudPreprocessor.normalize(points)

        centroid = normed.mean(axis=0)
        np.testing.assert_allclose(centroid, 0.0, atol=1e-6)

    def test_max_radius_le_one(self):
        from src.ml.pointnet.preprocessor import PointCloudPreprocessor

        rng = np.random.RandomState(7)
        points = rng.randn(512, 3) * 50
        normed = PointCloudPreprocessor.normalize(points)

        max_radius = np.max(np.linalg.norm(normed, axis=1))
        assert max_radius <= 1.0 + 1e-7


class TestPreprocessorAugment:
    """Augmentation should preserve shape."""

    def test_shape_preserved(self):
        from src.ml.pointnet.preprocessor import PointCloudPreprocessor

        points = np.random.randn(1024, 3)
        aug = PointCloudPreprocessor.augment(points)
        assert aug.shape == points.shape

    def test_augment_no_ops(self):
        from src.ml.pointnet.preprocessor import PointCloudPreprocessor

        points = np.random.randn(64, 3)
        aug = PointCloudPreprocessor.augment(
            points, rotation=False, jitter=False, scale=False
        )
        np.testing.assert_array_equal(aug, points)


class TestPreprocessorLoadSTLMock:
    """Mock trimesh to verify STL loading pipeline."""

    def test_load_from_stl_calls_trimesh(self):
        from src.ml.pointnet.preprocessor import PointCloudPreprocessor

        fake_points = np.random.randn(2048, 3)
        fake_face_idx = np.zeros(2048, dtype=int)

        mock_mesh = mock.MagicMock()
        mock_sample = mock.MagicMock(return_value=(fake_points, fake_face_idx))

        with mock.patch("src.ml.pointnet.preprocessor.HAS_TRIMESH", True), \
             mock.patch("src.ml.pointnet.preprocessor.trimesh") as mock_trimesh:
            mock_trimesh.load.return_value = mock_mesh
            mock_trimesh.sample.sample_surface = mock_sample

            pp = PointCloudPreprocessor(num_points=2048)
            result = pp.load_from_stl("/fake/part.stl", num_points=2048)

            mock_trimesh.load.assert_called_once_with("/fake/part.stl", force="mesh")
            mock_trimesh.sample.sample_surface.assert_called_once()
            assert result.shape == (2048, 3)


class TestPreprocessorLoadXYZ:
    """Test loading from XYZ text file."""

    def test_load_xyz_basic(self):
        from src.ml.pointnet.preprocessor import PointCloudPreprocessor

        content = "# header\n1.0 2.0 3.0\n4.0 5.0 6.0\n7.0 8.0 9.0\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xyz", delete=False
        ) as f:
            f.write(content)
            f.flush()
            tmp_path = f.name

        try:
            pp = PointCloudPreprocessor(num_points=4, normalize_default=False)
            result = pp.load_from_xyz(tmp_path, num_points=4)
            assert result.shape == (4, 3)
        finally:
            os.unlink(tmp_path)


class TestPreprocessorAdjustPointCount:
    """Subsample and pad logic."""

    def test_pad(self):
        from src.ml.pointnet.preprocessor import PointCloudPreprocessor

        pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        result = PointCloudPreprocessor._adjust_point_count(pts, 5)
        assert result.shape == (5, 3)

    def test_subsample(self):
        from src.ml.pointnet.preprocessor import PointCloudPreprocessor

        pts = np.random.randn(100, 3)
        result = PointCloudPreprocessor._adjust_point_count(pts, 10)
        assert result.shape == (10, 3)


# ------------------------------------------------------------------
# Analyzer fallback tests (no torch needed)
# ------------------------------------------------------------------


class TestAnalyzerFallback:
    """Without a model the analyzer should return fallback results."""

    def test_fallback_classify(self):
        from src.ml.pointnet.inference import PointNet3DAnalyzer

        analyzer = PointNet3DAnalyzer(model_path=None)

        # Create a dummy XYZ file to classify
        content = "\n".join(
            f"{np.random.randn()} {np.random.randn()} {np.random.randn()}"
            for _ in range(100)
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xyz", delete=False
        ) as f:
            f.write(content)
            f.flush()
            tmp_path = f.name

        try:
            result = analyzer.classify(tmp_path)
            assert result["status"] == "model_unavailable"
            assert result["label"] == "unknown"
            assert isinstance(result["confidence"], float)
            assert isinstance(result["probabilities"], dict)
        finally:
            os.unlink(tmp_path)

    def test_fallback_features(self):
        from src.ml.pointnet.inference import PointNet3DAnalyzer

        analyzer = PointNet3DAnalyzer(model_path=None, feature_dim=128)

        content = "\n".join(
            f"{np.random.randn()} {np.random.randn()} {np.random.randn()}"
            for _ in range(100)
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xyz", delete=False
        ) as f:
            f.write(content)
            f.flush()
            tmp_path = f.name

        try:
            result = analyzer.extract_features(tmp_path)
            assert result["status"] == "model_unavailable"
            assert result["dimension"] == 128
            assert len(result["vector"]) == 128
        finally:
            os.unlink(tmp_path)


class TestSupportedFormats:
    def test_returns_expected_list(self):
        from src.ml.pointnet.inference import PointNet3DAnalyzer

        formats = PointNet3DAnalyzer.supported_formats()
        assert formats == [".stl", ".obj", ".ply", ".xyz"]


# ------------------------------------------------------------------
# Torch-dependent model tests
# ------------------------------------------------------------------


class TestPointNetEncoder:
    def test_output_shape(self):
        torch = pytest.importorskip("torch")
        from src.ml.pointnet.model import PointNetEncoder

        encoder = PointNetEncoder(input_dim=3, global_feat_dim=1024)
        encoder.eval()
        x = torch.randn(2, 1024, 3)
        with torch.no_grad():
            global_feat, input_t, feat_t = encoder(x)

        assert global_feat.shape == (2, 1024)
        assert input_t.shape == (2, 3, 3)
        assert feat_t.shape == (2, 128, 128)


class TestPointNetClassifier:
    def test_output_shape(self):
        torch = pytest.importorskip("torch")
        from src.ml.pointnet.model import PointNetClassifier

        model = PointNetClassifier(num_classes=8, input_dim=3, global_feat_dim=1024)
        model.eval()
        x = torch.randn(2, 1024, 3)
        with torch.no_grad():
            logits, global_feat, feat_t = model(x)

        assert logits.shape == (2, 8)
        assert global_feat.shape == (2, 1024)


class TestPointNetFeatureExtractor:
    def test_output_shape(self):
        torch = pytest.importorskip("torch")
        from src.ml.pointnet.model import PointNetFeatureExtractor

        extractor = PointNetFeatureExtractor(input_dim=3, feature_dim=256)
        extractor.eval()
        x = torch.randn(2, 1024, 3)
        with torch.no_grad():
            features = extractor(x)

        assert features.shape == (2, 256)

    def test_output_normalized(self):
        torch = pytest.importorskip("torch")
        from src.ml.pointnet.model import PointNetFeatureExtractor

        extractor = PointNetFeatureExtractor(input_dim=3, feature_dim=256)
        extractor.eval()
        x = torch.randn(4, 512, 3)
        with torch.no_grad():
            features = extractor(x)

        norms = torch.norm(features, p=2, dim=1)
        torch.testing.assert_close(norms, torch.ones(4), atol=1e-5, rtol=1e-5)
