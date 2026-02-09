"""Additional tests for part_classifier to improve coverage.

Targets uncovered code paths in src/ml/part_classifier.py:
- ClassificationResult dataclass
- PartClassifier: __init__, _load_model, _infer_version, extract_features, predict
- PartClassifierV16: cache, speed modes, DWG conversion, predict_batch
- Global singleton functions

Note: These tests use extensive mocking to avoid requiring torch installation.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest


# --- Mock torch module before importing part_classifier ---

def _setup_torch_mock():
    """Set up comprehensive mock for torch module."""
    mock_torch = MagicMock()
    mock_torch.device.return_value = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False
    mock_torch.tensor.return_value = MagicMock()
    mock_torch.float32 = "float32"
    mock_torch.float16 = "float16"
    mock_torch.inference_mode.return_value.__enter__ = MagicMock()
    mock_torch.inference_mode.return_value.__exit__ = MagicMock()
    mock_torch.softmax.return_value = MagicMock()
    mock_torch.sort.return_value = (MagicMock(), MagicMock())
    mock_torch.randn.return_value = MagicMock()
    mock_torch.zeros.return_value = MagicMock()
    mock_torch.ones.return_value = MagicMock()
    mock_torch.save = MagicMock()
    mock_torch.load = MagicMock()
    mock_torch.jit.trace = MagicMock(side_effect=lambda m, x: m)
    mock_torch.cat.return_value = MagicMock()
    mock_torch.sigmoid.return_value = MagicMock()

    # Mock nn module
    mock_nn = MagicMock()
    mock_nn.Module = MagicMock
    mock_nn.Linear = MagicMock()
    mock_nn.BatchNorm1d = MagicMock()
    mock_nn.BatchNorm2d = MagicMock()
    mock_nn.ReLU = MagicMock()
    mock_nn.LeakyReLU = MagicMock()
    mock_nn.Dropout = MagicMock()
    mock_nn.Sequential = MagicMock()
    mock_nn.Conv2d = MagicMock()
    mock_nn.MaxPool2d = MagicMock()
    mock_nn.AdaptiveAvgPool2d = MagicMock()
    mock_nn.Flatten = MagicMock()
    mock_nn.Parameter = MagicMock()
    mock_torch.nn = mock_nn

    return mock_torch


# --- Test ClassificationResult without torch ---


class TestClassificationResultNoTorch:
    """Tests for ClassificationResult dataclass (no torch needed)."""

    def test_default_values(self):
        """ClassificationResult has correct defaults."""
        # Import with mocked torch
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            from src.ml.part_classifier import ClassificationResult

            result = ClassificationResult(
                category="轴类",
                confidence=0.95,
                probabilities={"轴类": 0.95, "其他": 0.05},
            )
            assert result.category == "轴类"
            assert result.confidence == 0.95
            assert result.features is None
            assert result.model_version == "v2"
            assert result.needs_review is False
            assert result.review_reason is None
            assert result.top2_category is None
            assert result.top2_confidence is None

    def test_with_all_fields(self):
        """ClassificationResult with all fields."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            from src.ml.part_classifier import ClassificationResult

            result = ClassificationResult(
                category="传动件",
                confidence=0.8,
                probabilities={"传动件": 0.8, "连接件": 0.15, "其他": 0.05},
                features={"feature1": 0.5},
                model_version="v16",
                needs_review=True,
                review_reason="置信度低于阈值",
                top2_category="连接件",
                top2_confidence=0.15,
            )
            assert result.category == "传动件"
            assert result.needs_review is True
            assert result.top2_category == "连接件"


# --- Test PartClassifierV16 configuration without loading models ---


class TestPartClassifierV16Config:
    """Tests for PartClassifierV16 configuration (no model loading)."""

    def test_invalid_speed_mode_raises_error(self):
        """Invalid speed mode raises ValueError."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            # Need to reimport after patching
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            with pytest.raises(ValueError, match="无效速度模式"):
                PartClassifierV16(speed_mode="invalid_mode")

    def test_valid_speed_modes(self):
        """All valid speed modes can be instantiated."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            for mode in ["accurate", "balanced", "fast", "v6_only"]:
                classifier = PartClassifierV16(speed_mode=mode)
                assert classifier.speed_mode == mode

    def test_cache_operations(self):
        """Cache operations work correctly."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            classifier = PartClassifierV16(enable_cache=True, cache_size=10)

            # Test cache key generation
            with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as f:
                f.write(b"test")
                temp_path = f.name

            try:
                key = classifier._get_file_cache_key(temp_path)
                assert len(key) == 16

                # Test cache put/get
                features = np.zeros(48, dtype=np.float32)
                image = np.zeros((128, 128), dtype=np.float32)

                classifier._cache_put(key, features, image)

                cached_features, cached_image = classifier._cache_get(key)
                assert cached_features is not None
                assert cached_image is not None
                np.testing.assert_array_equal(cached_features, features)
            finally:
                os.unlink(temp_path)

    def test_cache_miss(self):
        """Cache miss returns (None, None)."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            classifier = PartClassifierV16(enable_cache=True)
            features, image = classifier._cache_get("nonexistent_key")
            assert features is None
            assert image is None

    def test_cache_disabled(self):
        """Cache operations are no-op when disabled."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            classifier = PartClassifierV16(enable_cache=False)

            # Should not store
            classifier._cache_put("key", np.zeros(48), np.zeros((128, 128)))

            # Should return None
            features, image = classifier._cache_get("key")
            assert features is None
            assert image is None

    def test_cache_lru_eviction(self):
        """Cache evicts oldest entries when full."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            classifier = PartClassifierV16(enable_cache=True, cache_size=3)

            # Fill cache
            for i in range(5):
                classifier._cache_put(f"key{i}", np.zeros(48) + i, None)

            # First two keys should be evicted
            f0, _ = classifier._cache_get("key0")
            f1, _ = classifier._cache_get("key1")
            f4, _ = classifier._cache_get("key4")

            assert f0 is None  # Evicted
            assert f1 is None  # Evicted
            assert f4 is not None  # Still present

    def test_clear_cache(self):
        """clear_cache empties all caches."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            classifier = PartClassifierV16(enable_cache=True)
            classifier._cache_put("key", np.zeros(48), np.zeros((128, 128)))
            classifier._cache_stats["hits"] = 10
            classifier._cache_stats["misses"] = 5

            classifier.clear_cache()

            assert len(classifier._feature_cache) == 0
            assert len(classifier._image_cache) == 0
            assert classifier._cache_stats["hits"] == 0
            assert classifier._cache_stats["misses"] == 0

    def test_cache_stats(self):
        """get_cache_stats returns correct statistics."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            classifier = PartClassifierV16(enable_cache=True, cache_size=100)

            classifier._cache_put("key1", np.zeros(48), None)
            classifier._cache_get("key1")  # Hit
            classifier._cache_get("key2")  # Miss

            stats = classifier.get_cache_stats()
            assert stats["hits"] == 1
            assert stats["misses"] == 1
            assert stats["hit_rate"] == 0.5
            assert stats["cache_size"] == 1

    def test_fp16_properties(self):
        """FP16 properties return correct values."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            classifier = PartClassifierV16(use_fp16=False)
            assert classifier.use_fp16 is False
            assert classifier.dtype_str == "fp32"

    def test_jit_property(self):
        """JIT property returns correct value."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            classifier = PartClassifierV16(use_jit=True)
            assert classifier.use_jit is True

            classifier2 = PartClassifierV16(use_jit=False)
            assert classifier2.use_jit is False

    def test_check_needs_review_low_confidence(self):
        """_check_needs_review detects low confidence."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            classifier = PartClassifierV16(confidence_threshold=0.85)

            needs_review, reason = classifier._check_needs_review(
                "轴类", 0.7, "传动件", 0.2
            )

            assert needs_review is True
            assert "置信度" in reason

    def test_check_needs_review_small_margin(self):
        """_check_needs_review detects small margin between top1 and top2."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            classifier = PartClassifierV16()

            needs_review, reason = classifier._check_needs_review(
                "轴类", 0.9, "传动件", 0.85  # Margin = 0.05 < 0.15
            )

            assert needs_review is True
            assert "差距" in reason

    def test_check_needs_review_ambiguous_pair(self):
        """_check_needs_review detects known ambiguous pairs."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            classifier = PartClassifierV16()

            needs_review, reason = classifier._check_needs_review(
                "连接件", 0.9, "传动件", 0.05  # Known ambiguous pair
            )

            assert needs_review is True
            assert "边界案例" in reason

    def test_check_needs_review_no_issues(self):
        """_check_needs_review returns False when no issues."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            classifier = PartClassifierV16(confidence_threshold=0.5)

            needs_review, reason = classifier._check_needs_review(
                "轴类", 0.95, "壳体类", 0.03  # High confidence, large margin, not ambiguous
            )

            assert needs_review is False
            assert reason is None

    def test_get_render_size(self):
        """_get_render_size returns correct size for speed mode."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            accurate_classifier = PartClassifierV16(speed_mode="accurate")
            assert accurate_classifier._get_render_size() == 128

            fast_classifier = PartClassifierV16(speed_mode="fast")
            assert fast_classifier._get_render_size() == 96


# --- Test V16 Rendering without models ---


class TestV16RenderNoModel:
    """Tests for V16 rendering (no model loading)."""

    def test_render_dxf_fast_empty(self):
        """_render_dxf_fast returns None for empty drawing."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            classifier = PartClassifierV16()

            mock_msp = MagicMock()
            mock_msp.__iter__ = MagicMock(return_value=iter([]))

            mock_doc = MagicMock()
            mock_doc.modelspace.return_value = mock_msp

            with patch("ezdxf.readfile", return_value=mock_doc):
                result = classifier._render_dxf_fast("test.dxf")

            assert result is None

    def test_render_dxf_fast_with_line(self):
        """_render_dxf_fast renders LINE entity."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            classifier = PartClassifierV16(speed_mode="fast")

            mock_line = MagicMock()
            mock_line.dxftype.return_value = "LINE"
            mock_line.dxf.start.x = 0
            mock_line.dxf.start.y = 0
            mock_line.dxf.end.x = 100
            mock_line.dxf.end.y = 100

            mock_msp = MagicMock()
            mock_msp.__iter__ = MagicMock(return_value=iter([mock_line]))

            mock_doc = MagicMock()
            mock_doc.modelspace.return_value = mock_msp

            result = classifier._render_dxf_fast("test.dxf", ezdxf_doc=mock_doc)

            assert result is not None
            assert result.shape == (96, 96)  # Fast mode uses 96x96

    def test_render_dxf_fast_with_circle(self):
        """_render_dxf_fast renders CIRCLE entity."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            classifier = PartClassifierV16(speed_mode="fast")

            mock_circle = MagicMock()
            mock_circle.dxftype.return_value = "CIRCLE"
            mock_circle.dxf.center.x = 50
            mock_circle.dxf.center.y = 50
            mock_circle.dxf.radius = 25

            mock_msp = MagicMock()
            mock_msp.__iter__ = MagicMock(return_value=iter([mock_circle]))

            mock_doc = MagicMock()
            mock_doc.modelspace.return_value = mock_msp

            result = classifier._render_dxf_fast("test.dxf", ezdxf_doc=mock_doc)

            assert result is not None

    def test_render_dxf_fast_with_arc(self):
        """_render_dxf_fast renders ARC entity."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            classifier = PartClassifierV16(speed_mode="fast")

            mock_arc = MagicMock()
            mock_arc.dxftype.return_value = "ARC"
            mock_arc.dxf.center.x = 50
            mock_arc.dxf.center.y = 50
            mock_arc.dxf.radius = 25
            mock_arc.dxf.start_angle = 0
            mock_arc.dxf.end_angle = 90

            mock_msp = MagicMock()
            mock_msp.__iter__ = MagicMock(return_value=iter([mock_arc]))

            mock_doc = MagicMock()
            mock_doc.modelspace.return_value = mock_msp

            result = classifier._render_dxf_fast("test.dxf", ezdxf_doc=mock_doc)

            assert result is not None

    def test_render_dxf_fast_with_polyline(self):
        """_render_dxf_fast renders POLYLINE entity."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            classifier = PartClassifierV16(speed_mode="fast")

            mock_polyline = MagicMock()
            mock_polyline.dxftype.return_value = "LWPOLYLINE"
            mock_polyline.get_points.return_value = [(0, 0), (100, 0), (100, 100), (0, 100)]

            mock_msp = MagicMock()
            mock_msp.__iter__ = MagicMock(return_value=iter([mock_polyline]))

            mock_doc = MagicMock()
            mock_doc.modelspace.return_value = mock_msp

            result = classifier._render_dxf_fast("test.dxf", ezdxf_doc=mock_doc)

            assert result is not None

    def test_render_dxf_exception(self):
        """_render_dxf_fast returns None on exception."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            classifier = PartClassifierV16()

            with patch("ezdxf.readfile", side_effect=Exception("Read error")):
                result = classifier._render_dxf_fast("test.dxf")

            assert result is None


# --- Test V16 model loading errors ---


class TestV16ModelLoadingErrors:
    """Tests for V16 model loading error paths."""

    def test_v6_not_found_raises_error(self):
        """FileNotFoundError when V6 model doesn't exist."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            with tempfile.TemporaryDirectory() as tmp_dir:
                classifier = PartClassifierV16(model_dir=tmp_dir)

                with pytest.raises(FileNotFoundError, match="V6模型不存在"):
                    classifier._load_models()

    def test_predict_batch_empty(self):
        """predict_batch returns empty list for empty input."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            classifier = PartClassifierV16()
            results = classifier.predict_batch([])
            assert results == []


# --- Test file cache key generation edge cases ---


class TestFileCacheKeyEdgeCases:
    """Tests for file cache key generation edge cases."""

    def test_cache_key_nonexistent_file(self):
        """Cache key generated for nonexistent file (falls back to path only)."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            classifier = PartClassifierV16()
            key = classifier._get_file_cache_key("/nonexistent/path/file.dxf")
            assert len(key) == 16  # Should still generate a key


# --- Test speed mode configuration ---


class TestSpeedModeConfiguration:
    """Tests for speed mode configuration."""

    def test_speed_config_accurate(self):
        """Accurate mode has correct configuration."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            classifier = PartClassifierV16(speed_mode="accurate")
            config = classifier._speed_config

            assert config["v14_folds"] == 5
            assert config["use_fast_render"] is False
            assert config["img_size"] == 128

    def test_speed_config_balanced(self):
        """Balanced mode has correct configuration."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            classifier = PartClassifierV16(speed_mode="balanced")
            config = classifier._speed_config

            assert config["v14_folds"] == 3
            assert config["use_fast_render"] is True
            assert config["img_size"] == 96

    def test_speed_config_fast(self):
        """Fast mode has correct configuration."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            classifier = PartClassifierV16(speed_mode="fast")
            config = classifier._speed_config

            assert config["v14_folds"] == 1
            assert config["use_fast_render"] is True

    def test_speed_config_v6_only(self):
        """V6-only mode has correct configuration."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            classifier = PartClassifierV16(speed_mode="v6_only")
            config = classifier._speed_config

            assert config["v14_folds"] == 0


# --- Test constants and class attributes ---


class TestV16Constants:
    """Tests for V16 class constants."""

    def test_categories(self):
        """CATEGORIES constant is correct."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            expected = ["传动件", "其他", "壳体类", "轴类", "连接件"]
            assert PartClassifierV16.CATEGORIES == expected

    def test_image_sizes(self):
        """Image size constants are correct."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            assert PartClassifierV16.IMG_SIZE == 128
            assert PartClassifierV16.IMG_SIZE_FAST == 96

    def test_thresholds(self):
        """Threshold constants are correct."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            assert PartClassifierV16.CONFIDENCE_THRESHOLD == 0.85
            assert PartClassifierV16.MARGIN_THRESHOLD == 0.15

    def test_known_ambiguous_pairs(self):
        """Known ambiguous pairs are defined."""
        mock_torch = _setup_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
            import importlib
            import src.ml.part_classifier
            importlib.reload(src.ml.part_classifier)
            from src.ml.part_classifier import PartClassifierV16

            assert len(PartClassifierV16.KNOWN_AMBIGUOUS_PAIRS) >= 2
            assert ("连接件", "传动件") in PartClassifierV16.KNOWN_AMBIGUOUS_PAIRS
