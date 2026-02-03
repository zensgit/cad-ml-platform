"""
部件分类器单元测试
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# 测试模块导入
from src.ml.part_classifier import (
    ClassificationResult,
    PartClassifier,
    classify_part,
    get_part_classifier,
)


class TestClassificationResult:
    """ClassificationResult 数据类测试"""

    def test_basic_creation(self):
        """测试基本创建"""
        result = ClassificationResult(
            category="法兰",
            confidence=0.876,
            probabilities={"法兰": 0.876, "轴承": 0.05, "其他": 0.074},
        )
        assert result.category == "法兰"
        assert result.confidence == 0.876
        assert result.probabilities["法兰"] == 0.876
        assert result.features is None

    def test_with_features(self):
        """测试带特征的创建"""
        result = ClassificationResult(
            category="组件",
            confidence=0.9,
            probabilities={"组件": 0.9},
            features={"line_ratio": 0.5, "circle_ratio": 0.2},
        )
        assert result.features is not None
        assert result.features["line_ratio"] == 0.5


class TestPartClassifierFeatureExtraction:
    """特征提取测试"""

    @pytest.fixture
    def classifier_mock(self):
        """创建一个模拟的分类器（不加载模型）"""
        with patch.object(PartClassifier, "_load_model"):
            classifier = PartClassifier.__new__(PartClassifier)
            classifier.model_path = Path("fake_model.pt")
            classifier.model = None
            classifier.id_to_label = None
            classifier.device = "cpu"
            return classifier

    def test_extract_features_file_not_found(self, classifier_mock):
        """测试文件不存在时返回None"""
        result = classifier_mock.extract_features("/nonexistent/file.dxf")
        assert result is None

    @pytest.mark.skipif(
        not Path("data/training").exists(),
        reason="训练数据目录不存在",
    )
    def test_extract_features_real_file(self, classifier_mock):
        """测试从真实DXF文件提取特征"""
        # 找一个真实的DXF文件
        training_dir = Path("data/training")
        dxf_files = list(training_dir.glob("**/*.dxf"))
        if not dxf_files:
            pytest.skip("没有找到DXF文件")

        features = classifier_mock.extract_features(str(dxf_files[0]))
        if features is not None:
            # 验证特征维度
            assert features.shape == (28,), f"期望28维特征，得到{features.shape}"
            # 验证特征值范围
            assert np.all(features >= 0), "特征值应该非负"
            assert np.all(features <= 2), "特征值应该在合理范围内"


class TestPartClassifierIntegration:
    """集成测试 - 需要真实模型"""

    @pytest.fixture
    def real_classifier(self):
        """加载真实的分类器"""
        model_path = Path("models/cad_classifier_v2.pt")
        if not model_path.exists():
            pytest.skip("模型文件不存在")
        return PartClassifier(str(model_path))

    @pytest.mark.skipif(
        not Path("models/cad_classifier_v2.pt").exists(),
        reason="模型文件不存在",
    )
    def test_model_loading(self, real_classifier):
        """测试模型加载"""
        assert real_classifier.model is not None
        assert real_classifier.id_to_label is not None
        assert real_classifier.num_classes == 7

    @pytest.mark.skipif(
        not Path("models/cad_classifier_v2.pt").exists(),
        reason="模型文件不存在",
    )
    def test_categories(self, real_classifier):
        """测试类别标签"""
        expected_categories = {"其他", "弹簧", "法兰", "组件", "罐体", "轴承", "阀体"}
        actual_categories = set(real_classifier.id_to_label.values())
        assert actual_categories == expected_categories

    @pytest.mark.skipif(
        not Path("models/cad_classifier_v2.pt").exists()
        or not Path("data/training").exists(),
        reason="模型或训练数据不存在",
    )
    def test_predict_flange(self, real_classifier):
        """测试法兰分类"""
        flange_dir = Path("data/training/法兰")
        if not flange_dir.exists():
            pytest.skip("法兰目录不存在")

        dxf_files = list(flange_dir.glob("*.dxf"))
        if not dxf_files:
            pytest.skip("没有法兰DXF文件")

        result = real_classifier.predict(str(dxf_files[0]))
        assert result is not None
        assert result.category == "法兰"
        assert result.confidence > 0.5

    @pytest.mark.skipif(
        not Path("models/cad_classifier_v2.pt").exists()
        or not Path("data/training").exists(),
        reason="模型或训练数据不存在",
    )
    def test_predict_batch(self, real_classifier):
        """测试批量预测"""
        training_dir = Path("data/training")
        dxf_files = list(training_dir.glob("**/*.dxf"))[:3]
        if len(dxf_files) < 2:
            pytest.skip("DXF文件不足")

        results = real_classifier.predict_batch([str(f) for f in dxf_files])
        assert len(results) == len(dxf_files)
        assert all(r is not None for r in results)


class TestConvenienceFunctions:
    """便捷函数测试"""

    @pytest.mark.skipif(
        not Path("models/cad_classifier_v2.pt").exists(),
        reason="模型文件不存在",
    )
    def test_get_part_classifier_singleton(self):
        """测试单例模式"""
        # 重置单例
        import src.ml.part_classifier as module

        module._classifier = None

        classifier1 = get_part_classifier()
        classifier2 = get_part_classifier()
        assert classifier1 is classifier2

    @pytest.mark.skipif(
        not Path("models/cad_classifier_v2.pt").exists()
        or not Path("data/training").exists(),
        reason="模型或训练数据不存在",
    )
    def test_classify_part_function(self):
        """测试便捷分类函数"""
        # 重置单例
        import src.ml.part_classifier as module

        module._classifier = None

        training_dir = Path("data/training")
        dxf_files = list(training_dir.glob("**/*.dxf"))
        if not dxf_files:
            pytest.skip("没有DXF文件")

        result = classify_part(str(dxf_files[0]))
        assert result is not None
        assert result.category in {"其他", "弹簧", "法兰", "组件", "罐体", "轴承", "阀体"}
        assert 0 <= result.confidence <= 1


class TestCADAnalyzerIntegration:
    """CADAnalyzer 集成测试"""

    def test_ml_classifier_loader(self):
        """测试ML分类器加载器"""
        from src.core.analyzer import _get_ml_classifier

        # 重置加载状态
        import src.core.analyzer as module

        module._ml_classifier = None
        module._ml_classifier_loaded = False

        classifier = _get_ml_classifier()
        if Path("models/cad_classifier_v2.pt").exists():
            assert classifier is not None
        else:
            assert classifier is None

    def test_ml_classifier_loader_cached(self):
        """测试ML分类器缓存"""
        from src.core.analyzer import _get_ml_classifier

        # 重置加载状态
        import src.core.analyzer as module

        module._ml_classifier = None
        module._ml_classifier_loaded = False

        classifier1 = _get_ml_classifier()
        classifier2 = _get_ml_classifier()
        assert classifier1 is classifier2


class TestEdgeCases:
    """边界情况测试"""

    def test_model_file_not_found(self):
        """测试模型文件不存在"""
        with pytest.raises(FileNotFoundError):
            PartClassifier("/nonexistent/model.pt")

    @pytest.mark.skipif(
        not Path("models/cad_classifier_v2.pt").exists(),
        reason="模型文件不存在",
    )
    def test_predict_invalid_file(self):
        """测试无效文件"""
        classifier = PartClassifier("models/cad_classifier_v2.pt")
        result = classifier.predict("/nonexistent/file.dxf")
        assert result is None

    @pytest.mark.skipif(
        not Path("models/cad_classifier_v2.pt").exists(),
        reason="模型文件不存在",
    )
    def test_predict_empty_batch(self):
        """测试空批量"""
        classifier = PartClassifier("models/cad_classifier_v2.pt")
        results = classifier.predict_batch([])
        assert results == []
