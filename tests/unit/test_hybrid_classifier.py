"""
FilenameClassifier 和 HybridClassifier 单元测试
"""

import pytest
from pathlib import Path

# 测试数据
TEST_CASES = [
    # (filename, expected_extracted_name, expected_label_match)
    ("J2925001-01人孔v2.dxf", "人孔", True),
    ("J0224025-06-01-03出料凸缘v2.dxf", "出料凸缘", True),
    ("J0224071-11捕集器组件v2.dxf", "捕集器组件", True),
    ("BTJ01239901522-00拖轮组件v1.dxf", "拖轮组件", True),
    ("BTJ01231201522-00拖车DN1500v1.dxf", "拖车", True),
    ("BTJ01231101522-00自动进料装置v1.dxf", "自动进料装置", True),
    ("比较_LTJ012306102-0084调节螺栓v1 vs LTJ012306102-0084调节螺栓v2.dxf", "调节螺栓", True),
    ("J0224070-04-07捕集口v2.dxf", "捕集口", True),
    ("LTJ012306102-0084调节螺栓v1.dxf", "调节螺栓", True),
    ("J0224036-12真空组件v2.dxf", "真空组件", True),
    # 边界情况
    ("random_file.dxf", None, False),
    ("12345.dxf", None, False),
    ("", None, False),
]


class TestFilenameClassifier:
    """FilenameClassifier 测试"""

    @pytest.fixture
    def classifier(self):
        from src.ml.filename_classifier import FilenameClassifier
        return FilenameClassifier()

    def test_init(self, classifier):
        """测试初始化"""
        assert classifier is not None
        assert len(classifier.synonyms) > 0
        assert len(classifier.matcher) > 0

    @pytest.mark.parametrize("filename,expected_name,_", TEST_CASES)
    def test_extract_part_name(self, classifier, filename, expected_name, _):
        """测试零件名提取"""
        if not filename:
            assert classifier.extract_part_name(filename) is None
            return

        extracted = classifier.extract_part_name(filename)
        if expected_name:
            assert extracted is not None, f"Failed to extract from {filename}"
            assert expected_name in extracted or extracted in expected_name, \
                f"Expected '{expected_name}' in '{extracted}' for {filename}"
        else:
            # 对于无法提取的情况，允许返回 None 或无意义的结果
            pass

    @pytest.mark.parametrize("filename,_,should_match", TEST_CASES)
    def test_predict(self, classifier, filename, _, should_match):
        """测试预测"""
        if not filename:
            result = classifier.predict(filename)
            assert result["status"] == "extraction_failed"
            return

        result = classifier.predict(filename)

        if should_match:
            assert result["label"] is not None, f"No label for {filename}"
            assert result["confidence"] > 0, f"Zero confidence for {filename}"
            assert result["status"] == "matched"
        # 对于不应匹配的情况，不做强制断言

    def test_predict_batch(self, classifier):
        """测试批量预测"""
        filenames = [tc[0] for tc in TEST_CASES if tc[0]]
        results = classifier.predict_batch(filenames)
        assert len(results) == len(filenames)

    def test_singleton(self):
        """测试单例模式"""
        from src.ml.filename_classifier import get_filename_classifier, reset_filename_classifier

        reset_filename_classifier()
        c1 = get_filename_classifier()
        c2 = get_filename_classifier()
        assert c1 is c2


class TestHybridClassifier:
    """HybridClassifier 测试"""

    @pytest.fixture
    def classifier(self):
        from src.ml.hybrid_classifier import HybridClassifier
        return HybridClassifier()

    def test_init(self, classifier):
        """测试初始化"""
        assert classifier is not None
        assert classifier.filename_weight > 0
        assert classifier.graph2d_weight > 0

    def test_classify_filename_only(self, classifier):
        """测试仅文件名分类"""
        result = classifier.classify("J2925001-01人孔v2.dxf")

        assert result.label is not None
        assert result.confidence > 0
        assert result.filename_prediction is not None
        assert "filename" in result.decision_path[0] or "fusion" in str(result.decision_path)

    def test_classify_with_graph2d_result(self, classifier):
        """测试带 Graph2D 结果的分类"""
        graph2d_result = {
            "label": "传动件",
            "confidence": 0.17,
            "status": "ok",
        }

        result = classifier.classify(
            "J2925001-01人孔v2.dxf",
            graph2d_result=graph2d_result,
        )

        # 文件名置信度高，应该采用文件名结果
        assert result.label is not None
        assert result.graph2d_prediction is not None

    def test_classify_conflict_resolution(self, classifier):
        """测试冲突解决"""
        # 低置信度 Graph2D 应该被高置信度文件名覆盖
        graph2d_result = {
            "label": "传动件",
            "confidence": 0.17,
        }

        result = classifier.classify(
            "J2925001-01人孔v2.dxf",
            graph2d_result=graph2d_result,
        )

        # 文件名应该赢得冲突
        assert "人孔" in (result.label or "")

    def test_to_dict(self, classifier):
        """测试结果序列化"""
        result = classifier.classify("J2925001-01人孔v2.dxf")
        d = result.to_dict()

        assert "label" in d
        assert "confidence" in d
        assert "source" in d
        assert "decision_path" in d

    def test_singleton(self):
        """测试单例模式"""
        from src.ml.hybrid_classifier import get_hybrid_classifier, reset_hybrid_classifier

        reset_hybrid_classifier()
        c1 = get_hybrid_classifier()
        c2 = get_hybrid_classifier()
        assert c1 is c2


class TestIntegration:
    """集成测试"""

    def test_review_data_validation(self):
        """验证复核数据"""
        import csv

        review_path = Path("reports/experiments/20260123/soft_override_reviewed_20260124.csv")
        if not review_path.exists():
            pytest.skip("Review data not found")

        from src.ml.hybrid_classifier import get_hybrid_classifier

        classifier = get_hybrid_classifier()

        with open(review_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        correct = 0
        total = 0

        for row in rows:
            filename = Path(row.get("file", "")).name
            expected_label = row.get("correct_label", "")

            if not filename or not expected_label:
                continue

            result = classifier.classify(filename)
            total += 1

            if result.label and expected_label.lower() in result.label.lower():
                correct += 1
            elif result.label and result.label.lower() in expected_label.lower():
                correct += 1

        accuracy = correct / total if total > 0 else 0
        print(f"\nValidation: {correct}/{total} = {accuracy:.2%}")

        # 验收标准: >= 85%
        assert accuracy >= 0.85, f"Accuracy {accuracy:.2%} < 85%"
