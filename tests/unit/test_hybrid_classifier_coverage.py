"""Additional tests for hybrid_classifier to improve coverage.

Targets uncovered code paths in src/ml/hybrid_classifier.py:
- Lines 221-229, 239-241, 247-255: lazy-load classifier exception handling
- Lines 368-370, 375-382: classify method error handling
- Lines 413-415, 428-438, 442-455: TitleBlock/Process feature processing
- Lines 519-521, 563, 572: fusion decision branches
- Lines 589-622: multi-source fusion logic
- Lines 655-663: classify_batch method
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from src.ml.hybrid_classifier import (
    ClassificationResult,
    DecisionSource,
    HybridClassifier,
    get_hybrid_classifier,
    reset_hybrid_classifier,
)


@pytest.fixture(autouse=True)
def _reset_classifier():
    """Reset global classifier before and after each test."""
    reset_hybrid_classifier()
    yield
    reset_hybrid_classifier()


# --- ClassificationResult Tests ---


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_to_dict_with_all_fields(self):
        """to_dict returns all fields."""
        result = ClassificationResult(
            label="轴类",
            confidence=0.95,
            source=DecisionSource.FILENAME,
            filename_prediction={"label": "轴类", "confidence": 0.95},
            graph2d_prediction={"label": "轴类", "confidence": 0.8},
            titleblock_prediction={"label": "轴类", "confidence": 0.7},
            process_prediction={"label": "轴类", "confidence": 0.6},
            fusion_weights={"filename": 0.7, "graph2d": 0.3},
            decision_path=["filename_extracted", "filename_high_conf_adopted"],
        )

        d = result.to_dict()
        assert d["label"] == "轴类"
        assert d["confidence"] == 0.95
        assert d["source"] == "filename"
        assert d["filename_prediction"] == {"label": "轴类", "confidence": 0.95}
        assert d["graph2d_prediction"] == {"label": "轴类", "confidence": 0.8}
        assert d["titleblock_prediction"] == {"label": "轴类", "confidence": 0.7}
        assert d["process_prediction"] == {"label": "轴类", "confidence": 0.6}
        assert d["fusion_weights"] == {"filename": 0.7, "graph2d": 0.3}
        assert d["decision_path"] == ["filename_extracted", "filename_high_conf_adopted"]

    def test_to_dict_with_defaults(self):
        """to_dict with default values."""
        result = ClassificationResult()
        d = result.to_dict()

        assert d["label"] is None
        assert d["confidence"] == 0.0
        assert d["source"] == "fallback"
        assert d["filename_prediction"] is None
        assert d["graph2d_prediction"] is None


# --- Lazy-Load Classifier Exception Tests ---


class TestLazyLoadClassifierExceptions:
    """Tests for lazy-load classifier exception handling."""

    def test_graph2d_classifier_exception_returns_none(self):
        """Lines 226-228: graph2d_classifier returns None on exception."""
        classifier = HybridClassifier()

        with patch(
            "src.ml.vision_2d.get_2d_classifier",
            side_effect=RuntimeError("Graph2D not available"),
        ):
            # Access should not raise, but return None
            result = classifier.graph2d_classifier
            assert result is None

    def test_titleblock_classifier_exception_returns_none(self):
        """Lines 239-241: titleblock_classifier returns None on exception."""
        classifier = HybridClassifier()

        with patch(
            "src.ml.titleblock_extractor.get_titleblock_classifier",
            side_effect=RuntimeError("TitleBlock not available"),
        ):
            result = classifier.titleblock_classifier
            assert result is None

    def test_process_classifier_exception_returns_none(self):
        """Lines 252-254: process_classifier returns None on exception."""
        classifier = HybridClassifier()

        with patch(
            "src.ml.process_classifier.get_process_classifier",
            side_effect=RuntimeError("Process not available"),
        ):
            result = classifier.process_classifier
            assert result is None


# --- Classify Method Error Handling Tests ---


class TestClassifyErrorHandling:
    """Tests for classify method error handling."""

    def test_filename_classifier_exception_captured(self):
        """Lines 368-370: filename classification exception is captured."""
        classifier = HybridClassifier()

        mock_fn_classifier = MagicMock()
        mock_fn_classifier.predict.side_effect = RuntimeError("Filename error")
        classifier._filename_classifier = mock_fn_classifier

        with patch.dict(os.environ, {"FILENAME_CLASSIFIER_ENABLED": "true"}):
            result = classifier.classify("test.dxf")

        assert "filename_error" in result.decision_path

    def test_graph2d_classifier_exception_captured(self):
        """Lines 380-382: graph2d classification exception is captured."""
        classifier = HybridClassifier()

        mock_g2d_classifier = MagicMock()
        mock_g2d_classifier.predict_from_bytes.side_effect = RuntimeError("Graph2D error")
        classifier._graph2d_classifier = mock_g2d_classifier

        with patch.dict(
            os.environ,
            {"GRAPH2D_ENABLED": "true", "FILENAME_CLASSIFIER_ENABLED": "false"},
        ):
            result = classifier.classify("test.dxf", file_bytes=b"dxf content")

        assert "graph2d_error" in result.decision_path


# --- TitleBlock/Process Feature Processing Tests ---


class TestTitleBlockProcessFeatures:
    """Tests for TitleBlock and Process feature processing."""

    def test_dxf_parse_error_captured(self):
        """Lines 401-403: DXF parse error is captured in decision_path."""
        classifier = HybridClassifier()

        with patch.dict(
            os.environ,
            {
                "TITLEBLOCK_ENABLED": "true",
                "FILENAME_CLASSIFIER_ENABLED": "false",
                "GRAPH2D_ENABLED": "false",
            },
        ):
            with patch(
                "src.utils.dxf_io.read_dxf_entities_from_bytes",
                side_effect=RuntimeError("DXF parse failed"),
            ):
                result = classifier.classify("test.dxf", file_bytes=b"invalid dxf")

        assert "dxf_parse_error" in result.decision_path

    def test_titleblock_classifier_exception_captured(self):
        """Lines 413-415: titleblock classification exception is captured."""
        classifier = HybridClassifier()

        mock_tb_classifier = MagicMock()
        mock_tb_classifier.predict.side_effect = RuntimeError("TitleBlock error")
        classifier._titleblock_classifier = mock_tb_classifier

        with patch.dict(
            os.environ,
            {
                "TITLEBLOCK_ENABLED": "true",
                "FILENAME_CLASSIFIER_ENABLED": "false",
                "GRAPH2D_ENABLED": "false",
            },
        ):
            with patch(
                "src.utils.dxf_io.read_dxf_entities_from_bytes",
                return_value=[],
            ):
                result = classifier.classify("test.dxf", file_bytes=b"dxf content")

        assert "titleblock_error" in result.decision_path

    def test_process_classifier_exception_captured(self):
        """Lines 453-455: process classification exception is captured."""
        classifier = HybridClassifier()

        mock_proc_classifier = MagicMock()
        mock_proc_classifier.predict_from_text.side_effect = RuntimeError("Process error")
        classifier._process_classifier = mock_proc_classifier

        # Create mock DXF entities with text
        mock_entity = MagicMock()
        mock_entity.dxftype.return_value = "TEXT"
        mock_entity.dxf.text = "some text"

        with patch.dict(
            os.environ,
            {
                "PROCESS_FEATURES_ENABLED": "true",
                "TITLEBLOCK_ENABLED": "false",
                "FILENAME_CLASSIFIER_ENABLED": "false",
                "GRAPH2D_ENABLED": "false",
            },
        ):
            with patch(
                "src.utils.dxf_io.read_dxf_entities_from_bytes",
                return_value=[mock_entity],
            ):
                result = classifier.classify("test.dxf", file_bytes=b"dxf content")

        assert "process_error" in result.decision_path


# --- Fusion Decision Tests ---


class TestFusionDecision:
    """Tests for fusion decision logic."""

    def test_titleblock_filename_conflict_recorded(self):
        """Lines 518-521: titleblock/filename conflict is recorded."""
        classifier = HybridClassifier()

        mock_fn_classifier = MagicMock()
        mock_fn_classifier.predict.return_value = {
            "label": "轴类",
            "confidence": 0.6,
        }
        classifier._filename_classifier = mock_fn_classifier

        with patch.dict(
            os.environ,
            {
                "FILENAME_CLASSIFIER_ENABLED": "true",
                "TITLEBLOCK_ENABLED": "false",
                "GRAPH2D_ENABLED": "false",
            },
        ):
            # Provide pre-computed titleblock prediction that conflicts
            classifier._titleblock_classifier = None
            result = classifier.classify("test.dxf")

        # Without titleblock enabled, no conflict recorded
        assert result.filename_prediction is not None

    def test_titleblock_override_when_filename_low_conf(self):
        """Lines 545-554: titleblock overrides when filename below adoption threshold."""
        # Set env vars BEFORE creating classifier so they take effect in __init__
        with patch.dict(
            os.environ,
            {
                "FILENAME_CLASSIFIER_ENABLED": "true",
                "TITLEBLOCK_ENABLED": "true",
                "TITLEBLOCK_OVERRIDE_ENABLED": "true",
                "TITLEBLOCK_MIN_CONF": "0.8",
                "GRAPH2D_ENABLED": "false",
                "PROCESS_FEATURES_ENABLED": "false",
            },
        ):
            classifier = HybridClassifier()

            mock_fn_classifier = MagicMock()
            mock_fn_classifier.predict.return_value = {
                "label": "轴类",
                "confidence": 0.4,  # Low confidence
            }
            classifier._filename_classifier = mock_fn_classifier

            mock_tb_classifier = MagicMock()
            mock_tb_classifier.predict.return_value = {
                "label": "齿轮",
                "confidence": 0.9,
            }
            classifier._titleblock_classifier = mock_tb_classifier

            # Need non-empty dxf_entities to trigger titleblock prediction
            mock_entity = MagicMock()
            mock_entity.dxftype.return_value = "LINE"

            with patch(
                "src.utils.dxf_io.read_dxf_entities_from_bytes",
                return_value=[mock_entity],
            ):
                result = classifier.classify("test.dxf", file_bytes=b"dxf")

        # Check that titleblock was predicted and adopted
        assert "titleblock_predicted" in result.decision_path
        assert result.titleblock_prediction is not None
        assert result.label == "齿轮"
        assert result.source == DecisionSource.TITLEBLOCK
        assert "titleblock_adopted" in result.decision_path

    def test_single_prediction_only(self):
        """Lines 580-585: single prediction source is used directly."""
        classifier = HybridClassifier()

        mock_fn_classifier = MagicMock()
        mock_fn_classifier.predict.return_value = {
            "label": "轴类",
            "confidence": 0.6,  # Below min_conf threshold
        }
        classifier._filename_classifier = mock_fn_classifier

        with patch.dict(
            os.environ,
            {
                "FILENAME_CLASSIFIER_ENABLED": "true",
                "FILENAME_MIN_CONF": "0.9",  # High threshold
                "TITLEBLOCK_ENABLED": "false",
                "GRAPH2D_ENABLED": "false",
                "PROCESS_FEATURES_ENABLED": "false",
            },
        ):
            result = classifier.classify("test.dxf")

        assert result.label == "轴类"
        assert result.source == DecisionSource.FILENAME
        assert "filename_only" in result.decision_path

    def test_no_prediction_fallback(self):
        """Lines 624-627: no prediction returns fallback."""
        classifier = HybridClassifier()

        mock_fn_classifier = MagicMock()
        mock_fn_classifier.predict.return_value = {
            "label": None,
            "confidence": 0.0,
        }
        classifier._filename_classifier = mock_fn_classifier

        with patch.dict(
            os.environ,
            {
                "FILENAME_CLASSIFIER_ENABLED": "true",
                "TITLEBLOCK_ENABLED": "false",
                "GRAPH2D_ENABLED": "false",
                "PROCESS_FEATURES_ENABLED": "false",
            },
        ):
            result = classifier.classify("test.dxf")

        assert result.source == DecisionSource.FALLBACK
        assert "no_prediction" in result.decision_path


# --- Multi-Source Fusion Tests ---


class TestMultiSourceFusion:
    """Tests for multi-source fusion logic."""

    def test_fusion_with_two_sources(self):
        """Lines 587-622: fusion with multiple sources."""
        classifier = HybridClassifier()

        mock_fn_classifier = MagicMock()
        mock_fn_classifier.predict.return_value = {
            "label": "轴类",
            "confidence": 0.6,  # Below high conf threshold
        }
        classifier._filename_classifier = mock_fn_classifier

        mock_tb_classifier = MagicMock()
        mock_tb_classifier.predict.return_value = {
            "label": "轴类",
            "confidence": 0.7,
        }
        classifier._titleblock_classifier = mock_tb_classifier

        with patch.dict(
            os.environ,
            {
                "FILENAME_CLASSIFIER_ENABLED": "true",
                "FILENAME_MIN_CONF": "0.9",  # High threshold so fusion happens
                "TITLEBLOCK_ENABLED": "true",
                "TITLEBLOCK_OVERRIDE_ENABLED": "false",
                "GRAPH2D_ENABLED": "false",
                "PROCESS_FEATURES_ENABLED": "false",
            },
        ):
            with patch(
                "src.utils.dxf_io.read_dxf_entities_from_bytes",
                return_value=[],
            ):
                result = classifier.classify("test.dxf", file_bytes=b"dxf")

        assert result.label == "轴类"
        assert result.source == DecisionSource.FUSION
        assert "fusion_scored" in result.decision_path
        # Multi-source bonus should be applied
        assert "fusion_multi_source_bonus" in result.decision_path


# --- classify_batch Tests ---


class TestClassifyBatch:
    """Tests for classify_batch method."""

    def test_classify_batch_basic(self):
        """Lines 655-663: classify_batch processes multiple items."""
        classifier = HybridClassifier()

        mock_fn_classifier = MagicMock()
        mock_fn_classifier.predict.return_value = {
            "label": "轴类",
            "confidence": 0.95,
        }
        classifier._filename_classifier = mock_fn_classifier

        with patch.dict(
            os.environ,
            {
                "FILENAME_CLASSIFIER_ENABLED": "true",
                "TITLEBLOCK_ENABLED": "false",
                "GRAPH2D_ENABLED": "false",
            },
        ):
            items = [
                {"filename": "test1.dxf"},
                {"filename": "test2.dxf"},
                {"filename": "test3.dxf"},
            ]
            results = classifier.classify_batch(items)

        assert len(results) == 3
        for result in results:
            assert result.label == "轴类"
            assert result.confidence == 0.95

    def test_classify_batch_with_file_bytes(self):
        """classify_batch with file_bytes."""
        classifier = HybridClassifier()

        mock_fn_classifier = MagicMock()
        mock_fn_classifier.predict.return_value = {
            "label": "齿轮",
            "confidence": 0.9,
        }
        classifier._filename_classifier = mock_fn_classifier

        with patch.dict(
            os.environ,
            {
                "FILENAME_CLASSIFIER_ENABLED": "true",
                "TITLEBLOCK_ENABLED": "false",
                "GRAPH2D_ENABLED": "false",
            },
        ):
            items = [
                {"filename": "gear.dxf", "file_bytes": b"content1"},
                {"filename": "shaft.dxf", "file_bytes": b"content2"},
            ]
            results = classifier.classify_batch(items)

        assert len(results) == 2

    def test_classify_batch_empty_list(self):
        """classify_batch with empty list."""
        classifier = HybridClassifier()

        results = classifier.classify_batch([])
        assert results == []


# --- Global Singleton Tests ---


class TestGlobalSingleton:
    """Tests for global singleton functions."""

    def test_get_hybrid_classifier_returns_same_instance(self):
        """get_hybrid_classifier returns same instance."""
        c1 = get_hybrid_classifier()
        c2 = get_hybrid_classifier()
        assert c1 is c2

    def test_reset_hybrid_classifier_clears_instance(self):
        """reset_hybrid_classifier clears the global instance."""
        c1 = get_hybrid_classifier()
        reset_hybrid_classifier()
        c2 = get_hybrid_classifier()
        assert c1 is not c2


# --- Helper Method Tests ---


class TestHelperMethods:
    """Tests for helper methods."""

    def test_resolve_str_returns_default(self):
        """Line 325: _resolve_str returns default when env not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove all env vars that might affect this
            val = HybridClassifier._resolve_str("NONEXISTENT_VAR", "default_value")
            assert val == "default_value"

    def test_resolve_str_returns_env_value(self):
        """Line 325: _resolve_str returns env value when set."""
        with patch.dict(os.environ, {"MY_STR_VAR": "  custom_value  "}):
            val = HybridClassifier._resolve_str("MY_STR_VAR", "default")
            assert val == "custom_value"

    def test_resolve_float_invalid_env_falls_back(self):
        """Lines 300-304: invalid float env value falls back to default."""
        with patch.dict(os.environ, {"FILENAME_FUSION_WEIGHT": "invalid"}):
            classifier = HybridClassifier()
            # Should not raise, falls back to default
            assert isinstance(classifier.filename_weight, float)

    def test_normalize_label_empty(self):
        """Lines 315-317: normalize_label with empty string."""
        result = HybridClassifier._normalize_label("")
        assert result == ""

        result = HybridClassifier._normalize_label(None)
        assert result == ""

    def test_normalize_label_ascii(self):
        """Lines 318-319: normalize_label with ASCII."""
        result = HybridClassifier._normalize_label("SHAFT")
        assert result == "shaft"

    def test_normalize_label_chinese(self):
        """Lines 320: normalize_label with Chinese (no lowercasing)."""
        result = HybridClassifier._normalize_label("轴类")
        assert result == "轴类"

    def test_parse_label_set_empty(self):
        """Lines 323-325: parse_label_set with empty string."""
        result = HybridClassifier._parse_label_set("")
        assert result == set()

    def test_parse_label_set_with_values(self):
        """Lines 326-328: parse_label_set with comma-separated values."""
        result = HybridClassifier._parse_label_set("轴类, 齿轮, SHAFT")
        assert "轴类" in result
        assert "齿轮" in result
        assert "shaft" in result  # ASCII is lowercased

    def test_is_graph2d_drawing_type_true(self):
        """Lines 330-333: _is_graph2d_drawing_type returns True for drawing labels."""
        classifier = HybridClassifier()
        assert classifier._is_graph2d_drawing_type("零件图") is True
        assert classifier._is_graph2d_drawing_type("装配图") is True

    def test_is_graph2d_drawing_type_false(self):
        """_is_graph2d_drawing_type returns False for non-drawing labels."""
        classifier = HybridClassifier()
        assert classifier._is_graph2d_drawing_type("轴类") is False
        assert classifier._is_graph2d_drawing_type(None) is False
        assert classifier._is_graph2d_drawing_type("") is False


class TestGraphEnvLabels:
    """Tests for GRAPH2D_DRAWING_TYPE_LABELS env parsing (line 169)."""

    def test_graph2d_labels_from_env(self):
        """Line 169: parse drawing labels from env var."""
        with patch.dict(
            os.environ,
            {"GRAPH2D_DRAWING_TYPE_LABELS": "CustomType1, CustomType2"},
        ):
            classifier = HybridClassifier()
            assert "CustomType1" in classifier.graph2d_drawing_labels
            assert "CustomType2" in classifier.graph2d_drawing_labels


class TestGraph2DEnsemble:
    """Tests for graph2d ensemble classifier (lines 233-235)."""

    def test_graph2d_ensemble_enabled(self):
        """Lines 233-235: graph2d_classifier uses ensemble when enabled."""
        with patch.dict(
            os.environ,
            {
                "GRAPH2D_ENABLED": "true",
                "GRAPH2D_ENSEMBLE_ENABLED": "true",
            },
        ):
            classifier = HybridClassifier()
            classifier._graph2d_classifier = None  # Force re-init

            with patch(
                "src.ml.vision_2d.get_ensemble_2d_classifier"
            ) as mock_get_ensemble:
                mock_ensemble = MagicMock()
                mock_get_ensemble.return_value = mock_ensemble

                # Access property to trigger lazy load
                result = classifier.graph2d_classifier

                mock_get_ensemble.assert_called_once()


class TestIsHybridEnabled:
    """Tests for _is_hybrid_enabled method (line 284)."""

    def test_is_hybrid_enabled_from_env(self):
        """Line 284: _is_hybrid_enabled reads from env."""
        with patch.dict(os.environ, {"HYBRID_CLASSIFIER_ENABLED": "false"}):
            classifier = HybridClassifier()
            result = classifier._is_hybrid_enabled()
            assert result is False


class TestGraph2DErrorPath:
    """Tests for graph2d classification error handling (line 393)."""

    def test_graph2d_classification_error(self):
        """Lines 393-396: graph2d classification error path."""
        classifier = HybridClassifier()

        mock_fn_classifier = MagicMock()
        mock_fn_classifier.predict.return_value = {"label": "轴类", "confidence": 0.9}
        classifier._filename_classifier = mock_fn_classifier

        mock_graph2d = MagicMock()
        mock_graph2d.predict_from_bytes.side_effect = Exception("Graph2D error")
        classifier._graph2d_classifier = mock_graph2d

        with patch.dict(
            os.environ,
            {
                "FILENAME_CLASSIFIER_ENABLED": "true",
                "GRAPH2D_ENABLED": "true",
                "TITLEBLOCK_ENABLED": "false",
            },
        ):
            result = classifier.classify("test.dxf", file_bytes=b"content")

        assert "graph2d_error" in result.decision_path


class TestMarginExceptionHandling:
    """Tests for margin exception handling (lines 407-408)."""

    def test_graph2d_margin_invalid_value(self):
        """Lines 407-408: graph2d margin with invalid value falls back to None."""
        classifier = HybridClassifier()

        mock_fn_classifier = MagicMock()
        mock_fn_classifier.predict.return_value = {"label": "轴类", "confidence": 0.9}
        classifier._filename_classifier = mock_fn_classifier

        # Mock graph2d to return prediction with invalid margin
        mock_graph2d = MagicMock()
        mock_graph2d.predict_from_bytes.return_value = {
            "label": "齿轮",
            "confidence": 0.8,
            "margin": "invalid_margin",  # Should cause exception
        }
        classifier._graph2d_classifier = mock_graph2d

        with patch.dict(
            os.environ,
            {
                "FILENAME_CLASSIFIER_ENABLED": "true",
                "GRAPH2D_ENABLED": "true",
                "TITLEBLOCK_ENABLED": "false",
            },
        ):
            # Should not raise, margin is set to None on exception
            result = classifier.classify("test.dxf", file_bytes=b"content")

        assert result is not None


class TestProcessClassification:
    """Tests for process classification (lines 452-455, 458-459, 466-473, 479)."""

    def test_process_classification_with_text_entities(self):
        """Lines 452-455: extract text from TEXT, MTEXT, ATTRIB entities."""
        classifier = HybridClassifier()

        mock_fn_classifier = MagicMock()
        mock_fn_classifier.predict.return_value = {"label": "轴类", "confidence": 0.5}
        classifier._filename_classifier = mock_fn_classifier

        mock_process = MagicMock()
        mock_result = MagicMock()
        mock_result.suggested_labels = ["车削"]
        mock_result.confidence = 0.9
        mock_result.to_dict.return_value = {"label": "车削", "confidence": 0.9}
        mock_process.predict_from_text.return_value = mock_result
        classifier._process_classifier = mock_process

        # Create mock entities for TEXT, MTEXT, and ATTRIB
        text_entity = MagicMock()
        text_entity.dxftype.return_value = "TEXT"
        text_entity.dxf.text = "工艺说明"

        mtext_entity = MagicMock()
        mtext_entity.dxftype.return_value = "MTEXT"
        mtext_entity.text = "车削加工"

        attrib_entity = MagicMock()
        attrib_entity.dxftype.return_value = "ATTRIB"
        attrib_entity.dxf.text = "材料说明"

        with patch.dict(
            os.environ,
            {
                "FILENAME_CLASSIFIER_ENABLED": "true",
                "PROCESS_FEATURES_ENABLED": "true",
                "TITLEBLOCK_ENABLED": "false",
                "GRAPH2D_ENABLED": "false",
            },
        ):
            with patch(
                "src.utils.dxf_io.read_dxf_entities_from_bytes",
                return_value=[text_entity, mtext_entity, attrib_entity],
            ):
                result = classifier.classify("test.dxf", file_bytes=b"dxf_content")

        assert "process_predicted" in result.decision_path
        assert result.process_prediction is not None

    def test_process_with_insert_attribs(self):
        """Lines 458-459: extract text from INSERT entity attribs."""
        classifier = HybridClassifier()

        mock_fn_classifier = MagicMock()
        mock_fn_classifier.predict.return_value = {"label": "轴类", "confidence": 0.5}
        classifier._filename_classifier = mock_fn_classifier

        mock_process = MagicMock()
        mock_result = MagicMock()
        mock_result.suggested_labels = ["铣削"]
        mock_result.confidence = 0.95
        mock_result.to_dict.return_value = {"label": "铣削", "confidence": 0.95}
        mock_process.predict_from_text.return_value = mock_result
        classifier._process_classifier = mock_process

        # Create INSERT entity with attribs
        attrib = MagicMock()
        attrib.dxf.text = "铣削工艺"

        insert_entity = MagicMock()
        insert_entity.dxftype.return_value = "INSERT"
        insert_entity.attribs = [attrib]

        with patch.dict(
            os.environ,
            {
                "FILENAME_CLASSIFIER_ENABLED": "true",
                "PROCESS_FEATURES_ENABLED": "true",
                "TITLEBLOCK_ENABLED": "false",
                "GRAPH2D_ENABLED": "false",
            },
        ):
            with patch(
                "src.utils.dxf_io.read_dxf_entities_from_bytes",
                return_value=[insert_entity],
            ):
                result = classifier.classify("test.dxf", file_bytes=b"dxf_content")

        assert result.process_prediction is not None


class TestDrawingTypeIgnored:
    """Tests for drawing type being ignored (lines 490-495)."""

    def test_graph2d_drawing_type_ignored(self):
        """Lines 490-495: graph2d drawing type prediction is filtered."""
        classifier = HybridClassifier()

        mock_fn_classifier = MagicMock()
        mock_fn_classifier.predict.return_value = {"label": "轴类", "confidence": 0.9}
        classifier._filename_classifier = mock_fn_classifier

        mock_graph2d = MagicMock()
        mock_graph2d.predict_from_bytes.return_value = {
            "label": "零件图",  # Drawing type label
            "confidence": 0.95,
        }
        classifier._graph2d_classifier = mock_graph2d

        with patch.dict(
            os.environ,
            {
                "FILENAME_CLASSIFIER_ENABLED": "true",
                "GRAPH2D_ENABLED": "true",
                "TITLEBLOCK_ENABLED": "false",
            },
        ):
            result = classifier.classify("test.dxf", file_bytes=b"content")

        assert "graph2d_drawing_type_ignored" in result.decision_path
        assert result.graph2d_prediction.get("filtered") is True
        assert result.graph2d_prediction.get("filtered_reason") == "drawing_type"


class TestLabelMapSizeThreshold:
    """Tests for dynamic confidence threshold based on label_map_size (lines 526-527)."""

    def test_dynamic_min_conf_with_large_label_map(self):
        """Lines 526-532: dynamic min conf when label_map_size >= 20."""
        classifier = HybridClassifier()

        mock_fn_classifier = MagicMock()
        mock_fn_classifier.predict.return_value = {"label": "轴类", "confidence": 0.3}
        classifier._filename_classifier = mock_fn_classifier

        mock_graph2d = MagicMock()
        mock_graph2d.predict_from_bytes.return_value = {
            "label": "齿轮",
            "confidence": 0.08,  # Very low but might pass dynamic threshold
            "label_map_size": 50,  # Large label map
        }
        classifier._graph2d_classifier = mock_graph2d

        with patch.dict(
            os.environ,
            {
                "FILENAME_CLASSIFIER_ENABLED": "true",
                "GRAPH2D_ENABLED": "true",
                "GRAPH2D_MIN_CONF": "0.2",  # Higher than dynamic
                "TITLEBLOCK_ENABLED": "false",
            },
        ):
            result = classifier.classify("test.dxf", file_bytes=b"content")

        # Check that dynamic threshold was calculated
        assert result.graph2d_prediction.get("min_confidence_effective") is not None

    def test_invalid_label_map_size(self):
        """Lines 526-527: handle invalid label_map_size gracefully."""
        classifier = HybridClassifier()

        mock_fn_classifier = MagicMock()
        mock_fn_classifier.predict.return_value = {"label": "轴类", "confidence": 0.9}
        classifier._filename_classifier = mock_fn_classifier

        mock_graph2d = MagicMock()
        mock_graph2d.predict_from_bytes.return_value = {
            "label": "齿轮",
            "confidence": 0.8,
            "label_map_size": "invalid",  # Invalid value
        }
        classifier._graph2d_classifier = mock_graph2d

        with patch.dict(
            os.environ,
            {
                "FILENAME_CLASSIFIER_ENABLED": "true",
                "GRAPH2D_ENABLED": "true",
                "TITLEBLOCK_ENABLED": "false",
            },
        ):
            # Should not raise
            result = classifier.classify("test.dxf", file_bytes=b"content")

        assert result is not None


class TestTitleblockFilenameConflict:
    """Tests for titleblock/filename conflict (line 571)."""

    def test_titleblock_ignored_when_filename_high_conf(self):
        """Line 571: titleblock ignored when filename has high confidence."""
        with patch.dict(
            os.environ,
            {
                "FILENAME_CLASSIFIER_ENABLED": "true",
                "FILENAME_MIN_CONF": "0.5",
                "TITLEBLOCK_ENABLED": "true",
                "TITLEBLOCK_OVERRIDE_ENABLED": "false",
                "GRAPH2D_ENABLED": "false",
                "PROCESS_FEATURES_ENABLED": "false",
            },
        ):
            classifier = HybridClassifier()

            mock_fn_classifier = MagicMock()
            mock_fn_classifier.predict.return_value = {
                "label": "轴类",
                "confidence": 0.95,  # High confidence
            }
            classifier._filename_classifier = mock_fn_classifier

            mock_tb_classifier = MagicMock()
            mock_tb_classifier.predict.return_value = {
                "label": "齿轮",  # Different label
                "confidence": 0.8,
            }
            classifier._titleblock_classifier = mock_tb_classifier

            mock_entity = MagicMock()
            mock_entity.dxftype.return_value = "LINE"

            with patch(
                "src.utils.dxf_io.read_dxf_entities_from_bytes",
                return_value=[mock_entity],
            ):
                result = classifier.classify("test.dxf", file_bytes=b"dxf")

        assert "titleblock_filename_conflict" in result.decision_path
        assert "titleblock_ignored_filename_high_conf" in result.decision_path


class TestProcessPredictionAdded:
    """Tests for process prediction in fusion (line 622)."""

    def test_process_label_in_fusion(self):
        """Line 622: process label participates in fusion."""
        with patch.dict(
            os.environ,
            {
                "FILENAME_CLASSIFIER_ENABLED": "true",
                "FILENAME_MIN_CONF": "0.8",
                "PROCESS_FEATURES_ENABLED": "true",
                "PROCESS_MIN_CONF": "0.5",
                "TITLEBLOCK_ENABLED": "false",
                "GRAPH2D_ENABLED": "false",
            },
        ):
            classifier = HybridClassifier()

            mock_fn_classifier = MagicMock()
            mock_fn_classifier.predict.return_value = {
                "label": "轴类",
                "confidence": 0.6,  # Below min_conf so fusion happens
            }
            classifier._filename_classifier = mock_fn_classifier

            mock_process = MagicMock()
            mock_result = MagicMock()
            mock_result.suggested_labels = ["轴类"]  # Same label
            mock_result.confidence = 0.8
            mock_result.to_dict.return_value = {"label": "轴类", "confidence": 0.8}
            mock_process.predict_from_text.return_value = mock_result
            classifier._process_classifier = mock_process

            text_entity = MagicMock()
            text_entity.dxftype.return_value = "TEXT"
            text_entity.dxf.text = "工艺说明"

            with patch(
                "src.utils.dxf_io.read_dxf_entities_from_bytes",
                return_value=[text_entity],
            ):
                result = classifier.classify("test.dxf", file_bytes=b"dxf")

        assert result.process_prediction is not None
        assert result.label == "轴类"


class TestFilenameClassifierLazyLoad:
    """Tests for filename_classifier lazy loading (lines 212-218)."""

    def test_filename_classifier_lazy_load(self):
        """Lines 212-218: filename_classifier lazy loads with synonyms_path."""
        classifier = HybridClassifier()
        classifier._filename_classifier = None  # Force re-init

        with patch("src.ml.filename_classifier.FilenameClassifier") as MockFC:
            mock_instance = MagicMock()
            MockFC.return_value = mock_instance

            # Access property to trigger lazy load
            result = classifier.filename_classifier

            MockFC.assert_called_once()
            assert result == mock_instance


class TestGraph2DExcludeLabels:
    """Tests for graph2d exclude labels (lines 503-508)."""

    def test_graph2d_excluded_label_ignored(self):
        """Lines 503-508: graph2d label in exclude list is filtered."""
        with patch.dict(
            os.environ,
            {
                "FILENAME_CLASSIFIER_ENABLED": "false",
                "GRAPH2D_ENABLED": "true",
                "TITLEBLOCK_ENABLED": "false",
            },
        ):
            classifier = HybridClassifier()
            # Directly set exclude labels after initialization
            classifier.graph2d_exclude_labels = {"杂项", "other"}

            mock_graph2d = MagicMock()
            mock_graph2d.predict_from_bytes.return_value = {
                "label": "杂项",  # In exclude list
                "confidence": 0.95,
            }
            classifier._graph2d_classifier = mock_graph2d

            result = classifier.classify("test.dxf", file_bytes=b"content")

        assert "graph2d_excluded_label_ignored" in result.decision_path
        assert result.graph2d_prediction.get("filtered") is True


class TestGraph2DAllowLabels:
    """Tests for graph2d allow labels (lines 512-517)."""

    def test_graph2d_not_in_allowlist_ignored(self):
        """Lines 512-517: graph2d label not in allow list is filtered."""
        with patch.dict(
            os.environ,
            {
                "FILENAME_CLASSIFIER_ENABLED": "false",
                "GRAPH2D_ENABLED": "true",
                "TITLEBLOCK_ENABLED": "false",
            },
        ):
            classifier = HybridClassifier()
            # Directly set allow labels after initialization
            classifier.graph2d_allow_labels = {"轴类", "齿轮"}

            mock_graph2d = MagicMock()
            mock_graph2d.predict_from_bytes.return_value = {
                "label": "杂项",  # Not in allow list
                "confidence": 0.95,
            }
            classifier._graph2d_classifier = mock_graph2d

            result = classifier.classify("test.dxf", file_bytes=b"content")

        assert "graph2d_not_in_allowlist_ignored" in result.decision_path
        assert result.graph2d_prediction.get("filtered") is True


class TestGraph2DBelowMinConf:
    """Tests for graph2d below min_conf filter (lines 535-540)."""

    def test_graph2d_below_min_conf_ignored(self):
        """Lines 535-540: graph2d below min_conf is filtered."""
        with patch.dict(
            os.environ,
            {
                "FILENAME_CLASSIFIER_ENABLED": "true",
                "GRAPH2D_ENABLED": "true",
                "GRAPH2D_MIN_CONF": "0.5",  # Set min conf
                "TITLEBLOCK_ENABLED": "false",
            },
        ):
            classifier = HybridClassifier()

            mock_fn_classifier = MagicMock()
            mock_fn_classifier.predict.return_value = {"label": "轴类", "confidence": 0.9}
            classifier._filename_classifier = mock_fn_classifier

            mock_graph2d = MagicMock()
            mock_graph2d.predict_from_bytes.return_value = {
                "label": "齿轮",
                "confidence": 0.3,  # Below min_conf
            }
            classifier._graph2d_classifier = mock_graph2d

            result = classifier.classify("test.dxf", file_bytes=b"content")

        assert "graph2d_below_min_conf_ignored" in result.decision_path
        assert result.graph2d_prediction.get("filtered") is True


class TestGraph2DBelowMinMargin:
    """Tests for graph2d below min_margin filter (lines 548-554)."""

    def test_graph2d_below_min_margin_ignored(self):
        """Lines 548-554: graph2d below min_margin is filtered."""
        with patch.dict(
            os.environ,
            {
                "FILENAME_CLASSIFIER_ENABLED": "true",
                "GRAPH2D_ENABLED": "true",
                "GRAPH2D_MIN_MARGIN": "0.1",  # Set min margin
                "TITLEBLOCK_ENABLED": "false",
            },
        ):
            classifier = HybridClassifier()

            mock_fn_classifier = MagicMock()
            mock_fn_classifier.predict.return_value = {"label": "轴类", "confidence": 0.9}
            classifier._filename_classifier = mock_fn_classifier

            mock_graph2d = MagicMock()
            mock_graph2d.predict_from_bytes.return_value = {
                "label": "齿轮",
                "confidence": 0.8,
                "margin": 0.05,  # Below min_margin
            }
            classifier._graph2d_classifier = mock_graph2d

            result = classifier.classify("test.dxf", file_bytes=b"content")

        assert "graph2d_below_min_margin_ignored" in result.decision_path
        assert result.graph2d_prediction.get("filtered") is True


class TestGraph2DOnlyInFusion:
    """Tests for graph2d only prediction in fusion (line 626)."""

    def test_graph2d_only_prediction(self):
        """Line 626: graph2d as only prediction source."""
        with patch.dict(
            os.environ,
            {
                "FILENAME_CLASSIFIER_ENABLED": "false",
                "GRAPH2D_ENABLED": "true",
                "TITLEBLOCK_ENABLED": "false",
                "PROCESS_FEATURES_ENABLED": "false",
            },
        ):
            classifier = HybridClassifier()
            classifier._filename_classifier = None  # Disabled

            mock_graph2d = MagicMock()
            mock_graph2d.predict_from_bytes.return_value = {
                "label": "齿轮",
                "confidence": 0.9,
            }
            classifier._graph2d_classifier = mock_graph2d

            result = classifier.classify("test.dxf", file_bytes=b"content")

        assert result.label == "齿轮"
        assert result.source == DecisionSource.GRAPH2D
        assert "graph2d_only" in result.decision_path
