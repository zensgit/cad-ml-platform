"""Unit tests for the IntentClassifier."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.core.assistant.intent_classifier import IntentClassifier


@pytest.fixture()
def classifier() -> IntentClassifier:
    return IntentClassifier()


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------


class TestTraining:
    def test_classifier_trains(self, classifier: IntentClassifier) -> None:
        assert classifier.is_trained is True
        assert classifier._trained is True

    def test_training_data_has_at_least_60_examples(self) -> None:
        assert len(IntentClassifier.TRAINING_DATA) >= 60


# ------------------------------------------------------------------
# Intent classification
# ------------------------------------------------------------------


class TestClassification:
    def test_classify_material_query(self, classifier: IntentClassifier) -> None:
        result = classifier.classify("SUS304密度")
        assert result["intent"] == "material_property"

    def test_classify_process_query(self, classifier: IntentClassifier) -> None:
        result = classifier.classify("法兰盘加工工艺")
        assert result["intent"] == "process_route"

    def test_classify_cost_query(self, classifier: IntentClassifier) -> None:
        result = classifier.classify("这个零件加工费用估算多少钱")
        assert result["intent"] == "cost_estimation"

    def test_classify_gdt_query(self, classifier: IntentClassifier) -> None:
        result = classifier.classify("什么是平面度公差含义")
        assert result["intent"] in ("gdt_interpretation", "gdt_application")

    def test_classify_welding_query(self, classifier: IntentClassifier) -> None:
        result = classifier.classify("不锈钢TIG焊接参数要求")
        assert result["intent"] == "welding_parameters"

    def test_classify_tolerance_query(self, classifier: IntentClassifier) -> None:
        result = classifier.classify("IT7公差等级的数值是多少")
        assert result["intent"] == "tolerance_lookup"


# ------------------------------------------------------------------
# Output format
# ------------------------------------------------------------------


class TestOutputFormat:
    def test_confidence_between_0_and_1(
        self, classifier: IntentClassifier
    ) -> None:
        result = classifier.classify("铝合金6061的硬度")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_top_3_returns_3(self, classifier: IntentClassifier) -> None:
        result = classifier.classify("铝合金6061的硬度")
        assert len(result["top_3"]) == 3

    def test_top_3_each_entry_is_tuple(
        self, classifier: IntentClassifier
    ) -> None:
        result = classifier.classify("铝合金6061的硬度")
        for entry in result["top_3"]:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            assert isinstance(entry[0], str)
            assert isinstance(entry[1], float)

    def test_method_is_trained_classifier(
        self, classifier: IntentClassifier
    ) -> None:
        result = classifier.classify("铝合金6061的硬度")
        assert result["method"] == "trained_classifier"


# ------------------------------------------------------------------
# Supported intents
# ------------------------------------------------------------------


class TestSupportedIntents:
    def test_supported_intents_not_empty(
        self, classifier: IntentClassifier
    ) -> None:
        intents = classifier.get_supported_intents()
        assert len(intents) >= 5

    def test_material_property_in_intents(
        self, classifier: IntentClassifier
    ) -> None:
        intents = classifier.get_supported_intents()
        assert "material_property" in intents

    def test_process_route_in_intents(
        self, classifier: IntentClassifier
    ) -> None:
        intents = classifier.get_supported_intents()
        assert "process_route" in intents


# ------------------------------------------------------------------
# Fallback without sklearn
# ------------------------------------------------------------------


class TestFallback:
    def test_fallback_without_sklearn(self) -> None:
        """When sklearn import fails, classifier should fall back gracefully."""
        with patch.dict("sys.modules", {
            "sklearn": None,
            "sklearn.feature_extraction.text": None,
            "sklearn.linear_model": None,
            "sklearn.preprocessing": None,
        }):
            clf = IntentClassifier()
            assert clf._trained is False
            result = clf.classify("SUS304密度")
            assert result["method"] == "fallback"
            assert result["intent"] == "material_property"

    def test_fallback_general_question(self) -> None:
        """Unrecognised query falls back to general_question."""
        with patch.dict("sys.modules", {
            "sklearn": None,
            "sklearn.feature_extraction.text": None,
            "sklearn.linear_model": None,
            "sklearn.preprocessing": None,
        }):
            clf = IntentClassifier()
            result = clf.classify("你好呀")
            assert result["intent"] == "general_question"
            assert result["method"] == "fallback"
