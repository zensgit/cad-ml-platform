"""Unit coverage for ML classifier confidence extraction."""

from __future__ import annotations

import math

import pytest

import src.ml.classifier as classifier


class _ProbaModel:
    classes_ = ["A", "B"]

    def predict(self, X):
        return ["B" for _ in X]

    def predict_proba(self, X):
        return [[0.1, 0.9] for _ in X]


class _DecisionModel:
    classes_ = ["A", "B", "C"]

    def predict(self, X):
        return ["C" for _ in X]

    def decision_function(self, X):
        return [[0.1, 0.2, 2.0] for _ in X]


@pytest.mark.parametrize("vector", [[0.0, 1.0], [1.0, 2.0]])
def test_predict_confidence_from_proba(monkeypatch: pytest.MonkeyPatch, vector) -> None:
    monkeypatch.setattr(classifier, "_MODEL", _ProbaModel())
    monkeypatch.setattr(classifier, "_MODEL_VERSION", "test")
    monkeypatch.setattr(classifier, "_MODEL_HASH", "hash")
    monkeypatch.setattr(classifier, "load_model", lambda: None)

    result = classifier.predict(vector)

    assert result["predicted_type"] == "B"
    assert result["confidence_source"] == "predict_proba"
    assert result["confidence"] == pytest.approx(0.9)


def test_predict_confidence_from_decision_function(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(classifier, "_MODEL", _DecisionModel())
    monkeypatch.setattr(classifier, "_MODEL_VERSION", "test")
    monkeypatch.setattr(classifier, "_MODEL_HASH", "hash")
    monkeypatch.setattr(classifier, "load_model", lambda: None)

    result = classifier.predict([1.0, 0.5])

    assert result["predicted_type"] == "C"
    assert result["confidence_source"] == "decision_function"

    expected = math.exp(2.0) / (math.exp(0.1) + math.exp(0.2) + math.exp(2.0))
    assert result["confidence"] == pytest.approx(expected)
