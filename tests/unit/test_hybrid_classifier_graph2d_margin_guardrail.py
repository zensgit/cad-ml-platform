from __future__ import annotations

import pytest


def test_graph2d_margin_below_min_is_ignored(monkeypatch) -> None:
    # Isolate the Graph2D path: disable filename/titleblock/process signals.
    monkeypatch.setenv("FILENAME_CLASSIFIER_ENABLED", "false")
    monkeypatch.setenv("TITLEBLOCK_ENABLED", "false")
    monkeypatch.setenv("PROCESS_FEATURES_ENABLED", "false")
    # Ensure confidence isn't the reason the prediction is filtered.
    monkeypatch.setenv("GRAPH2D_MIN_CONF", "0.0")
    monkeypatch.setenv("GRAPH2D_MIN_MARGIN", "0.2")

    from src.ml.hybrid_classifier import HybridClassifier

    clf = HybridClassifier()
    result = clf.classify(
        filename="masked.dxf",
        graph2d_result={
            "label": "人孔",
            "confidence": 0.9,
            "margin": 0.01,
            "top2_confidence": 0.89,
            "label_map_size": 47,
            "status": "ok",
        },
    )

    assert "graph2d_below_min_margin_ignored" in (result.decision_path or [])
    assert result.graph2d_prediction is not None
    assert result.graph2d_prediction.get("filtered") is True
    assert result.graph2d_prediction.get("filtered_reason") == "below_min_margin"
    assert result.label is None
    assert result.source is not None


def test_graph2d_margin_above_min_is_accepted(monkeypatch) -> None:
    monkeypatch.setenv("FILENAME_CLASSIFIER_ENABLED", "false")
    monkeypatch.setenv("TITLEBLOCK_ENABLED", "false")
    monkeypatch.setenv("PROCESS_FEATURES_ENABLED", "false")
    monkeypatch.setenv("GRAPH2D_MIN_CONF", "0.0")
    monkeypatch.setenv("GRAPH2D_MIN_MARGIN", "0.2")

    from src.ml.hybrid_classifier import HybridClassifier

    clf = HybridClassifier()
    result = clf.classify(
        filename="masked.dxf",
        graph2d_result={
            "label": "人孔",
            "confidence": 0.9,
            "margin": 0.5,
            "top2_confidence": 0.4,
            "label_map_size": 47,
            "status": "ok",
        },
    )

    assert result.label == "人孔"
    assert result.source is not None
    assert "graph2d_only" in (result.decision_path or [])

