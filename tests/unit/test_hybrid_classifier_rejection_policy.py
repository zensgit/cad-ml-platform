from __future__ import annotations

from typing import Any, Dict, Optional

from src.ml.hybrid_classifier import DecisionSource, HybridClassifier


class _DummyFilenameClassifier:
    def __init__(self, label: Optional[str], confidence: float) -> None:
        self._label = label
        self._confidence = confidence

    def predict(self, filename: str) -> Dict[str, Any]:
        _ = filename
        return {"label": self._label, "confidence": self._confidence}


def test_hybrid_rejects_low_confidence_final_result(monkeypatch) -> None:
    monkeypatch.setenv("HYBRID_CLASSIFIER_ENABLED", "true")
    monkeypatch.setenv("FILENAME_CLASSIFIER_ENABLED", "true")
    monkeypatch.setenv("GRAPH2D_ENABLED", "false")
    monkeypatch.setenv("TITLEBLOCK_ENABLED", "false")
    monkeypatch.setenv("PROCESS_FEATURES_ENABLED", "false")
    monkeypatch.setenv("HISTORY_SEQUENCE_ENABLED", "false")
    monkeypatch.setenv("HYBRID_REJECT_ENABLED", "true")
    monkeypatch.setenv("HYBRID_REJECT_MIN_CONFIDENCE", "0.95")

    clf = HybridClassifier()
    clf._filename_classifier = _DummyFilenameClassifier(label="轴类", confidence=0.90)

    result = clf.classify(filename="demo.dxf", file_bytes=None)
    assert result.label is None
    assert result.source == DecisionSource.FALLBACK
    assert result.confidence == 0.0
    assert "final_below_reject_min_conf" in (result.decision_path or [])
    assert isinstance(result.rejection, dict)
    assert result.rejection.get("raw_label") == "轴类"
    assert float(result.rejection.get("raw_confidence") or 0.0) == 0.9


def test_hybrid_keeps_label_when_confidence_above_reject_threshold(monkeypatch) -> None:
    monkeypatch.setenv("HYBRID_CLASSIFIER_ENABLED", "true")
    monkeypatch.setenv("FILENAME_CLASSIFIER_ENABLED", "true")
    monkeypatch.setenv("GRAPH2D_ENABLED", "false")
    monkeypatch.setenv("TITLEBLOCK_ENABLED", "false")
    monkeypatch.setenv("PROCESS_FEATURES_ENABLED", "false")
    monkeypatch.setenv("HISTORY_SEQUENCE_ENABLED", "false")
    monkeypatch.setenv("HYBRID_REJECT_ENABLED", "true")
    monkeypatch.setenv("HYBRID_REJECT_MIN_CONFIDENCE", "0.80")

    clf = HybridClassifier()
    clf._filename_classifier = _DummyFilenameClassifier(label="轴类", confidence=0.90)

    result = clf.classify(filename="demo.dxf", file_bytes=None)
    assert result.label == "轴类"
    assert result.source == DecisionSource.FILENAME
    assert result.rejection is None
    assert "final_below_reject_min_conf" not in (result.decision_path or [])
