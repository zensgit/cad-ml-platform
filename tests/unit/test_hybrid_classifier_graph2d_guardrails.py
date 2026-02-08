from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from src.ml.hybrid_classifier import DecisionSource, HybridClassifier
from src.ml.hybrid_config import reset_config


class _DummyFilenameClassifier:
    def __init__(self, label: Optional[str], confidence: float):
        self._label = label
        self._confidence = confidence

    def predict(self, filename: str) -> Dict[str, Any]:
        return {
            "label": self._label,
            "confidence": self._confidence,
            "match_type": "exact",
            "extracted_name": self._label,
            "status": "matched" if self._label else "no_match",
        }


class _DummyTitleBlockClassifier:
    def __init__(self, label: str, confidence: float):
        self._label = label
        self._confidence = confidence

    def predict(self, dxf_entities: List[Any]) -> Dict[str, Any]:
        return {
            "label": self._label,
            "confidence": self._confidence,
            "status": "matched",
        }


def _new_classifier(monkeypatch: pytest.MonkeyPatch) -> HybridClassifier:
    reset_config()
    monkeypatch.setenv("HYBRID_CLASSIFIER_ENABLED", "true")
    monkeypatch.setenv("PROCESS_FEATURES_ENABLED", "false")
    return HybridClassifier()


def test_graph2d_excluded_label_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPH2D_FUSION_EXCLUDE_LABELS", "传动件")
    monkeypatch.delenv("GRAPH2D_FUSION_ALLOW_LABELS", raising=False)

    clf = _new_classifier(monkeypatch)
    clf._filename_classifier = _DummyFilenameClassifier(label=None, confidence=0.0)

    result = clf.classify(
        filename="J0000000-00UNKNOWNv1.dxf",
        graph2d_result={"label": "传动件", "confidence": 0.99},
    )

    assert result.label is None
    assert result.source == DecisionSource.FALLBACK
    assert "graph2d_excluded_label_ignored" in (result.decision_path or [])


def test_graph2d_allowlist_filters_non_allowed_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GRAPH2D_FUSION_EXCLUDE_LABELS", raising=False)
    monkeypatch.setenv("GRAPH2D_FUSION_ALLOW_LABELS", "人孔")

    clf = _new_classifier(monkeypatch)
    clf._filename_classifier = _DummyFilenameClassifier(label=None, confidence=0.0)

    result = clf.classify(
        filename="J0000000-00UNKNOWNv1.dxf",
        graph2d_result={"label": "出料凸缘", "confidence": 0.99},
    )

    assert result.label is None
    assert result.source == DecisionSource.FALLBACK
    assert "graph2d_not_in_allowlist_ignored" in (result.decision_path or [])


def test_titleblock_override_beats_graph2d(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GRAPH2D_FUSION_EXCLUDE_LABELS", raising=False)
    monkeypatch.delenv("GRAPH2D_FUSION_ALLOW_LABELS", raising=False)
    monkeypatch.setenv("TITLEBLOCK_ENABLED", "true")
    monkeypatch.setenv("TITLEBLOCK_OVERRIDE_ENABLED", "true")
    monkeypatch.setenv("TITLEBLOCK_MIN_CONF", "0.75")

    # Avoid real DXF parsing; we only need a non-None entity list.
    from src.utils import dxf_io

    def _fake_read_dxf_entities_from_bytes(_: bytes):  # type: ignore[no-untyped-def]
        return [object()]

    monkeypatch.setattr(
        dxf_io,
        "read_dxf_entities_from_bytes",
        _fake_read_dxf_entities_from_bytes,
    )

    clf = _new_classifier(monkeypatch)
    clf._filename_classifier = _DummyFilenameClassifier(label=None, confidence=0.0)
    clf._titleblock_classifier = _DummyTitleBlockClassifier(label="人孔", confidence=0.9)

    result = clf.classify(
        filename="J0000000-00UNKNOWNv1.dxf",
        file_bytes=b"not-a-real-dxf",
        graph2d_result={"label": "人孔", "confidence": 0.99},
    )

    assert result.label == "人孔"
    assert result.source == DecisionSource.TITLEBLOCK
    assert "titleblock_adopted" in (result.decision_path or [])
