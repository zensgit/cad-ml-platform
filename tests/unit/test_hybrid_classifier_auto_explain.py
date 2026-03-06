from __future__ import annotations

from types import SimpleNamespace

from src.ml.hybrid_classifier import DecisionSource, HybridClassifier
from src.ml.hybrid_config import reset_config


class _TextEntity:
    def __init__(self, dtype: str = "TEXT") -> None:
        self._dtype = dtype
        self.dxf = SimpleNamespace(
            text="名称: 人孔",
            insert=SimpleNamespace(x=80.0, y=10.0),
        )

    def dxftype(self) -> str:
        return self._dtype


def test_hybrid_classifier_auto_enables_titleblock_and_generates_explanation(
    monkeypatch,
) -> None:
    class _TitleBlockStub:
        def predict(self, entities):  # noqa: ANN001, ANN201
            assert entities
            return {"label": "人孔", "confidence": 0.83, "source": "titleblock"}

    monkeypatch.setenv("FILENAME_CLASSIFIER_ENABLED", "false")
    monkeypatch.setenv("GRAPH2D_ENABLED", "false")
    monkeypatch.setenv("TITLEBLOCK_ENABLED", "false")
    monkeypatch.setenv("TITLEBLOCK_AUTO_ENABLE", "true")
    monkeypatch.setenv("PROCESS_FEATURES_ENABLED", "false")
    monkeypatch.setenv("HISTORY_SEQUENCE_ENABLED", "false")
    monkeypatch.setenv("HYBRID_EXPLANATION_ENABLED", "true")
    monkeypatch.setattr(
        "src.utils.dxf_io.read_dxf_entities_from_bytes",
        lambda file_bytes: [_TextEntity()],
    )

    reset_config()
    classifier = HybridClassifier()
    classifier._titleblock_classifier = _TitleBlockStub()

    result = classifier.classify(filename="sample.dxf", file_bytes=b"DXF")

    assert result.source == DecisionSource.TITLEBLOCK
    assert result.label == "人孔"
    assert "titleblock_auto_enabled" in (result.decision_path or [])
    assert "titleblock_predicted" in (result.decision_path or [])
    assert result.fusion_metadata is not None
    assert result.fusion_metadata.get("selected_by") == "titleblock"
    assert result.explanation is not None
    reset_config()


def test_hybrid_classifier_auto_enables_history_when_sidecar_exists(
    monkeypatch,
) -> None:
    class _HistoryStub:
        def predict_from_h5_file(self, file_path):  # noqa: ANN001, ANN201
            return {
                "label": "阀体",
                "confidence": 0.88,
                "status": "ok",
                "file_path": file_path,
            }

    monkeypatch.setenv("FILENAME_CLASSIFIER_ENABLED", "false")
    monkeypatch.setenv("GRAPH2D_ENABLED", "false")
    monkeypatch.setenv("TITLEBLOCK_ENABLED", "false")
    monkeypatch.setenv("PROCESS_FEATURES_ENABLED", "false")
    monkeypatch.setenv("HISTORY_SEQUENCE_ENABLED", "false")
    monkeypatch.setenv("HISTORY_SEQUENCE_AUTO_ENABLE", "true")
    monkeypatch.setenv("HYBRID_EXPLANATION_ENABLED", "true")

    reset_config()
    classifier = HybridClassifier()
    classifier._history_sequence_classifier = _HistoryStub()

    result = classifier.classify(
        filename="sample.dxf",
        history_file_path="/tmp/sample.h5",
    )

    assert result.source == DecisionSource.HISTORY
    assert result.label == "阀体"
    assert "history_auto_enabled" in (result.decision_path or [])
    assert "history_predicted" in (result.decision_path or [])
    assert result.history_prediction is not None
    assert result.explanation is not None
    reset_config()


def test_hybrid_classifier_advanced_fusion_emits_metadata_and_contributions(
    monkeypatch,
) -> None:
    class _FilenameStub:
        def predict(self, filename):  # noqa: ANN001, ANN201
            return {"label": "Bolt", "confidence": 0.71, "source": "filename"}

    monkeypatch.setenv("FILENAME_CLASSIFIER_ENABLED", "true")
    monkeypatch.setenv("GRAPH2D_ENABLED", "false")
    monkeypatch.setenv("TITLEBLOCK_ENABLED", "false")
    monkeypatch.setenv("PROCESS_FEATURES_ENABLED", "false")
    monkeypatch.setenv("HISTORY_SEQUENCE_ENABLED", "false")
    monkeypatch.setenv("HYBRID_ADVANCED_FUSION_ENABLED", "true")
    monkeypatch.setenv("HYBRID_FUSION_STRATEGY", "weighted_average")
    monkeypatch.setenv("HYBRID_EXPLANATION_ENABLED", "true")

    reset_config()
    classifier = HybridClassifier(filename_min_conf=0.95)
    classifier._filename_classifier = _FilenameStub()

    result = classifier.classify(
        filename="Bolt_M6x20.dxf",
        graph2d_result={"label": "bolt", "confidence": 0.82, "status": "ok"},
    )

    assert result.source == DecisionSource.FUSION
    assert result.label == "Bolt"
    assert "fusion_engine_weighted_average" in (result.decision_path or [])
    assert result.fusion_metadata is not None
    assert result.fusion_metadata.get("strategy") == "weighted_average"
    assert (result.source_contributions or {}).get("filename", 0.0) > 0.0
    assert (result.source_contributions or {}).get("graph2d", 0.0) > 0.0
    assert result.explanation is not None
    reset_config()
