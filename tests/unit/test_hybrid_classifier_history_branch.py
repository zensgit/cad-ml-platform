from __future__ import annotations

from pathlib import Path

import src.ml.history_sequence_classifier as history_mod
from src.ml.hybrid_config import reset_config
from src.ml.hybrid_classifier import DecisionSource, HybridClassifier


def test_hybrid_classifier_adopts_high_conf_history_signal(monkeypatch) -> None:
    monkeypatch.setenv("FILENAME_CLASSIFIER_ENABLED", "false")
    monkeypatch.setenv("GRAPH2D_ENABLED", "false")
    monkeypatch.setenv("TITLEBLOCK_ENABLED", "false")
    monkeypatch.setenv("PROCESS_FEATURES_ENABLED", "false")
    monkeypatch.setenv("HISTORY_SEQUENCE_ENABLED", "true")

    reset_config()
    classifier = HybridClassifier(history_min_conf=0.4, history_weight=0.5)
    result = classifier.classify(
        filename="sample.dxf",
        history_result={
            "status": "ok",
            "label": "轴类",
            "confidence": 0.91,
            "source": "history_sequence_prototype",
        },
    )

    assert result.label == "轴类"
    assert result.source == DecisionSource.HISTORY
    assert "history_high_conf_adopted" in result.decision_path
    assert result.history_prediction is not None
    reset_config()


def test_hybrid_classifier_filters_low_conf_history_signal(monkeypatch) -> None:
    monkeypatch.setenv("FILENAME_CLASSIFIER_ENABLED", "false")
    monkeypatch.setenv("GRAPH2D_ENABLED", "false")
    monkeypatch.setenv("TITLEBLOCK_ENABLED", "false")
    monkeypatch.setenv("PROCESS_FEATURES_ENABLED", "false")
    monkeypatch.setenv("HISTORY_SEQUENCE_ENABLED", "true")

    reset_config()
    classifier = HybridClassifier(history_min_conf=0.8, history_weight=0.5)
    result = classifier.classify(
        filename="sample.dxf",
        history_result={
            "status": "ok",
            "label": "轴类",
            "confidence": 0.31,
            "source": "history_sequence_prototype",
        },
    )

    assert result.label is None
    assert result.source == DecisionSource.FALLBACK
    assert "history_below_min_conf_ignored" in result.decision_path
    assert result.history_prediction is not None
    assert result.history_prediction.get("filtered") is True
    reset_config()


def test_hybrid_history_classifier_uses_configured_paths(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, str] = {}
    proto = tmp_path / "proto.json"
    proto.write_text("{}", encoding="utf-8")
    model = tmp_path / "model.pt"
    model.write_bytes(b"")

    class _StubHistoryClassifier:
        def __init__(
            self, prototypes_path=None, model_path=None, **kwargs
        ):  # noqa: ANN001
            captured["prototypes_path"] = str(prototypes_path or "")
            captured["model_path"] = str(model_path or "")
            captured["prototype_token_weight"] = str(
                kwargs.get("prototype_token_weight")
            )
            captured["prototype_bigram_weight"] = str(
                kwargs.get("prototype_bigram_weight")
            )

        def predict_from_h5_file(self, file_path):  # noqa: ANN001, ANN201
            return {"status": "ok", "label": "轴类", "confidence": 0.8}

    monkeypatch.setenv("FILENAME_CLASSIFIER_ENABLED", "false")
    monkeypatch.setenv("GRAPH2D_ENABLED", "false")
    monkeypatch.setenv("TITLEBLOCK_ENABLED", "false")
    monkeypatch.setenv("PROCESS_FEATURES_ENABLED", "false")
    monkeypatch.setenv("HISTORY_SEQUENCE_ENABLED", "true")
    monkeypatch.setenv("HISTORY_SEQUENCE_PROTOTYPES_PATH", str(proto))
    monkeypatch.setenv("HISTORY_SEQUENCE_MODEL_PATH", str(model))
    monkeypatch.setenv("HISTORY_SEQUENCE_PROTOTYPE_TOKEN_WEIGHT", "0.7")
    monkeypatch.setenv("HISTORY_SEQUENCE_PROTOTYPE_BIGRAM_WEIGHT", "1.4")
    monkeypatch.setattr(
        history_mod,
        "HistorySequenceClassifier",
        _StubHistoryClassifier,
    )

    reset_config()
    classifier = HybridClassifier()
    _ = classifier.history_sequence_classifier
    assert captured["prototypes_path"] == str(proto)
    assert captured["model_path"] == str(model)
    assert captured["prototype_token_weight"] == "0.7"
    assert captured["prototype_bigram_weight"] == "1.4"
    reset_config()


def test_hybrid_fusion_normalizes_ascii_labels_without_splitting(monkeypatch) -> None:
    class _FilenameStub:
        def predict(self, filename):  # noqa: ANN001, ANN201
            return {"label": "Bolt", "confidence": 0.7, "source": "filename"}

    monkeypatch.setenv("FILENAME_CLASSIFIER_ENABLED", "true")
    monkeypatch.setenv("GRAPH2D_ENABLED", "false")
    monkeypatch.setenv("TITLEBLOCK_ENABLED", "false")
    monkeypatch.setenv("PROCESS_FEATURES_ENABLED", "false")
    monkeypatch.setenv("HISTORY_SEQUENCE_ENABLED", "false")

    reset_config()
    classifier = HybridClassifier(
        filename_weight=0.5,
        graph2d_weight=0.5,
        filename_min_conf=0.99,
    )
    classifier._filename_classifier = _FilenameStub()

    result = classifier.classify(  # noqa: SLF001
        filename="Bolt_M6x20.dxf",
        graph2d_result={"label": "bolt", "confidence": 0.8, "label_map_size": 10},
    )

    assert result.source == DecisionSource.FUSION
    assert result.label == "Bolt"
    assert result.confidence >= 0.84
    assert "fusion_multi_source_bonus" in result.decision_path
    reset_config()
