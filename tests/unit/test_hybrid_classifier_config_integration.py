from __future__ import annotations

from pathlib import Path

from src.ml.hybrid_classifier import HybridClassifier
from src.ml.hybrid_config import reset_config


def test_hybrid_classifier_uses_yaml_defaults(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / "hybrid.yaml"
    cfg.write_text(
        "\n".join(
            [
                "filename:",
                "  fusion_weight: 0.82",
                "  min_confidence: 0.88",
                "graph2d:",
                "  enabled: true",
                "  min_confidence: 0.41",
                "process:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HYBRID_CONFIG_PATH", str(cfg))
    monkeypatch.delenv("FILENAME_FUSION_WEIGHT", raising=False)
    monkeypatch.delenv("FILENAME_MIN_CONF", raising=False)
    monkeypatch.delenv("GRAPH2D_ENABLED", raising=False)
    monkeypatch.delenv("GRAPH2D_MIN_CONF", raising=False)
    monkeypatch.delenv("PROCESS_FEATURES_ENABLED", raising=False)
    reset_config()

    classifier = HybridClassifier()
    assert classifier.filename_weight == 0.82
    assert classifier.filename_min_conf == 0.88
    assert classifier.graph2d_min_conf == 0.41
    assert classifier._is_graph2d_enabled() is True
    assert classifier._is_process_enabled() is False

    reset_config()
