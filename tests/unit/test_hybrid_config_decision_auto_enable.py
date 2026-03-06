from __future__ import annotations

from pathlib import Path

from src.ml.hybrid_config import HybridClassifierConfig


def test_hybrid_config_loads_auto_enable_and_decision_overrides(
    tmp_path: Path, monkeypatch
) -> None:
    cfg_path = tmp_path / "hybrid.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "auto_enable:",
                "  titleblock_on_text: false",
                "  history_on_path: false",
                "decision:",
                "  advanced_fusion_enabled: false",
                "  fusion_strategy: voting",
                "  auto_select_fusion: false",
                "  explanation_enabled: false",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TITLEBLOCK_AUTO_ENABLE", "true")
    monkeypatch.setenv("HISTORY_SEQUENCE_AUTO_ENABLE", "true")
    monkeypatch.setenv("HYBRID_ADVANCED_FUSION_ENABLED", "true")
    monkeypatch.setenv("HYBRID_FUSION_STRATEGY", "weighted_average")
    monkeypatch.setenv("HYBRID_AUTO_SELECT_FUSION", "true")
    monkeypatch.setenv("HYBRID_EXPLANATION_ENABLED", "true")

    cfg = HybridClassifierConfig.from_sources(config_path=cfg_path)

    assert cfg.auto_enable.titleblock_on_text is True
    assert cfg.auto_enable.history_on_path is True
    assert cfg.decision.advanced_fusion_enabled is True
    assert cfg.decision.fusion_strategy == "weighted_average"
    assert cfg.decision.auto_select_fusion is True
    assert cfg.decision.explanation_enabled is True
