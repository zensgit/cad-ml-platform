from __future__ import annotations

from pathlib import Path

from src.ml.hybrid_config import HybridClassifierConfig, get_config, reset_config


def test_hybrid_config_file_then_env_override(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / "hybrid.yaml"
    cfg.write_text(
        "\n".join(
            [
                "enabled: true",
                "filename:",
                "  enabled: false",
                "  min_confidence: 0.66",
                "graph2d:",
                "  enabled: true",
                "  fusion_weight: 0.44",
                "process:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FILENAME_MIN_CONF", "0.91")
    config = HybridClassifierConfig.from_sources(config_path=cfg)
    assert config.filename.enabled is False
    assert config.filename.min_confidence == 0.91
    assert config.graph2d.enabled is True
    assert config.graph2d.fusion_weight == 0.44
    assert config.process.enabled is False


def test_get_config_honors_hybrid_config_path(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / "hybrid.yaml"
    cfg.write_text(
        "\n".join(
            [
                "filename:",
                "  min_confidence: 0.73",
                "titleblock:",
                "  enabled: true",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HYBRID_CONFIG_PATH", str(cfg))
    reset_config()
    loaded = get_config()
    assert loaded.filename.min_confidence == 0.73
    assert loaded.titleblock.enabled is True
    reset_config()
