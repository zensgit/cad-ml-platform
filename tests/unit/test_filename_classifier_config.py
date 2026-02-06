from __future__ import annotations

from pathlib import Path

from src.ml.filename_classifier import FilenameClassifier
from src.ml.hybrid_config import reset_config


def test_filename_classifier_reads_synonyms_from_hybrid_config(
    tmp_path: Path, monkeypatch
) -> None:
    synonyms = tmp_path / "synonyms.json"
    synonyms.write_text('{"人孔": ["人孔"]}', encoding="utf-8")
    cfg = tmp_path / "hybrid.yaml"
    cfg.write_text(
        "\n".join(
            [
                "filename:",
                f"  synonyms_path: {synonyms}",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HYBRID_CONFIG_PATH", str(cfg))
    monkeypatch.delenv("FILENAME_SYNONYMS_PATH", raising=False)
    reset_config()

    classifier = FilenameClassifier()
    assert "人孔" in classifier.synonyms
    assert classifier.predict("J2925001-01人孔v2.dxf")["label"] == "人孔"
    reset_config()


def test_filename_classifier_env_overrides_config_threshold(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = tmp_path / "hybrid.yaml"
    cfg.write_text(
        "\n".join(
            [
                "filename:",
                "  exact_match_conf: 0.77",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HYBRID_CONFIG_PATH", str(cfg))
    monkeypatch.setenv("FILENAME_EXACT_MATCH_CONF", "0.91")
    reset_config()

    classifier = FilenameClassifier()
    assert classifier.exact_match_conf == 0.91
    reset_config()
