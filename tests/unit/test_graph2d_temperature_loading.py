from __future__ import annotations

import json
from pathlib import Path


def test_graph2d_temperature_env_overrides_calibration(monkeypatch, tmp_path: Path) -> None:
    from src.ml.vision_2d import Graph2DClassifier

    cal_path = tmp_path / "graph2d_calibration.json"
    cal_path.write_text(json.dumps({"temperature": 0.5}), encoding="utf-8")

    monkeypatch.setenv("GRAPH2D_TEMPERATURE_CALIBRATION_PATH", str(cal_path))
    monkeypatch.setenv("GRAPH2D_TEMPERATURE", "0.25")

    clf = Graph2DClassifier(model_path="models/does_not_exist.pth")
    assert clf.temperature == 0.25
    assert clf.temperature_source == "env"


def test_graph2d_temperature_calibration_file_used(monkeypatch, tmp_path: Path) -> None:
    from src.ml.vision_2d import Graph2DClassifier

    cal_path = tmp_path / "graph2d_calibration.json"
    cal_path.write_text(json.dumps({"temperature": 0.5}), encoding="utf-8")

    monkeypatch.delenv("GRAPH2D_TEMPERATURE", raising=False)
    monkeypatch.setenv("GRAPH2D_TEMPERATURE_CALIBRATION_PATH", str(cal_path))

    clf = Graph2DClassifier(model_path="models/does_not_exist.pth")
    assert clf.temperature == 0.5
    assert clf.temperature_source == str(cal_path)

