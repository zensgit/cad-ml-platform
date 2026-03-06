from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict


def test_diagnose_graph2d_reports_low_confidence_rate(
    monkeypatch, tmp_path: Path
) -> None:
    dxf_dir = tmp_path / "dxfs"
    dxf_dir.mkdir(parents=True, exist_ok=True)
    (dxf_dir / "sample.dxf").write_bytes(b"dummy-dxf-bytes")

    class FakeGraph2DClassifier:
        def __init__(self, model_path: str | None = None) -> None:
            self.model_path = model_path
            self.label_map = {"X": 0}

        def predict_from_bytes(self, data: bytes, file_name: str) -> Dict[str, Any]:
            _ = (data, file_name)
            return {
                "status": "ok",
                "label": "X",
                "confidence": 0.1,
                "temperature": 1.0,
                "temperature_source": "none",
                "label_map_size": len(self.label_map),
            }

    monkeypatch.setattr(
        "src.ml.vision_2d.Graph2DClassifier", FakeGraph2DClassifier, raising=True
    )

    from scripts.diagnose_graph2d_on_dxf_dir import main

    out_dir = tmp_path / "out"
    rc = main(
        [
            "--dxf-dir",
            str(dxf_dir),
            "--model-path",
            "dummy.pth",
            "--output-dir",
            str(out_dir),
            "--max-files",
            "1",
            "--low-conf-threshold",
            "0.2",
        ]
    )
    assert rc == 0

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    confidence = summary["confidence"]
    assert confidence["low_conf_threshold"] == 0.2
    assert confidence["low_conf_count"] == 1
    assert confidence["low_conf_rate"] == 1.0


def test_eval_trend_load_history_recognizes_explicit_ocr_type(
    monkeypatch, tmp_path: Path
) -> None:
    import scripts.eval_trend as eval_trend

    monkeypatch.setattr(eval_trend, "ROOT", tmp_path)
    history_dir = tmp_path / "eval_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    (history_dir / "combined.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-03-06T00:00:00Z",
                "type": "combined",
                "combined": {"combined_score": 0.8, "vision_weight": 0.5},
            }
        ),
        encoding="utf-8",
    )
    (history_dir / "ocr.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-03-06T01:00:00Z",
                "type": "ocr",
                "metrics": {"dimension_recall": 0.9, "brier_score": 0.1},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(eval_trend, "HISTORY_DIR", history_dir)
    combined, ocr_only = eval_trend.load_history()

    assert len(combined) == 1
    assert len(ocr_only) == 1


def test_eval_with_history_script_has_valid_bash_syntax() -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "eval_with_history.sh"
    proc = subprocess.run(
        ["bash", "-n", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
