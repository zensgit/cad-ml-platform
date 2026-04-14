from __future__ import annotations

import json
import os
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
    combined, ocr_only, history_sequence = eval_trend.load_history()

    assert len(combined) == 1
    assert len(ocr_only) == 1
    assert history_sequence == []


def test_eval_with_history_script_has_valid_bash_syntax() -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "eval_with_history.sh"
    proc = subprocess.run(
        ["bash", "-n", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_eval_with_history_writes_coarse_history_metrics(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "eval_with_history.sh"
    report_dir = tmp_path / "eval_history"
    history_output_dir = tmp_path / "history_eval"

    ocr_script = tmp_path / "fake_ocr.py"
    ocr_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                'print(\'dimension_recall=0.8 brier_score=0.1 edge_f1=0.2\')',
            ]
        ),
        encoding="utf-8",
    )
    ocr_script.chmod(0o755)

    build_script = tmp_path / "fake_build.py"
    build_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json, pathlib, sys",
                "out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])",
                "out.parent.mkdir(parents=True, exist_ok=True)",
                "payload = {'labels': {'轴类': {'token_weights': {'1': 1.0}}}}",
                "out.write_text(",
                "    json.dumps(payload, ensure_ascii=False),",
                "    encoding='utf-8',",
                ")",
            ]
        ),
        encoding="utf-8",
    )
    build_script.chmod(0o755)

    eval_script = tmp_path / "fake_eval.py"
    eval_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import csv, json, pathlib, sys",
                "out = pathlib.Path(sys.argv[sys.argv.index('--output-dir') + 1])",
                "out.mkdir(parents=True, exist_ok=True)",
                "summary = {",
                "  'coverage': 1.0,",
                "  'accuracy_overall': 0.5,",
                "  'macro_f1_overall': 0.5,",
                "  'coarse_accuracy_on_ok': 1.0,",
                "  'coarse_accuracy_overall': 1.0,",
                "  'coarse_macro_f1_on_ok': 1.0,",
                "  'coarse_macro_f1_overall': 1.0,",
                "  'exact_top_mismatches': [{'expected': '捕集口', 'predicted': '人孔', 'count': 1}],",
                "  'coarse_top_mismatches': []",
                "}",
                "(out / 'summary.json').write_text(",
                "    json.dumps(summary, ensure_ascii=False),",
                "    encoding='utf-8',",
                ")",
                "with (out / 'results.csv').open('w', encoding='utf-8', newline='') as handle:",
                "    writer = csv.DictWriter(handle, fieldnames=['ok'])",
                "    writer.writeheader()",
                "    writer.writerow({'ok': 'Y'})",
                "print(",
                "    json.dumps({'summary_path': str(out / 'summary.json')}, ensure_ascii=False)",
                ")",
            ]
        ),
        encoding="utf-8",
    )
    eval_script.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "EVAL_HISTORY_REPORT_DIR": str(report_dir),
            "EVAL_HISTORY_OCR_SCRIPT": str(ocr_script),
            "EVAL_HISTORY_BUILD_SCRIPT": str(build_script),
            "EVAL_HISTORY_EVAL_SCRIPT": str(eval_script),
            "HISTORY_SEQUENCE_EVAL_ENABLE": "true",
            "HISTORY_SEQUENCE_EVAL_H5_DIR": str(tmp_path),
            "HISTORY_SEQUENCE_EVAL_OUTPUT_DIR": str(history_output_dir),
        }
    )

    proc = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        env=env,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout

    history_files = sorted(report_dir.glob("*_history_sequence.json"))
    assert history_files
    payload = json.loads(history_files[-1].read_text(encoding="utf-8"))
    history_metrics = payload["history_metrics"]
    assert history_metrics["coarse_accuracy_overall"] == 1.0
    assert history_metrics["coarse_macro_f1_overall"] == 1.0
    assert history_metrics["exact_top_mismatches"] == [
        {"expected": "捕集口", "predicted": "人孔", "count": 1}
    ]
    assert history_metrics["coarse_top_mismatches"] == []
