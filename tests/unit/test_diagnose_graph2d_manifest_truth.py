"""Regression tests for scripts/diagnose_graph2d_on_dxf_dir.py manifest truth mode.

Keep this test torch-free: we stub the `src.ml.vision_2d` module so the script
does not import heavy optional dependencies.
"""

from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict

import pytest


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_diagnose_graph2d_supports_manifest_truth_mode(tmp_path, monkeypatch):
    # Arrange: create fake DXF inputs (bytes need not be valid for this test).
    dxf_dir = tmp_path / "dxfs"
    dxf_dir.mkdir()
    (dxf_dir / "a.dxf").write_bytes(b"dummy-a")
    (dxf_dir / "b.dxf").write_bytes(b"dummy-b")

    # Manifest provides the ground-truth coarse labels.
    manifest_csv = tmp_path / "manifest.csv"
    _write_manifest(
        manifest_csv,
        rows=[
            {
                "file_name": "a.dxf",
                "label_cn": "bucket_a",
                "relative_path": "a.dxf",
                "label_confidence": "0.9500",
            },
            {
                "file_name": "b.dxf",
                "label_cn": "bucket_b",
                "relative_path": "b.dxf",
                "label_confidence": "0.9500",
            },
        ],
    )

    class StubGraph2DClassifier:
        def __init__(self, model_path: str):
            self.model_path = model_path
            self.label_map = {"bucket_a": 0, "bucket_b": 1}

        def predict_from_bytes(self, data: bytes, file_name: str) -> Dict[str, Any]:
            _ = data
            # Predict bucket_a for both files: 1 correct, 1 wrong => accuracy 0.5
            return {
                "status": "ok",
                "label": "bucket_a",
                "confidence": 0.9 if file_name == "a.dxf" else 0.6,
                "temperature": 1.0,
                "temperature_source": "test",
            }

    fake_vision = types.ModuleType("src.ml.vision_2d")
    fake_vision.Graph2DClassifier = StubGraph2DClassifier  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.ml.vision_2d", fake_vision)

    # Act: run the script main() with manifest truth enabled.
    import scripts.diagnose_graph2d_on_dxf_dir as diagnose

    out_dir = tmp_path / "out"
    argv = [
        "diagnose_graph2d_on_dxf_dir.py",
        "--dxf-dir",
        str(dxf_dir),
        "--model-path",
        "dummy.pth",
        "--manifest-csv",
        str(manifest_csv),
        "--max-files",
        "2",
        "--seed",
        "1",
        "--output-dir",
        str(out_dir),
    ]
    monkeypatch.setattr(diagnose.sys, "argv", argv)
    rc = diagnose.main()

    # Assert: manifest truth is reflected in the summary and accuracy is computed.
    assert rc == 0
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["true_labels"]["source"] == "manifest"
    assert summary["true_labels"]["coverage"] == 2
    assert summary["accuracy"] == 0.5
    assert summary["per_class_accuracy"] is not None

    preds_path = out_dir / "predictions.csv"
    assert preds_path.exists()
    with preds_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [r.get("relative_path") for r in rows] == ["a.dxf", "b.dxf"]


def test_diagnose_graph2d_manifest_truth_requires_existing_manifest(tmp_path, monkeypatch):
    import scripts.diagnose_graph2d_on_dxf_dir as diagnose

    dxf_dir = tmp_path / "dxfs"
    dxf_dir.mkdir()
    (dxf_dir / "a.dxf").write_bytes(b"dummy-a")

    argv = [
        "diagnose_graph2d_on_dxf_dir.py",
        "--dxf-dir",
        str(dxf_dir),
        "--model-path",
        "dummy.pth",
        "--manifest-csv",
        str(tmp_path / "missing.csv"),
        "--max-files",
        "1",
        "--seed",
        "1",
        "--output-dir",
        str(tmp_path / "out"),
    ]
    monkeypatch.setattr(diagnose.sys, "argv", argv)

    with pytest.raises(SystemExit):
        diagnose.main()
