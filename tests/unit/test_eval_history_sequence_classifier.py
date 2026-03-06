from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

h5py = pytest.importorskip("h5py")


def _write_h5(path: Path, rows: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("vec", data=rows, dtype="i4")


def test_eval_history_sequence_classifier_writes_summary(tmp_path: Path) -> None:
    _write_h5(tmp_path / "a.h5", [[1, 0], [2, 0], [1, 0], [2, 0]])
    _write_h5(tmp_path / "b.h5", [[9, 0], [8, 0], [9, 0], [8, 0]])

    manifest = [
        {"h5_path": str((tmp_path / "a.h5").resolve()), "label": "A"},
        {"h5_path": str((tmp_path / "b.h5").resolve()), "label": "B"},
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    prototypes = {
        "labels": {
            "A": {"token_weights": {"1": 0.6, "2": 0.4}},
            "B": {"token_weights": {"9": 0.6, "8": 0.4}},
        }
    }
    prototypes_path = tmp_path / "prototypes.json"
    prototypes_path.write_text(json.dumps(prototypes), encoding="utf-8")

    output_dir = tmp_path / "eval"
    cmd = [
        sys.executable,
        "scripts/eval_history_sequence_classifier.py",
        "--manifest",
        str(manifest_path),
        "--prototypes-path",
        str(prototypes_path),
        "--min-seq-len",
        "2",
        "--output-dir",
        str(output_dir),
    ]
    proc = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr or proc.stdout
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["total"] == 2
    assert summary["ok_count"] == 2
    assert summary["accuracy_overall"] >= 0.99
