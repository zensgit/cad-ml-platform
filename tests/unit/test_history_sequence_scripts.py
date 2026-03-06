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


def test_build_history_sequence_prototypes_from_manifest(tmp_path: Path) -> None:
    _write_h5(tmp_path / "shaft_a.h5", [[6, 0], [10, 0], [15, 0]])
    _write_h5(tmp_path / "shaft_b.h5", [[6, 0], [10, 0], [10, 0]])
    _write_h5(tmp_path / "link_a.h5", [[1, 0], [2, 0], [1, 0]])

    manifest = [
        {"h5_path": str((tmp_path / "shaft_a.h5").resolve()), "label": "shaft"},
        {"h5_path": str((tmp_path / "shaft_b.h5").resolve()), "label": "shaft"},
        {"h5_path": str((tmp_path / "link_a.h5").resolve()), "label": "link"},
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    output_path = tmp_path / "prototypes.json"
    cmd = [
        sys.executable,
        "scripts/build_history_sequence_prototypes.py",
        "--manifest",
        str(manifest_path),
        "--top-k",
        "4",
        "--min-samples-per-label",
        "2",
        "--output",
        str(output_path),
    ]
    proc = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert set((payload.get("labels") or {}).keys()) == {"shaft"}
    meta = payload.get("meta") or {}
    assert int(meta.get("used_samples", 0)) == 3
    assert int(meta.get("input_pairs", 0)) == 3


def test_eval_history_sequence_classifier_from_manifest(tmp_path: Path) -> None:
    _write_h5(tmp_path / "A.h5", [[1, 0], [2, 0], [1, 0], [2, 0]])
    _write_h5(tmp_path / "B.h5", [[8, 0], [8, 0], [9, 0], [8, 0]])

    manifest = [
        {"h5_path": str((tmp_path / "A.h5").resolve()), "label": "A"},
        {"h5_path": str((tmp_path / "B.h5").resolve()), "label": "B"},
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    prototypes = {
        "labels": {
            "A": {"token_weights": {"1": 0.7, "2": 0.7}},
            "B": {"token_weights": {"8": 0.7, "9": 0.7}},
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
    assert summary["macro_f1_overall"] >= 0.99
