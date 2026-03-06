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


def test_build_history_sequence_prototypes_writes_payload(tmp_path: Path) -> None:
    _write_h5(tmp_path / "a1.h5", [[1, 0], [2, 0], [1, 0], [2, 0]])
    _write_h5(tmp_path / "a2.h5", [[1, 0], [2, 0], [1, 0]])
    _write_h5(tmp_path / "b1.h5", [[9, 0], [9, 0], [8, 0]])

    manifest = [
        {"h5_path": str((tmp_path / "a1.h5").resolve()), "label": "A"},
        {"h5_path": str((tmp_path / "a2.h5").resolve()), "label": "A"},
        {"h5_path": str((tmp_path / "b1.h5").resolve()), "label": "B"},
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    output_path = tmp_path / "prototypes.json"
    cmd = [
        sys.executable,
        "scripts/build_history_sequence_prototypes.py",
        "--manifest",
        str(manifest_path),
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
    assert set((payload.get("labels") or {}).keys()) == {"A"}
    assert (payload.get("meta") or {}).get("used_samples") == 3
