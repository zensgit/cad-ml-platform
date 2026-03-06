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


def test_tune_history_sequence_weights_picks_bigram_weight(tmp_path: Path) -> None:
    _write_h5(tmp_path / "A.h5", [[1, 0], [2, 0], [1, 0], [2, 0], [1, 0], [2, 0]])
    _write_h5(tmp_path / "B.h5", [[1, 0], [1, 0], [2, 0], [2, 0], [2, 0], [2, 0]])

    manifest = [
        {"h5_path": str((tmp_path / "A.h5").resolve()), "label": "A"},
        {"h5_path": str((tmp_path / "B.h5").resolve()), "label": "B"},
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    prototypes = {
        "labels": {
            "A": {
                "token_weights": {"1": 0.0, "2": 0.0},
                "bigram_weights": {"1,2": 1.5, "2,1": 1.2},
            },
            "B": {
                "token_weights": {"1": 0.0, "2": 0.0},
                "bigram_weights": {"1,1": 1.2, "2,2": 1.2},
            },
        }
    }
    prototypes_path = tmp_path / "prototypes.json"
    prototypes_path.write_text(json.dumps(prototypes), encoding="utf-8")

    output_dir = tmp_path / "tuning"
    cmd = [
        sys.executable,
        "scripts/tune_history_sequence_weights.py",
        "--manifest",
        str(manifest_path),
        "--prototypes-path",
        str(prototypes_path),
        "--token-weight-grid",
        "1.0",
        "--bigram-weight-grid",
        "0.0,1.0",
        "--objective",
        "accuracy_overall",
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

    best_path = output_dir / "best_config.json"
    assert best_path.exists()
    payload = json.loads(best_path.read_text(encoding="utf-8"))
    best = payload.get("best") or {}
    assert float(best.get("bigram_weight", -1.0)) == 1.0
    assert float(best.get("accuracy_overall", 0.0)) >= 0.99
    env_file = Path((payload.get("artifacts") or {}).get("recommended_env_file") or "")
    assert env_file.exists()
    env_text = env_file.read_text(encoding="utf-8")
    assert "HISTORY_SEQUENCE_PROTOTYPE_TOKEN_WEIGHT=1.0" in env_text
    assert "HISTORY_SEQUENCE_PROTOTYPE_BIGRAM_WEIGHT=1.0" in env_text
