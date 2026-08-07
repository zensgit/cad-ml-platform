"""Real-path tests for Track E E1 operator dry-run entry."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import run_track_e_e1_dry_run as runner


def _write_manifest(tmp: Path) -> Path:
    root = tmp / "data"
    root.mkdir()
    rows = []
    for i, (name, label, body) in enumerate(
        [
            ("gear.dxf", "gear", b"A"),
            ("gear2.dxf", "gear", b"B"),
            ("bolt.dxf", "bolt", b"C"),
            ("nut.dxf", "nut", b"D"),
            ("plate.dxf", "plate", b"E"),
            ("washer.dxf", "washer", b"F"),
        ]
    ):
        p = root / name
        p.write_bytes(body)
        rows.append({"file_path": str(p.relative_to(root)), "taxonomy_v2_class": label})
    man = tmp / "manifest.csv"
    with man.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file_path", "taxonomy_v2_class"])
        w.writeheader()
        w.writerows(rows)
    return man, root


def test_run_dry_run_writes_artifacts_and_never_unlocks(tmp_path: Path) -> None:
    man, root = _write_manifest(tmp_path)
    out = tmp_path / "out"
    summary = runner.run_dry_run(
        manifest_csv=man,
        root=root,
        out_dir=out,
        holdout_fraction=0.4,
    )
    assert summary["unlocks_retraining"] is False
    assert Path(summary["split_artifact"]).is_file()
    assert Path(summary["manifest_artifact"]).is_file()
    split = json.loads(Path(summary["split_artifact"]).read_text(encoding="utf-8"))
    manifest = json.loads(Path(summary["manifest_artifact"]).read_text(encoding="utf-8"))
    assert split["unlocks_retraining"] is False
    assert manifest["unlocks_retraining"] is False
    assert len(summary["split_digest"]) == 64
    assert len(summary["manifest_digest"]) == 64


def test_cli_main_round_trip(tmp_path: Path) -> None:
    man, root = _write_manifest(tmp_path)
    out = tmp_path / "cli-out"
    rc = runner.main(
        [
            "--manifest",
            str(man),
            "--root",
            str(root),
            "--out",
            str(out),
            "--holdout-fraction",
            "0.4",
        ]
    )
    assert rc == 0
    assert (out / "e1_split_artifact.json").is_file()


def test_module_does_not_import_gate() -> None:
    import inspect

    src = inspect.getsource(runner)
    assert "import eval_integrity_gate" not in src
    assert "from eval_integrity_gate" not in src
