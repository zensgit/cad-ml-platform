from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "ci"
    / "archive_graph2d_context_drift_artifacts.py"
)


def test_archive_script_copies_and_writes_manifest(tmp_path: Path) -> None:
    src_a = tmp_path / "a.json"
    src_b = tmp_path / "b.md"
    src_a.write_text("{}", encoding="utf-8")
    src_b.write_text("# report\n", encoding="utf-8")

    output_root = tmp_path / "reports" / "experiments"
    manifest_json = tmp_path / "manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--output-root",
            str(output_root),
            "--date",
            "20260217",
            "--bucket",
            "graph2d_context_drift_test",
            "--artifact",
            str(src_a),
            "--artifact",
            str(src_b),
            "--manifest-json",
            str(manifest_json),
            "--require-exists",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["copied_count"] == 2
    assert manifest["missing_count"] == 0
    archive_dir = Path(manifest["archive_dir"])
    assert (archive_dir / "a.json").exists()
    assert (archive_dir / "b.md").exists()


def test_archive_script_fails_when_required_artifact_missing(tmp_path: Path) -> None:
    src_a = tmp_path / "a.json"
    src_a.write_text("{}", encoding="utf-8")
    missing = tmp_path / "missing.json"
    manifest_json = tmp_path / "manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--output-root",
            str(tmp_path / "reports" / "experiments"),
            "--date",
            "20260217",
            "--bucket",
            "graph2d_context_drift_test",
            "--artifact",
            str(src_a),
            "--artifact",
            str(missing),
            "--manifest-json",
            str(manifest_json),
            "--require-exists",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["missing_count"] == 1
