from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "ci"
    / "archive_experiment_dirs.py"
)


def _make_experiment_dir(root: Path, token: str, file_name: str = "result.json") -> Path:
    target = root / token
    target.mkdir(parents=True, exist_ok=True)
    (target / file_name).write_text("{\"ok\": true}\n", encoding="utf-8")
    return target


def test_archive_experiment_dirs_explicit_and_delete(tmp_path: Path) -> None:
    experiments_root = tmp_path / "reports" / "experiments"
    archive_root = tmp_path / "archives"
    manifest_json = tmp_path / "manifest.json"
    _make_experiment_dir(experiments_root, "20260217")
    _make_experiment_dir(experiments_root, "20260219")

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--experiments-root",
            str(experiments_root),
            "--archive-root",
            str(archive_root),
            "--dir",
            "20260217",
            "--delete-source",
            "--today",
            "20260220",
            "--manifest-json",
            str(manifest_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["selected_count"] == 1
    assert manifest["archived_count"] == 1
    assert manifest["deleted_count"] == 1
    assert not (experiments_root / "20260217").exists()
    assert (experiments_root / "20260219").exists()

    archive_file = Path(manifest["rows"][0]["archive_file"])
    assert archive_file.exists()
    with tarfile.open(archive_file, mode="r:gz") as handle:
        names = handle.getnames()
    assert any(name.startswith("20260217/") for name in names)


def test_archive_experiment_dirs_dry_run_by_age(tmp_path: Path) -> None:
    experiments_root = tmp_path / "reports" / "experiments"
    archive_root = tmp_path / "archives"
    manifest_json = tmp_path / "manifest.json"
    _make_experiment_dir(experiments_root, "20260210")
    _make_experiment_dir(experiments_root, "20260219")

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--experiments-root",
            str(experiments_root),
            "--archive-root",
            str(archive_root),
            "--today",
            "20260220",
            "--keep-latest-days",
            "7",
            "--dry-run",
            "--manifest-json",
            str(manifest_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["selected_count"] == 1
    assert manifest["archived_count"] == 0
    assert manifest["rows"][0]["token"] == "20260210"
    assert manifest["rows"][0]["status"] == "dry_run"
    assert (experiments_root / "20260210").exists()
    assert (experiments_root / "20260219").exists()


def test_archive_experiment_dirs_require_exists_fails(tmp_path: Path) -> None:
    experiments_root = tmp_path / "reports" / "experiments"
    archive_root = tmp_path / "archives"
    manifest_json = tmp_path / "manifest.json"
    _make_experiment_dir(experiments_root, "20260217")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--experiments-root",
            str(experiments_root),
            "--archive-root",
            str(archive_root),
            "--dir",
            "20260217",
            "--dir",
            "20260218",
            "--require-exists",
            "--manifest-json",
            str(manifest_json),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["missing_count"] == 1
