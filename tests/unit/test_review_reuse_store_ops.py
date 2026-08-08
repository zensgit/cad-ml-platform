"""Tests for review_reuse_store_ops backup/cleanup CLI."""

from __future__ import annotations

import tarfile
import time
from pathlib import Path

from scripts.review_reuse_store_ops import cmd_backup, cmd_cleanup, main


def _seed_tenant(store: Path, tenant: str, age_days: float) -> Path:
    tdir = store / tenant / "tasks"
    tdir.mkdir(parents=True, exist_ok=True)
    f = tdir / "task1.json"
    f.write_text('{"task_id":"task1"}', encoding="utf-8")
    mtime = time.time() - (age_days * 86400.0)
    # touch mtime
    import os

    os.utime(f, (mtime, mtime))
    return tdir


def test_backup_creates_tarball(tmp_path: Path) -> None:
    store = tmp_path / "store"
    _seed_tenant(store, "tenant-a", 1.0)
    out = tmp_path / "backups"
    assert cmd_backup(store, out) == 0
    archives = list(out.glob("review_reuse_store_*.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0], "r:gz") as tar:
        names = tar.getnames()
    assert any("tenant-a" in n for n in names)


def test_cleanup_dry_run_and_apply(tmp_path: Path) -> None:
    store = tmp_path / "store"
    _seed_tenant(store, "old-tenant", 60.0)
    _seed_tenant(store, "new-tenant", 1.0)

    assert (
        cmd_cleanup(store, older_than_days=30, dry_run=True, tenant=None) == 0
    )
    assert (store / "old-tenant").is_dir()
    assert (store / "new-tenant").is_dir()

    assert (
        cmd_cleanup(store, older_than_days=30, dry_run=False, tenant=None) == 0
    )
    assert not (store / "old-tenant").exists()
    assert (store / "new-tenant").is_dir()


def test_main_cleanup_apply_flag(tmp_path: Path) -> None:
    store = tmp_path / "store"
    _seed_tenant(store, "old-t", 90.0)
    rc = main(
        [
            "cleanup",
            "--store-dir",
            str(store),
            "--older-than-days",
            "30",
            "--apply",
        ]
    )
    assert rc == 0
    assert not (store / "old-t").exists()
