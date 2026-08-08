"""Tests for review_reuse_store_ops backup/cleanup/list CLI."""

from __future__ import annotations

import json
import os
import tarfile
import time
from pathlib import Path

from scripts.review_reuse_store_ops import (
    cmd_backup,
    cmd_cleanup,
    cmd_list,
    collect_tenant_summaries,
    main,
)


def _seed_tenant(
    store: Path, tenant: str, age_days: float, *, task_count: int = 1
) -> Path:
    tdir = store / tenant / "tasks"
    tdir.mkdir(parents=True, exist_ok=True)
    mtime = time.time() - (age_days * 86400.0)
    for i in range(task_count):
        f = tdir / f"task{i + 1}.json"
        f.write_text(f'{{"task_id":"task{i + 1}"}}', encoding="utf-8")
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


def test_list_empty_store(tmp_path: Path) -> None:
    store = tmp_path / "empty_store"
    store.mkdir()
    rows = collect_tenant_summaries(store)
    assert rows == []
    assert cmd_list(store, as_json=False) == 0
    assert cmd_list(store, as_json=True) == 0


def test_list_two_tenants(tmp_path: Path, capsys) -> None:
    store = tmp_path / "store"
    _seed_tenant(store, "alpha", 2.0, task_count=2)
    _seed_tenant(store, "beta", 10.0, task_count=1)

    rows = collect_tenant_summaries(store)
    by_tenant = {r["tenant"]: r for r in rows}
    assert set(by_tenant) == {"alpha", "beta"}
    assert by_tenant["alpha"]["task_count"] == 2
    assert by_tenant["beta"]["task_count"] == 1
    assert by_tenant["alpha"]["age_days"] is not None
    assert by_tenant["beta"]["age_days"] is not None
    assert abs(float(by_tenant["alpha"]["age_days"]) - 2.0) < 0.05
    assert abs(float(by_tenant["beta"]["age_days"]) - 10.0) < 0.05

    assert cmd_list(store, as_json=True) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["store_dir"] == str(store.resolve())
    tenants = {t["tenant"]: t for t in payload["tenants"]}
    assert tenants["alpha"]["task_count"] == 2
    assert tenants["beta"]["task_count"] == 1

    rc = main(["list", "--store-dir", str(store)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "tenant=alpha" in text
    assert "tenant=beta" in text
    assert "tasks=2" in text
    assert "tasks=1" in text
