"""Tests for review_reuse_store_ops backup/cleanup/list CLI."""

from __future__ import annotations

import hashlib
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
from src.core.review_reuse.service import ReviewReuseService
from src.core.review_reuse.store import FilesystemReviewReuseStore


def _tenant_dir(store: Path, tenant: str) -> Path:
    digest = hashlib.sha256(tenant.encode("utf-8")).hexdigest()
    return store / f"tenant-v1-{digest}"


def _seed_tenant(
    store: Path, tenant: str, age_days: float, *, task_count: int = 1
) -> Path:
    ledger = FilesystemReviewReuseStore(store)
    service = ReviewReuseService(ledger)
    try:
        for i in range(task_count):
            service.create_task(
                tenant_id=tenant,
                file_name=f"task{i + 1}.dxf",
                file_bytes=f"task-{i + 1}".encode("utf-8"),
            )
    finally:
        ledger.close()
    tdir = _tenant_dir(store, tenant) / "tasks"
    mtime = time.time() - (age_days * 86400.0)
    for task_file in tdir.glob("*.json"):
        os.utime(task_file, (mtime, mtime))
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
    assert any(_tenant_dir(store, "tenant-a").name in name for name in names)
    assert any(name.endswith("/tenant.json") for name in names)


def test_cleanup_dry_run_and_apply(tmp_path: Path) -> None:
    store = tmp_path / "store"
    _seed_tenant(store, "old-tenant", 60.0)
    _seed_tenant(store, "new-tenant", 1.0)

    assert cmd_cleanup(store, older_than_days=30, dry_run=True, tenant=None) == 0
    assert _tenant_dir(store, "old-tenant").is_dir()
    assert _tenant_dir(store, "new-tenant").is_dir()

    assert cmd_cleanup(store, older_than_days=30, dry_run=False, tenant=None) == 0
    assert not _tenant_dir(store, "old-tenant").exists()
    assert _tenant_dir(store, "new-tenant").is_dir()


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
    assert not _tenant_dir(store, "old-t").exists()


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


def test_ops_fail_closed_on_corrupt_sidecar(tmp_path: Path) -> None:
    store = tmp_path / "store"
    tasks_dir = _seed_tenant(store, "tenant-a", 60.0)
    sidecar = tasks_dir.parent / "tenant.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    assert cmd_list(store) == 2
    assert cmd_backup(store, tmp_path / "backups") == 2
    assert cmd_cleanup(store, older_than_days=30, dry_run=True, tenant=None) == 2
    assert cmd_cleanup(store, older_than_days=30, dry_run=False, tenant=None) == 2
    assert tasks_dir.parent.is_dir()


def test_cleanup_apply_rejects_active_writer(tmp_path: Path) -> None:
    store = tmp_path / "store"
    tasks_dir = _seed_tenant(store, "tenant-a", 60.0)
    writer = FilesystemReviewReuseStore(store)
    try:
        assert cmd_cleanup(store, older_than_days=30, dry_run=False, tenant=None) == 2
        assert tasks_dir.parent.is_dir()
    finally:
        writer.close()
