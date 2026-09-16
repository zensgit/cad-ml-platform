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
from src.core.review_reuse.store import tenant_dir_key


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


def test_list_uses_tenant_meta_id(tmp_path: Path) -> None:
    store = tmp_path / "store"
    hashed = "aaaaaaaaaaaaaaaaaaaaaaaa"
    tdir = store / hashed
    tasks = tdir / "tasks"
    tasks.mkdir(parents=True)
    (tasks / "task1.json").write_text('{"task_id":"task1"}', encoding="utf-8")
    (tdir / "tenant_meta.json").write_text(
        '{"tenant_id":"a/b"}', encoding="utf-8"
    )
    rows = collect_tenant_summaries(store)
    assert rows[0]["tenant"] == "a/b"
    assert rows[0]["task_count"] == 1


def _seed_hashed_tenant(
    store: Path, tenant: str, age_days: float
) -> Path:
    hashed = tenant_dir_key(tenant)
    tdir = store / hashed
    tasks = tdir / "tasks"
    tasks.mkdir(parents=True, exist_ok=True)
    f = tasks / "task1.json"
    f.write_text('{"task_id":"task1"}', encoding="utf-8")
    mtime = time.time() - (age_days * 86400.0)
    os.utime(f, (mtime, mtime))
    (tdir / "tenant_meta.json").write_text(
        json.dumps({"tenant_id": tenant}), encoding="utf-8"
    )
    return tdir


def test_cleanup_matches_hashed_tenant_by_original_id(
    tmp_path: Path, capsys
) -> None:
    store = tmp_path / "store"
    hashed_dir = _seed_hashed_tenant(store, "pilot-tenant", 60.0)
    other_dir = _seed_hashed_tenant(store, "other-tenant", 60.0)

    assert (
        cmd_cleanup(
            store, older_than_days=30, dry_run=True, tenant="pilot-tenant"
        )
        == 0
    )
    dry = capsys.readouterr()
    assert "would_delete tenant=pilot-tenant" in dry.out
    assert hashed_dir.is_dir()
    assert other_dir.is_dir()

    assert (
        cmd_cleanup(
            store, older_than_days=30, dry_run=False, tenant="pilot-tenant"
        )
        == 0
    )
    applied = capsys.readouterr()
    assert "deleted tenant=pilot-tenant" in applied.out
    assert not hashed_dir.exists()
    assert other_dir.is_dir()


def test_cleanup_matches_hashed_dir_basename(tmp_path: Path) -> None:
    store = tmp_path / "store"
    hashed_dir = _seed_hashed_tenant(store, "pilot-tenant", 60.0)
    hashed = hashed_dir.name
    assert (
        cmd_cleanup(store, older_than_days=30, dry_run=False, tenant=hashed)
        == 0
    )
    assert not hashed_dir.exists()


def test_list_and_cleanup_recover_legacy_sanitized_tenant_id(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store"
    hashed_dir = _seed_hashed_tenant(store, "a/b", 10.0)
    legacy = store / "a_b" / "tasks"
    legacy.mkdir(parents=True)
    task = legacy / "task1.json"
    task.write_text('{"task_id":"task1","tenant_id":"a/b"}', encoding="utf-8")
    mtime = time.time() - (20.0 * 86400.0)
    os.utime(task, (mtime, mtime))

    rows = collect_tenant_summaries(store)
    by_tenant = {r["tenant"]: r for r in rows}
    assert set(by_tenant) == {"a/b"}
    assert by_tenant["a/b"]["task_count"] == 1

    assert (
        cmd_cleanup(store, older_than_days=5, dry_run=False, tenant="a/b")
        == 0
    )
    assert not hashed_dir.exists()
    assert not (store / "a_b").exists()


def test_cleanup_refuses_mixed_legacy_tenant_dir(
    tmp_path: Path, capsys
) -> None:
    store = tmp_path / "store"
    hashed_dir = _seed_hashed_tenant(store, "a/b", 60.0)
    mixed = store / "a_b" / "tasks"
    mixed.mkdir(parents=True)
    old = time.time() - (60.0 * 86400.0)
    for name, tenant in (("task-slash.json", "a/b"), ("task-under.json", "a_b")):
        path = mixed / name
        path.write_text(
            json.dumps({"task_id": name, "tenant_id": tenant}),
            encoding="utf-8",
        )
        os.utime(path, (old, old))

    rows = collect_tenant_summaries(store)
    by_tenant = {r["tenant"]: r for r in rows}
    assert "a/b" in by_tenant
    assert "a_b" in by_tenant
    assert by_tenant["a/b"]["task_count"] == 1
    assert by_tenant["a_b"]["task_count"] == 2

    assert (
        cmd_cleanup(store, older_than_days=30, dry_run=False, tenant="a/b")
        == 1
    )
    err = capsys.readouterr().err
    assert "refused_mixed" in err
    assert "a/b" in err
    assert "a_b" in err
    assert not hashed_dir.exists()
    assert (store / "a_b").is_dir()
    assert (mixed / "task-under.json").is_file()


def test_list_merges_legacy_and_hashed_same_tenant(tmp_path: Path) -> None:
    store = tmp_path / "store"
    hashed_dir = _seed_hashed_tenant(store, "pilot-tenant", 10.0)
    extra = hashed_dir / "tasks" / "task2.json"
    extra.write_text('{"task_id":"task2"}', encoding="utf-8")
    extra_mtime = time.time() - (10.0 * 86400.0)
    os.utime(extra, (extra_mtime, extra_mtime))
    _seed_tenant(store, "pilot-tenant", 20.0, task_count=1)

    rows = collect_tenant_summaries(store)
    by_tenant = {r["tenant"]: r for r in rows}
    assert set(by_tenant) == {"pilot-tenant"}
    # task1 is in both layouts; task2 only hashed → unique count 2, not 3.
    assert by_tenant["pilot-tenant"]["task_count"] == 2
    assert abs(float(by_tenant["pilot-tenant"]["age_days"]) - 10.0) < 0.05


def test_cleanup_unknown_tenant_returns_not_found(
    tmp_path: Path, capsys
) -> None:
    store = tmp_path / "store"
    _seed_hashed_tenant(store, "pilot-tenant", 60.0)
    assert (
        cmd_cleanup(
            store, older_than_days=30, dry_run=True, tenant="missing-tenant"
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "tenant not found: missing-tenant" in err
