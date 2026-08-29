"""Fail-first integrity contract for the ReviewReuse ER1 store."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import pytest

from src.core.review_reuse.models import ReviewReuseTask, TaskStatus
from src.core.review_reuse.store import FilesystemReviewReuseStore


def _task(
    tenant_id: str,
    *,
    task_id: str | None = None,
    idempotency_key: str | None = None,
    idempotency_digest: str | None = None,
) -> ReviewReuseTask:
    return ReviewReuseTask(
        task_id=task_id or str(uuid.uuid4()),
        tenant_id=tenant_id,
        status=TaskStatus.evidence_ready,
        created_at=1.0,
        updated_at=1.0,
        source_file_name="part.dxf",
        source_content_sha256="a" * 64,
        idempotency_key=idempotency_key,
        idempotency_digest=idempotency_digest,
        revision=1,
        trace_id=str(uuid.uuid4()),
        calibration_version="workbench-mvp-0",
        calibration_status="uncalibrated",
        error_code=None,
    )


def _tenant_dir(root: Path, tenant_id: str) -> Path:
    digest = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()
    return root / f"tenant-v1-{digest}"


def _close(store: FilesystemReviewReuseStore) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        close()


def _error_code(exc: BaseException) -> str | None:
    return getattr(exc, "code", None)


def _create(
    store: FilesystemReviewReuseStore,
    task: ReviewReuseTask,
) -> ReviewReuseTask:
    return store.create_if_absent(  # type: ignore[attr-defined]
        task,
        key=task.idempotency_key,
        digest=getattr(task, "idempotency_digest", None),
    )


def test_tenant_segments_do_not_collide(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = FilesystemReviewReuseStore(root)
    try:
        first = _task("tenant.a")
        second = _task("tenant-a")
        _create(store, first)
        _create(store, second)

        tenant_dirs = sorted(path.name for path in root.iterdir() if path.is_dir())
        assert tenant_dirs == sorted(
            [_tenant_dir(root, "tenant.a").name, _tenant_dir(root, "tenant-a").name]
        )
        assert tenant_dirs[0] != tenant_dirs[1]
    finally:
        _close(store)


def test_dotdot_cannot_escape_store_root(tmp_path: Path) -> None:
    root = tmp_path / "container" / "store"
    store = FilesystemReviewReuseStore(root)
    task = _task("..")
    escaped_path = root.parent / "tasks" / f"{task.task_id}.json"
    try:
        with pytest.raises(Exception) as raised:
            _create(store, task)
        assert _error_code(raised.value) == "tenant_invalid"
        assert not escaped_path.exists()
    finally:
        _close(store)


def test_reads_do_not_create_tenant_paths(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = FilesystemReviewReuseStore(root)
    try:
        before = {path.name for path in root.iterdir()}
        tenant_id = "read-only-tenant"
        assert store.get(tenant_id, str(uuid.uuid4())) is None
        assert store.get_by_idempotency(tenant_id, "unused-key") is None
        assert store.list_for_tenant(tenant_id) == []
        assert {path.name for path in root.iterdir()} == before
        assert not _tenant_dir(root, tenant_id).exists()
    finally:
        _close(store)


def test_loaded_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = FilesystemReviewReuseStore(root)
    task = _task("tenant-a")
    try:
        _create(store, task)
        sidecar = _tenant_dir(root, task.tenant_id) / "tenant.json"
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        payload["tenant_id"] = "tenant-b"
        sidecar.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(Exception) as raised:
            store.get(task.tenant_id, task.task_id)
        assert _error_code(raised.value) == "store_record_corrupt"
    finally:
        _close(store)


def test_corrupt_index_and_record_fail_closed(tmp_path: Path) -> None:
    index_root = tmp_path / "index-store"
    index_store = FilesystemReviewReuseStore(index_root)
    indexed = _task(
        "tenant-index",
        idempotency_key="create-key",
        idempotency_digest="b" * 64,
    )
    try:
        _create(index_store, indexed)
        index_path = _tenant_dir(index_root, indexed.tenant_id) / "idempotency.json"
        index_path.write_text("{not-json", encoding="utf-8")
        with pytest.raises(Exception) as index_error:
            index_store.get_by_idempotency(indexed.tenant_id, "create-key")
        assert _error_code(index_error.value) == "store_index_corrupt"
    finally:
        _close(index_store)

    record_root = tmp_path / "record-store"
    record_store = FilesystemReviewReuseStore(record_root)
    record = _task("tenant-record")
    try:
        _create(record_store, record)
        record_path = (
            _tenant_dir(record_root, record.tenant_id)
            / "tasks"
            / (f"{record.task_id}.json")
        )
        record_path.write_text("{not-json", encoding="utf-8")
        with pytest.raises(Exception) as record_error:
            record_store.list_for_tenant(record.tenant_id)
        assert _error_code(record_error.value) == "store_record_corrupt"
    finally:
        _close(record_store)


def test_legacy_migration_aborts_on_collision(tmp_path: Path) -> None:
    from src.core.review_reuse import store as store_module

    root = tmp_path / "legacy-store"
    tasks_dir = root / "tenant_a" / "tasks"
    tasks_dir.mkdir(parents=True)
    first = _task("tenant/a")
    second = _task("tenant?a")
    (tasks_dir / f"{first.task_id}.json").write_text(
        json.dumps(first.model_dump(mode="json")), encoding="utf-8"
    )
    (tasks_dir / f"{second.task_id}.json").write_text(
        json.dumps(second.model_dump(mode="json")), encoding="utf-8"
    )

    migrate = getattr(store_module, "migrate_legacy_store", None)
    assert migrate is not None
    with pytest.raises(Exception) as raised:
        migrate(root, apply=True)
    assert _error_code(raised.value) == "store_record_corrupt"
    assert "collision" in str(raised.value).lower()
    assert list(root.glob("tenant-v1-*")) == []


def test_second_writer_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "store"
    first = FilesystemReviewReuseStore(root)
    try:
        with pytest.raises(Exception) as raised:
            FilesystemReviewReuseStore(root)
        assert _error_code(raised.value) == "store_writer_conflict"
    finally:
        _close(first)


def test_create_if_absent_recovers_one_task(tmp_path: Path) -> None:
    root = tmp_path / "store"
    key = "recover-key"
    digest = "c" * 64
    original = _task(
        "tenant-recovery",
        idempotency_key=key,
        idempotency_digest=digest,
    )

    first_store = FilesystemReviewReuseStore(root)
    try:
        created = _create(first_store, original)
        assert created.task_id == original.task_id
    finally:
        _close(first_store)

    index_path = _tenant_dir(root, original.tenant_id) / "idempotency.json"
    index_path.unlink()

    retry = _task(
        original.tenant_id,
        idempotency_key=key,
        idempotency_digest=digest,
    )
    second_store = FilesystemReviewReuseStore(root)
    try:
        replayed = _create(second_store, retry)
        assert replayed.task_id == original.task_id
        assert index_path.exists()
        task_files = list(
            (_tenant_dir(root, original.tenant_id) / "tasks").glob("*.json")
        )
        assert [path.stem for path in task_files] == [original.task_id]
    finally:
        _close(second_store)
