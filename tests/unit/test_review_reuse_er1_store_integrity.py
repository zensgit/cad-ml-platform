"""Fail-first integrity contract for the ReviewReuse ER1 store."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

import pytest

from src.core.review_reuse.canonical import canonical_sha256
from src.core.review_reuse.evidence import build_evidence_pack
from src.core.review_reuse.models import ReviewReuseTask, TaskStatus
from src.core.review_reuse.store import (
    FilesystemReviewReuseStore,
    ReviewReuseStoreError,
)


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
        status=TaskStatus.running,
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


def test_tasks_symlink_cannot_escape_store_root(tmp_path: Path) -> None:
    root = tmp_path / "container" / "store"
    store = FilesystemReviewReuseStore(root)
    task = _task("tenant-symlink")
    tenant_dir = _tenant_dir(root, task.tenant_id)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        tenant_dir.mkdir()
        (tenant_dir / "tenant.json").write_text(
            json.dumps(store._sidecar_payload(task.tenant_id)),
            encoding="utf-8",
        )
        (tenant_dir / "tasks").symlink_to(outside, target_is_directory=True)

        with pytest.raises(Exception) as raised:
            _create(store, task)
        assert _error_code(raised.value) == "store_record_corrupt"
        assert list(outside.iterdir()) == []
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
        idempotency_digest=canonical_sha256(
            {"tenant_id": "tenant-index", "source_content_sha256": "a" * 64}
        ),
    )
    try:
        _create(index_store, indexed)
        index_path = _tenant_dir(index_root, indexed.tenant_id) / "idempotency.json"
        index_path.write_text("{not-json", encoding="utf-8")
        with pytest.raises(Exception) as index_error:
            index_store.get_by_idempotency(indexed.tenant_id, "create-key")
        assert _error_code(index_error.value) == "store_index_corrupt"
        with pytest.raises(Exception) as list_error:
            index_store.list_for_tenant(indexed.tenant_id)
        assert _error_code(list_error.value) == "store_index_corrupt"
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


def test_non_i_json_is_rejected_on_write_and_read(tmp_path: Path) -> None:
    write_root = tmp_path / "write-store"
    write_store = FilesystemReviewReuseStore(write_root)
    invalid = _task("tenant-write")
    invalid.updated_at = float("nan")
    try:
        with pytest.raises(Exception) as write_error:
            _create(write_store, invalid)
        assert _error_code(write_error.value) == "store_record_corrupt"
    finally:
        _close(write_store)

    read_root = tmp_path / "read-store"
    read_store = FilesystemReviewReuseStore(read_root)
    stored = _task("tenant-read")
    try:
        _create(read_store, stored)
        record_path = (
            _tenant_dir(read_root, stored.tenant_id)
            / "tasks"
            / f"{stored.task_id}.json"
        )
        original_payload = record_path.read_text(encoding="utf-8")
        record_path.write_text(
            original_payload.replace('"updated_at":1.0', '"updated_at":NaN'),
            encoding="utf-8",
        )
        with pytest.raises(Exception) as read_error:
            read_store.get(stored.tenant_id, stored.task_id)
        assert _error_code(read_error.value) == "store_record_corrupt"

        record_path.write_text(
            original_payload.replace('"task_id":', '"task_id":"duplicate","task_id":'),
            encoding="utf-8",
        )
        with pytest.raises(Exception) as duplicate_error:
            read_store.get(stored.tenant_id, stored.task_id)
        assert _error_code(duplicate_error.value) == "store_record_corrupt"
    finally:
        _close(read_store)


def test_calibration_status_uses_closed_vocabulary(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = FilesystemReviewReuseStore(root)
    task = _task("tenant-calibration")
    try:
        _create(store, task)
        record_path = (
            _tenant_dir(root, task.tenant_id) / "tasks" / f"{task.task_id}.json"
        )
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        payload["calibration_status"] = "calibrated-ish"
        record_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(Exception) as raised:
            store.get(task.tenant_id, task.task_id)
        assert _error_code(raised.value) == "store_record_corrupt"
    finally:
        _close(store)


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


def test_writer_refuses_unmigrated_store_before_writes(tmp_path: Path) -> None:
    from src.core.review_reuse.store import migrate_legacy_store

    root = tmp_path / "legacy-store"
    task = _task("tenant-legacy")
    tasks_dir = root / task.tenant_id / "tasks"
    tasks_dir.mkdir(parents=True)
    (tasks_dir / f"{task.task_id}.json").write_text(
        json.dumps(task.model_dump(mode="json")), encoding="utf-8"
    )

    with pytest.raises(Exception) as raised:
        FilesystemReviewReuseStore(root)
    assert _error_code(raised.value) == "store_record_corrupt"
    assert list(root.glob("tenant-v1-*")) == []

    report = migrate_legacy_store(root, apply=True)
    assert report["tasks"] == 1
    migrated = FilesystemReviewReuseStore(root)
    try:
        assert migrated.get(task.tenant_id, task.task_id) is not None
    finally:
        _close(migrated)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork semantics")
def test_forked_child_close_does_not_release_parent_writer_lease(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    parent = FilesystemReviewReuseStore(root)
    child_pid = os.fork()
    if child_pid == 0:  # pragma: no cover - assertion runs in the parent
        parent.close()
        os._exit(0)

    try:
        _pid, status = os.waitpid(child_pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        with pytest.raises(Exception) as raised:
            FilesystemReviewReuseStore(root)
        assert _error_code(raised.value) == "store_writer_conflict"
    finally:
        _close(parent)


def test_create_if_absent_recovers_one_task(tmp_path: Path) -> None:
    root = tmp_path / "store"
    key = "recover-key"
    digest = canonical_sha256(
        {"tenant_id": "tenant-recovery", "source_content_sha256": "a" * 64}
    )
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


def test_restart_cleans_atomic_write_temporary_files(tmp_path: Path) -> None:
    root = tmp_path / "store"
    task = _task("tenant-temp-recovery")
    writer = FilesystemReviewReuseStore(root)
    try:
        _create(writer, task)
    finally:
        _close(writer)

    tasks_dir = _tenant_dir(root, task.tenant_id) / "tasks"
    stale = tasks_dir / f".{task.task_id}.json.deadbeef"
    stale.write_text("partial", encoding="utf-8")

    recovered = FilesystemReviewReuseStore(root)
    try:
        assert recovered.list_for_tenant(task.tenant_id)[0].task_id == task.task_id
        assert not stale.exists()
    finally:
        _close(recovered)


def test_partial_tenant_creation_does_not_brick_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    store = FilesystemReviewReuseStore(root)
    task = _task("tenant-staging")
    tenant_dir = _tenant_dir(root, task.tenant_id)
    original_write = store._atomic_write_json

    def fail_sidecar(path: Path, payload: object) -> None:
        if path.name == "tenant.json":
            raise ReviewReuseStoreError(
                "store_record_corrupt", "simulated sidecar failure"
            )
        original_write(path, payload)

    monkeypatch.setattr(store, "_atomic_write_json", fail_sidecar)
    try:
        with pytest.raises(ReviewReuseStoreError):
            _create(store, task)
        assert not tenant_dir.exists()

        monkeypatch.setattr(store, "_atomic_write_json", original_write)
        assert _create(store, task).task_id == task.task_id
    finally:
        _close(store)


def test_store_rejects_noncanonical_create_digest(tmp_path: Path) -> None:
    store = FilesystemReviewReuseStore(tmp_path / "store")
    task = _task(
        "tenant-digest",
        idempotency_key="create-key",
        idempotency_digest="f" * 64,
    )
    try:
        with pytest.raises(Exception) as raised:
            _create(store, task)
        assert _error_code(raised.value) == "store_record_corrupt"
        assert store.list_for_tenant(task.tenant_id) == []
    finally:
        _close(store)


def test_evidence_digest_tampering_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = FilesystemReviewReuseStore(root)
    task = _task("tenant-evidence")
    task.status = TaskStatus.evidence_ready
    task.evidence_pack = build_evidence_pack(task)
    try:
        _create(store, task)
        record_path = (
            _tenant_dir(root, task.tenant_id) / "tasks" / f"{task.task_id}.json"
        )
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        payload["evidence_pack"]["confidence"]["band"] = "tampered"
        record_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(Exception) as raised:
            store.get(task.tenant_id, task.task_id)
        assert _error_code(raised.value) == "store_record_corrupt"
    finally:
        _close(store)


def test_legacy_migration_dry_run_then_apply(tmp_path: Path) -> None:
    from src.core.review_reuse.store import migrate_legacy_store

    root = tmp_path / "legacy-store"
    tasks_dir = root / "tenant-a" / "tasks"
    tasks_dir.mkdir(parents=True)
    task = _task(
        "tenant-a",
        idempotency_key="legacy-key",
        idempotency_digest="d" * 64,
    )
    (tasks_dir / f"{task.task_id}.json").write_text(
        json.dumps(task.model_dump(mode="json")), encoding="utf-8"
    )
    (root / "tenant-a" / "idempotency.json").write_text(
        json.dumps({"legacy-key": task.task_id}), encoding="utf-8"
    )

    dry_run = migrate_legacy_store(root)
    assert dry_run == {
        "apply": False,
        "legacy_directories": 1,
        "tasks": 1,
        "tenants": 1,
    }
    assert list(root.glob("tenant-v1-*")) == []

    applied = migrate_legacy_store(root, apply=True)
    backup = Path(applied["backup"])
    assert backup.is_dir()
    assert (backup / "tenant-a" / "tasks" / f"{task.task_id}.json").is_file()
    reader = FilesystemReviewReuseStore(root, read_only=True)
    try:
        migrated = reader.get(task.tenant_id, task.task_id)
        assert migrated is not None
        assert migrated.idempotency_digest == canonical_sha256(
            {
                "tenant_id": task.tenant_id,
                "source_content_sha256": task.source_content_sha256,
            }
        )
        replayed = reader.get_by_idempotency(task.tenant_id, "legacy-key")
        assert replayed is not None and replayed.task_id == task.task_id
    finally:
        _close(reader)


def test_legacy_migration_holds_new_store_lease_during_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.core.review_reuse import store as store_module

    root = tmp_path / "legacy-store"
    task = _task("tenant-a")
    tasks_dir = root / task.tenant_id / "tasks"
    tasks_dir.mkdir(parents=True)
    (tasks_dir / f"{task.task_id}.json").write_text(
        json.dumps(task.model_dump(mode="json")), encoding="utf-8"
    )

    real_replace = store_module.os.replace
    observed: list[str | None] = []

    def checked_replace(source: str | Path, destination: str | Path) -> None:
        real_replace(source, destination)
        source_path = Path(source)
        if Path(destination) == root and source_path.name.startswith(
            f".{root.name}.migration-"
        ):
            competing = None
            try:
                competing = FilesystemReviewReuseStore(root)
            except Exception as exc:
                observed.append(_error_code(exc))
            else:
                observed.append(None)
            finally:
                if competing is not None:
                    _close(competing)

    monkeypatch.setattr(store_module.os, "replace", checked_replace)
    report = store_module.migrate_legacy_store(root, apply=True)

    assert Path(report["backup"]).is_dir()
    assert observed == ["store_writer_conflict"]


def test_legacy_prefix_like_tenant_is_not_mistaken_for_new_layout(
    tmp_path: Path,
) -> None:
    from src.core.review_reuse.store import migrate_legacy_store

    root = tmp_path / "legacy-store"
    tenant_id = "tenant-v1-legacy"
    task = _task(tenant_id)
    tasks_dir = root / tenant_id / "tasks"
    tasks_dir.mkdir(parents=True)
    (tasks_dir / f"{task.task_id}.json").write_text(
        json.dumps(task.model_dump(mode="json")), encoding="utf-8"
    )

    report = migrate_legacy_store(root)
    assert "already_migrated" not in report
    assert report["legacy_directories"] == 1
    assert report["tasks"] == 1


def test_legacy_path_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    from src.core.review_reuse.store import migrate_legacy_store

    root = tmp_path / "legacy-store"
    task = _task("tenant-b")
    tasks_dir = root / "tenant-a" / "tasks"
    tasks_dir.mkdir(parents=True)
    (tasks_dir / f"{task.task_id}.json").write_text(
        json.dumps(task.model_dump(mode="json")), encoding="utf-8"
    )

    with pytest.raises(Exception) as raised:
        migrate_legacy_store(root)
    assert _error_code(raised.value) == "store_record_corrupt"


def test_legacy_invalid_idempotency_key_fails_dry_run(tmp_path: Path) -> None:
    from src.core.review_reuse.store import migrate_legacy_store

    root = tmp_path / "legacy-store"
    task = _task("tenant-a")
    task.idempotency_key = " "
    task.idempotency_digest = "d" * 64
    tasks_dir = root / task.tenant_id / "tasks"
    tasks_dir.mkdir(parents=True)
    (tasks_dir / f"{task.task_id}.json").write_text(
        json.dumps(task.model_dump(mode="json")), encoding="utf-8"
    )

    with pytest.raises(Exception) as raised:
        migrate_legacy_store(root)
    assert _error_code(raised.value) == "store_record_corrupt"
