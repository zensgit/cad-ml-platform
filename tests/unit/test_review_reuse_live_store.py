"""Live dedup mapping + filesystem durable store tests."""

from __future__ import annotations

import errno
import json
import multiprocessing
import os
import tempfile
import threading
import time
from pathlib import Path

import pytest

from src.core.review_reuse.dedup_adapter import (
    ENV_LIVE_DEDUP,
    map_raw_hits_to_candidates,
    recall_candidates,
    set_live_recall_hook,
)
from src.core.review_reuse.dedup_live import vision_response_to_hits
from src.core.review_reuse.precision import apply_precision
from src.core.review_reuse.models import (
    CandidateState,
    RejectionReason,
    ReviewReuseTask,
    TaskEvent,
    TaskEventType,
    TaskStatus,
)
from src.core.review_reuse.service import ReviewReuseService
from src.core.review_reuse.store import (
    ENV_STORE,
    ENV_STORE_DIR,
    FilesystemReviewReuseStore,
    InMemoryReviewReuseStore,
    StoreLockUnavailableError,
    _StoreFileLock,
    create_review_reuse_store,
    tenant_dir_key,
)


def test_vision_response_to_hits_maps_buckets() -> None:
    resp = {
        "duplicates": [
            {
                "file_hash": "abc",
                "file_name": "part.dxf",
                "similarity": 0.97,
                "visual_similarity": 0.96,
                "precision_score": 0.95,
                "verdict": "duplicate",
                "match_level": 4,
                "levels": {"l4": {"precision_score": 0.95}},
            }
        ],
        "similar": [
            {
                "file_hash": "def",
                "similarity": 0.85,
                "verdict": "similar",
                "match_level": 2,
            }
        ],
    }
    hits = vision_response_to_hits(resp)
    assert len(hits) == 2
    assert hits[0]["candidate_id"] == "abc"
    assert hits[0]["state"] == "duplicate"
    # Fused remote precision_score is not geometry-only L4.
    assert hits[0]["scores"]["geometric"] is None
    assert "precision-l4" not in hits[0]["methods"]
    assert hits[1]["state"] == "similar"
    # Visual similarity must not be copied into geometric (strategy §3.3).
    assert hits[1]["scores"]["geometric"] is None
    assert hits[1]["scores"]["semantic"] == 0.85


def test_vision_response_forwards_inline_geom_json() -> None:
    """Live matches with geom_json but no precision_score must still L4."""
    geom = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]}
        ]
    }
    hits = vision_response_to_hits(
        {
            "similar": [
                {
                    "drawing_id": "not-a-file-hash",
                    "similarity": 0.88,
                    "verdict": "similar",
                    "match_level": 2,
                    "geom_json": geom,
                }
            ]
        }
    )
    assert hits[0]["geom_json"] == geom
    assert hits[0]["scores"]["geometric"] is None
    assert "precision-l4" not in hits[0]["methods"]
    cands = map_raw_hits_to_candidates(
        hits, content_sha="ab", file_name="query.json"
    )
    assert cands[0].provenance.get("geom_json") == geom
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(geom).encode("utf-8"),
    )
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") == 1.0
    assert RejectionReason.missing_geom_json.value not in out[0].rejection_reasons
    assert RejectionReason.vision_only_unverified.value not in out[0].rejection_reasons
    # Live geometry is transient: used for L4, then stripped before persist.
    assert "geom_json" not in (out[0].provenance or {})


def test_live_fused_precision_score_does_not_skip_local_l4() -> None:
    """A high fused precision_score must not override geometry-only local L4."""
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [100.0, 0.0]},
            {"type": "TEXT", "insert": [1.0, 1.0], "text": "note"},
        ]
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [0.0, 100.0]},
            {"type": "TEXT", "insert": [1.0, 1.0], "text": "note"},
        ]
    }
    hits = vision_response_to_hits(
        {
            "similar": [
                {
                    "drawing_id": "fused-remote",
                    "similarity": 0.9,
                    "precision_score": 0.92,
                    "verdict": "similar",
                    "match_level": 4,
                    "geom_json": other,
                }
            ]
        }
    )
    assert hits[0]["scores"]["geometric"] is None
    cands = map_raw_hits_to_candidates(
        hits, content_sha="ab", file_name="query.json"
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    geo = out[0].scores.get("geometric")
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert geo is not None
    assert geo < 0.55
    assert RejectionReason.low_precision_score.value in out[0].rejection_reasons


def test_vision_boolean_precision_score_is_not_l4() -> None:
    hits = vision_response_to_hits(
        {
            "duplicates": [
                {
                    "file_hash": "bool-l4",
                    "precision_score": True,
                    "verdict": "duplicate",
                    "match_level": 4,
                }
            ]
        }
    )
    assert hits[0]["scores"]["geometric"] is None
    assert "precision-l4" not in hits[0]["methods"]
    cands = map_raw_hits_to_candidates(
        hits, content_sha="ab", file_name="q.png"
    )
    out = apply_precision(cands, file_name="q.png", file_bytes=b"x")
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None


def test_live_default_hook_path_with_inject(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_LIVE_DEDUP, "true")

    def _hook(fn: str, fb: bytes, sha: str):
        return vision_response_to_hits(
            {
                "duplicates": [
                    {
                        "file_hash": "h1",
                        "similarity": 0.99,
                        "verdict": "duplicate",
                        "match_level": 3,
                    }
                ],
                "similar": [],
            }
        )

    set_live_recall_hook(_hook)
    try:
        cands = recall_candidates(
            file_name="a.dxf", file_bytes=b"x", content_sha="0" * 64
        )
        assert len(cands) == 1
        assert cands[0].state == CandidateState.duplicate
        assert cands[0].candidate_id == "h1"
    finally:
        set_live_recall_hook(None)
        monkeypatch.delenv(ENV_LIVE_DEDUP, raising=False)


def test_default_live_recall_requests_geometric(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live hook asks vision for geom payloads but does not trust fused scores."""
    from src.core.review_reuse.dedup_live import default_live_recall

    captured: dict = {}

    class _FakeClient:
        async def search_2d(self, **kwargs):
            captured.update(kwargs)
            return {
                "duplicates": [
                    {
                        "file_hash": "h-geom",
                        "visual_similarity": 0.91,
                        "precision_score": 0.94,
                        "verdict": "duplicate",
                        "match_level": 4,
                        "levels": {"l4": {"precision_score": 0.94}},
                    }
                ],
                "similar": [],
            }

    monkeypatch.setattr(
        "src.core.dedupcad_vision.DedupCadVisionClient",
        lambda: _FakeClient(),
    )
    hits = default_live_recall("a.dxf", b"x", "0" * 64)
    assert captured.get("enable_geometric") is True
    assert captured.get("enable_ml") is False
    assert hits[0]["scores"]["geometric"] is None
    assert hits[0]["scores"]["semantic"] == 0.91
    assert "precision-l4" not in hits[0]["methods"]


def test_run_coro_applies_timeout_without_running_loop() -> None:
    import asyncio
    import time

    from src.core.review_reuse.dedup_live import _run_coro

    async def _slow() -> str:
        await asyncio.sleep(2)
        return "done"

    started = time.monotonic()
    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        _run_coro(_slow(), timeout=0.05)
    assert time.monotonic() - started < 1.5


def test_run_coro_timeout_when_loop_already_running() -> None:
    import asyncio
    import time

    from src.core.review_reuse.dedup_live import _run_coro

    async def _inner() -> None:
        async def _slow() -> str:
            await asyncio.sleep(2)
            return "done"

        started = time.monotonic()
        with pytest.raises((TimeoutError, asyncio.TimeoutError)):
            _run_coro(_slow(), timeout=0.05)
        assert time.monotonic() - started < 1.5

    asyncio.run(_inner())


def test_run_coro_timeout_runs_canceled_cleanup() -> None:
    """Nested-loop timeout must finish CancelledError cleanup before close."""
    import asyncio
    import warnings

    from src.core.review_reuse.dedup_live import _run_coro

    cleaned = threading.Event()

    async def _inner() -> None:
        async def _slow() -> str:
            try:
                await asyncio.sleep(2)
                return "done"
            finally:
                cleaned.set()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises((TimeoutError, asyncio.TimeoutError)):
                _run_coro(_slow(), timeout=0.05)
        assert cleaned.wait(timeout=1.0)
        assert not any(
            "destroyed but it is pending" in str(item.message) for item in caught
        )

    asyncio.run(_inner())


def test_run_coro_nested_success_does_not_wait_drain_timeout() -> None:
    """Nested success must not pay drain.result's 1s timeout."""
    import asyncio
    import time

    from src.core.review_reuse.dedup_live import _run_coro

    async def _inner() -> None:
        async def _fast() -> int:
            return 42

        started = time.monotonic()
        assert _run_coro(_fast(), timeout=2) == 42
        assert time.monotonic() - started < 0.5

    asyncio.run(_inner())


def test_run_coro_completes_within_timeout() -> None:
    from src.core.review_reuse.dedup_live import _run_coro

    async def _fast() -> int:
        return 42

    assert _run_coro(_fast(), timeout=1) == 42


def test_run_coro_stops_worker_when_startup_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    from src.core.review_reuse import dedup_live

    orig_event = threading.Event
    first: dict = {"ev": None}

    def _factory() -> threading.Event:
        ev = orig_event()
        if first["ev"] is None:
            first["ev"] = ev

            def _wait(timeout: float | None = None) -> bool:
                return False

            ev.wait = _wait  # type: ignore[method-assign]
        return ev

    monkeypatch.setattr(dedup_live.threading, "Event", _factory)

    async def _inner() -> None:
        async def _fast() -> int:
            return 1

        before = {
            id(t)
            for t in threading.enumerate()
            if t.name == "review-reuse-live-recall" and t.is_alive()
        }
        with pytest.raises((TimeoutError, asyncio.TimeoutError)):
            dedup_live._run_coro(_fast(), timeout=1)
        time.sleep(0.05)
        after = [
            t
            for t in threading.enumerate()
            if t.name == "review-reuse-live-recall"
            and t.is_alive()
            and id(t) not in before
        ]
        assert after == []

    asyncio.run(_inner())


def test_create_task_live_fused_precision_is_not_geometric_l4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_LIVE_DEDUP, "true")
    set_live_recall_hook(None)

    class _FakeClient:
        async def search_2d(self, **kwargs):
            return {
                "duplicates": [
                    {
                        "file_hash": "live-geom-1",
                        "precision_score": 0.88,
                        "visual_similarity": 0.7,
                        "verdict": "duplicate",
                        "match_level": 4,
                    }
                ],
                "similar": [],
            }

    monkeypatch.setattr(
        "src.core.dedupcad_vision.DedupCadVisionClient",
        lambda: _FakeClient(),
    )
    try:
        svc = ReviewReuseService(InMemoryReviewReuseStore())
        task = svc.create_task(
            tenant_id="t-live",
            file_name="a.dxf",
            file_bytes=b"x",
        )
        cand = task.candidates[0]
        assert cand.candidate_id == "live-geom-1"
        assert cand.scores.get("geometric") is None
        assert "vision_only_unverified" in cand.rejection_reasons
        assert "precision-l4" not in (cand.verification.get("methods") or [])
    finally:
        set_live_recall_hook(None)
        monkeypatch.delenv(ENV_LIVE_DEDUP, raising=False)


def test_filesystem_store_survives_reload(tmp_path: Path) -> None:
    store1 = FilesystemReviewReuseStore(tmp_path / "tasks")
    svc1 = ReviewReuseService(store1)
    task = svc1.create_task(
        tenant_id="tenant-a",
        file_name="p.dxf",
        file_bytes=b"dxf-bytes",
        idempotency_key="idem-fs-1",
        seed_candidates=[
            {
                "candidate_id": "c1",
                "state": "similar",
                "scores": {"geometric": 0.8, "semantic": 0.7},
            }
        ],
    )
    assert task.status == TaskStatus.evidence_ready

    # New store instance same root — restart simulation
    store2 = FilesystemReviewReuseStore(tmp_path / "tasks")
    loaded = store2.get("tenant-a", task.task_id)
    assert loaded is not None
    assert loaded.task_id == task.task_id
    assert loaded.source_content_sha256 == task.source_content_sha256
    assert len(loaded.candidates) == 1
    again = store2.get_by_idempotency("tenant-a", "idem-fs-1")
    assert again is not None and again.task_id == task.task_id
    assert store2.get("tenant-b", task.task_id) is None
    listed = store2.list_for_tenant("tenant-a")
    assert len(listed) == 1


def _running_task(tenant_id: str, task_id: str) -> ReviewReuseTask:
    now = time.time()
    return ReviewReuseTask(
        task_id=task_id,
        tenant_id=tenant_id,
        status=TaskStatus.running,
        created_at=now,
        updated_at=now,
        source_file_name="a.dxf",
        source_content_sha256="ab",
        trace_id="tr",
    )


def test_update_atomically_serializes_writers() -> None:
    store = InMemoryReviewReuseStore()
    store.put(_running_task("ten", "t1"))
    n = 25

    def _bump() -> None:
        def updater(current):
            assert current is not None
            current.events = list(current.events) + [
                TaskEvent(event_type=TaskEventType.recall_started, ts=time.time())
            ]
            return current

        store.update_atomically("ten", "t1", updater)

    threads = [threading.Thread(target=_bump) for _ in range(n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    found = store.get("ten", "t1")
    assert found is not None
    assert len(found.events) == n


def test_update_atomically_does_not_write_when_updater_raises() -> None:
    store = InMemoryReviewReuseStore()
    store.put(_running_task("ten", "t-raise"))

    def updater(current):
        del current
        raise RuntimeError("no-write")

    with pytest.raises(RuntimeError, match="no-write"):
        store.update_atomically("ten", "t-raise", updater)
    found = store.get("ten", "t-raise")
    assert found is not None
    assert found.status == TaskStatus.running


def _fs_lock_bump_events(root: str, n: int, result_path: str) -> None:
    from src.core.review_reuse.models import TaskEvent, TaskEventType
    from src.core.review_reuse.store import FilesystemReviewReuseStore

    store = FilesystemReviewReuseStore(root)
    for _ in range(n):
        def updater(current):
            if current is None:
                raise RuntimeError("missing task")
            current.events = list(current.events) + [
                TaskEvent(event_type=TaskEventType.recall_started, ts=time.time())
            ]
            return current

        store.update_atomically("ten", "t-lock", updater)
    Path(result_path).write_text("ok", encoding="utf-8")


def _fs_idempotent_put(root: str, task_id: str, result_path: str) -> None:
    from src.core.review_reuse.models import ReviewReuseTask, TaskStatus
    from src.core.review_reuse.store import FilesystemReviewReuseStore

    store = FilesystemReviewReuseStore(root)
    now = time.time()
    task = ReviewReuseTask(
        task_id=task_id,
        tenant_id="ten",
        status=TaskStatus.running,
        created_at=now,
        updated_at=now,
        source_file_name="a.dxf",
        source_content_sha256="ab",
        trace_id="tr",
        idempotency_key="idem-mp",
    )
    got = store.put_new_idempotent(task)
    Path(result_path).write_text(got.task_id, encoding="utf-8")


def test_store_file_lock_reopens_fd_after_pid_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prefork children must not flock the inherited file description."""
    lock = _StoreFileLock(tmp_path / ".review_reuse.lock")
    monkeypatch.setattr("src.core.review_reuse.store.os.getpid", lambda: 1001)
    lock._ensure_fd()
    assert lock._fd_pid == 1001
    monkeypatch.setattr("src.core.review_reuse.store.os.getpid", lambda: 2002)
    lock._ensure_fd()
    assert lock._fd_pid == 2002
    assert lock._fd is not None


def test_store_file_lock_fails_closed_without_fcntl_or_msvcrt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "src.core.review_reuse.store._import_optional", lambda _name: None
    )
    with pytest.raises(StoreLockUnavailableError, match="inter-process lock"):
        with _StoreFileLock(tmp_path / ".review_reuse.lock"):
            pass


def test_store_file_lock_uses_msvcrt_when_fcntl_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    class _FakeMsvcrt:
        LK_NBLCK = 1
        LK_UNLCK = 0

        @staticmethod
        def locking(_fd: int, mode: int, _nbytes: int) -> None:
            calls.append(mode)

    def _import_optional(name: str) -> object:
        if name == "fcntl":
            return None
        if name == "msvcrt":
            return _FakeMsvcrt
        raise AssertionError(name)

    monkeypatch.setattr(
        "src.core.review_reuse.store._import_optional", _import_optional
    )
    with _StoreFileLock(tmp_path / ".review_reuse.lock"):
        assert calls == [_FakeMsvcrt.LK_NBLCK]
    assert calls == [_FakeMsvcrt.LK_NBLCK, _FakeMsvcrt.LK_UNLCK]


def _patch_msvcrt_backend(
    monkeypatch: pytest.MonkeyPatch, fake: object
) -> None:
    def _import_optional(name: str) -> object:
        if name == "fcntl":
            return None
        if name == "msvcrt":
            return fake
        raise AssertionError(name)

    monkeypatch.setattr(
        "src.core.review_reuse.store._import_optional", _import_optional
    )


def test_store_file_lock_msvcrt_permanent_error_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}

    class _FakeMsvcrt:
        LK_NBLCK = 1
        LK_UNLCK = 0

        @staticmethod
        def locking(_fd: int, _mode: int, _nbytes: int) -> None:
            calls["n"] += 1
            raise OSError(errno.EINVAL, "locking not supported")

    _patch_msvcrt_backend(monkeypatch, _FakeMsvcrt)
    with pytest.raises(OSError) as raised:
        with _StoreFileLock(tmp_path / ".review_reuse.lock"):
            pass
    assert raised.value.errno == errno.EINVAL
    assert calls["n"] == 1


def test_store_file_lock_msvcrt_retries_contention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}

    class _FakeMsvcrt:
        LK_NBLCK = 1
        LK_UNLCK = 0

        @staticmethod
        def locking(_fd: int, mode: int, _nbytes: int) -> None:
            if mode == _FakeMsvcrt.LK_NBLCK:
                calls["n"] += 1
                if calls["n"] < 3:
                    raise OSError(errno.EACCES, "region already locked")

    _patch_msvcrt_backend(monkeypatch, _FakeMsvcrt)
    monkeypatch.setattr("src.core.review_reuse.store.time.sleep", lambda _s: None)
    with _StoreFileLock(tmp_path / ".review_reuse.lock"):
        assert calls["n"] == 3


def test_filesystem_update_atomically_serializes_processes(tmp_path: Path) -> None:
    root = tmp_path / "tasks"
    store = FilesystemReviewReuseStore(root)
    store.put(_running_task("ten", "t-lock"))
    n = 12
    results = [tmp_path / "r1", tmp_path / "r2"]
    ctx = multiprocessing.get_context("spawn")
    procs = [
        ctx.Process(
            target=_fs_lock_bump_events, args=(str(root), n, str(results[i]))
        )
        for i in range(2)
    ]
    for proc in procs:
        proc.start()
    try:
        for proc in procs:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
                raise AssertionError("filesystem lock worker hung")
            assert proc.exitcode == 0
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.kill()
    found = store.get("ten", "t-lock")
    assert found is not None
    assert len(found.events) == n * 2
    assert all(path.read_text(encoding="utf-8") == "ok" for path in results)


def test_filesystem_put_new_idempotent_serializes_processes(tmp_path: Path) -> None:
    root = tmp_path / "tasks"
    FilesystemReviewReuseStore(root)  # create lock root
    results = [tmp_path / "id1", tmp_path / "id2"]
    ctx = multiprocessing.get_context("spawn")
    procs = [
        ctx.Process(
            target=_fs_idempotent_put,
            args=(str(root), f"task-{i}", str(results[i])),
        )
        for i in range(2)
    ]
    for proc in procs:
        proc.start()
    try:
        for proc in procs:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
                raise AssertionError("idempotent create worker hung")
            assert proc.exitcode == 0
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.kill()
    ids = {path.read_text(encoding="utf-8") for path in results}
    assert len(ids) == 1
    store = FilesystemReviewReuseStore(root)
    listed = store.list_for_tenant("ten")
    assert len(listed) == 1
    assert listed[0].task_id in ids


def test_filesystem_atomic_write_uses_unique_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names: list[str] = []
    real_mkstemp = tempfile.mkstemp

    def _spy(*args, **kwargs):
        fd, name = real_mkstemp(*args, **kwargs)
        names.append(name)
        return fd, name

    monkeypatch.setattr("src.core.review_reuse.store.tempfile.mkstemp", _spy)
    store = FilesystemReviewReuseStore(tmp_path / "tasks")
    store.put(_running_task("ten", "t-tmp"))
    assert names
    for name in names:
        path = Path(name)
        assert path.suffix == ".tmp"
        assert not path.exists()
    leftovers = list((tmp_path / "tasks").rglob("*.tmp"))
    assert leftovers == []


def test_filesystem_update_atomically_keeps_terminal_status(tmp_path: Path) -> None:
    store = FilesystemReviewReuseStore(tmp_path / "tasks")
    store.put(_running_task("ten", "t-fs"))

    def _cancel(current):
        assert current is not None
        current.status = TaskStatus.canceled
        return current

    store.update_atomically("ten", "t-fs", _cancel)

    def _pipeline(current):
        assert current is not None
        if current.status in (TaskStatus.canceled, TaskStatus.decided):
            return current
        current.status = TaskStatus.evidence_ready
        return current

    kept = store.update_atomically("ten", "t-fs", _pipeline)
    assert kept.status == TaskStatus.canceled
    assert store.get("ten", "t-fs").status == TaskStatus.canceled


def test_create_store_factory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(ENV_STORE, "memory")
    assert isinstance(create_review_reuse_store(), InMemoryReviewReuseStore)
    monkeypatch.setenv(ENV_STORE, "filesystem")
    monkeypatch.setenv(ENV_STORE_DIR, str(tmp_path / "fs"))
    store = create_review_reuse_store()
    assert isinstance(store, FilesystemReviewReuseStore)


def test_filesystem_put_refuses_hashed_dir_occupied_by_legacy_tenant(
    tmp_path: Path,
) -> None:
    from src.core.review_reuse.service import ReviewReuseError

    root = tmp_path / "tasks"
    occupant = tenant_dir_key("pilot-tenant")
    tasks = root / occupant / "tasks"
    tasks.mkdir(parents=True)
    (tasks / "task1.json").write_text(
        json.dumps(
            {
                "task_id": "legacy-1",
                "tenant_id": occupant,
                "status": "evidence_ready",
                "created_at": 1.0,
                "updated_at": 1.0,
                "source_file_name": "a.dxf",
                "source_content_sha256": "ab",
                "trace_id": "tr",
            }
        ),
        encoding="utf-8",
    )
    (root / occupant / "tenant_meta.json").write_text(
        json.dumps({"tenant_id": occupant}), encoding="utf-8"
    )
    store = FilesystemReviewReuseStore(root)
    svc = ReviewReuseService(store)
    with pytest.raises(ReviewReuseError) as ei:
        svc.create_task(
            tenant_id="pilot-tenant",
            file_name="p.dxf",
            file_bytes=b"x",
        )
    assert ei.value.code == "store_conflict"
    assert (tasks / "task1.json").is_file()
    leftover = json.loads((tasks / "task1.json").read_text(encoding="utf-8"))
    assert leftover["tenant_id"] == occupant


def test_filesystem_put_refuses_unreadable_tenant_meta(tmp_path: Path) -> None:
    from src.core.review_reuse.service import ReviewReuseError

    root = tmp_path / "tasks"
    hashed = root / tenant_dir_key("pilot-tenant")
    hashed.mkdir(parents=True)
    meta = hashed / "tenant_meta.json"
    meta.write_text("{not-json", encoding="utf-8")
    store = FilesystemReviewReuseStore(root)
    svc = ReviewReuseService(store)
    with pytest.raises(ReviewReuseError) as ei:
        svc.create_task(
            tenant_id="pilot-tenant",
            file_name="p.dxf",
            file_bytes=b"x",
        )
    assert ei.value.code == "store_conflict"
    assert meta.read_text(encoding="utf-8") == "{not-json"


def test_cancel_translates_occupied_hashed_dir(tmp_path: Path) -> None:
    from src.core.review_reuse.service import ReviewReuseError

    root = tmp_path / "tasks"
    tenant = "pilot-tenant"
    hashed = root / tenant_dir_key(tenant)
    (hashed / "tasks").mkdir(parents=True)
    (hashed / "tasks" / "foreign.json").write_text(
        json.dumps(
            {
                "task_id": "foreign",
                "tenant_id": "other-tenant",
                "status": "evidence_ready",
                "created_at": 1.0,
                "updated_at": 1.0,
                "source_file_name": "a.dxf",
                "source_content_sha256": "ab",
                "trace_id": "tr",
            }
        ),
        encoding="utf-8",
    )
    (hashed / "tenant_meta.json").write_text(
        json.dumps({"tenant_id": "other-tenant"}), encoding="utf-8"
    )
    legacy = root / tenant
    (legacy / "tasks").mkdir(parents=True)
    task = _running_task(tenant, "legacy-1")
    (legacy / "tasks" / "legacy-1.json").write_text(
        json.dumps(task.model_dump(mode="json")), encoding="utf-8"
    )
    store = FilesystemReviewReuseStore(root)
    svc = ReviewReuseService(store)
    loaded = svc.get_task(tenant, "legacy-1")
    assert loaded.task_id == "legacy-1"
    with pytest.raises(ReviewReuseError) as ei:
        svc.cancel(tenant, "legacy-1")
    assert ei.value.code == "store_conflict"
    leftover = json.loads((legacy / "tasks" / "legacy-1.json").read_text(encoding="utf-8"))
    assert leftover["status"] == "running"


def test_filesystem_store_tenant_path_no_collision(tmp_path: Path) -> None:
    """Sanitized names a/b and a_b must not share a directory."""
    store = FilesystemReviewReuseStore(tmp_path / "tasks")
    svc = ReviewReuseService(store)
    seed = [
        {
            "candidate_id": "c1",
            "state": "similar",
            "scores": {"geometric": 0.8, "semantic": 0.7},
            "methods": ["precision-l4"],
        }
    ]
    t_slash = svc.create_task(
        tenant_id="a/b",
        file_name="p.dxf",
        file_bytes=b"one",
        seed_candidates=seed,
    )
    t_under = svc.create_task(
        tenant_id="a_b",
        file_name="p.dxf",
        file_bytes=b"two",
        seed_candidates=seed,
    )
    assert t_slash.task_id != t_under.task_id
    assert store.get("a/b", t_slash.task_id) is not None
    assert store.get("a_b", t_slash.task_id) is None
    assert store.get("a/b", t_under.task_id) is None
    assert store.get("a_b", t_under.task_id) is not None
    assert store.get("a/b", t_slash.task_id).tenant_id == "a/b"
    assert len(store.list_for_tenant("a/b")) == 1
    assert len(store.list_for_tenant("a_b")) == 1


def test_list_for_tenant_prefers_hashed_over_stale_legacy(tmp_path: Path) -> None:
    """put() writes hashed; leftover legacy JSON must not win the listing."""
    root = tmp_path / "tasks"
    store = FilesystemReviewReuseStore(root)
    svc = ReviewReuseService(store)
    task = svc.create_task(
        tenant_id="pilot-tenant",
        file_name="p.dxf",
        file_bytes=b"dxf-bytes",
        seed_candidates=[
            {
                "candidate_id": "c1",
                "state": "similar",
                "scores": {"geometric": 0.8, "semantic": 0.7},
                "methods": ["precision-l4"],
            }
        ],
    )
    hashed_path = (
        root / tenant_dir_key("pilot-tenant") / "tasks" / f"{task.task_id}.json"
    )
    legacy_dir = root / "pilot-tenant" / "tasks"
    legacy_dir.mkdir(parents=True)
    legacy_path = legacy_dir / f"{task.task_id}.json"
    legacy_path.write_text(hashed_path.read_text(encoding="utf-8"), encoding="utf-8")

    canceled = task.model_copy(update={"status": TaskStatus.canceled})
    store.put(canceled)

    listed = store.list_for_tenant("pilot-tenant")
    assert len(listed) == 1
    assert listed[0].status == TaskStatus.canceled
    stale = json.loads(legacy_path.read_text(encoding="utf-8"))
    assert stale["status"] == TaskStatus.evidence_ready.value


def test_corrupt_hashed_task_does_not_resurrect_legacy(tmp_path: Path) -> None:
    """Hashed JSON that exists but is junk must not fall through to leftover legacy."""
    root = tmp_path / "tasks"
    store = FilesystemReviewReuseStore(root)
    canceled = store.put(
        ReviewReuseTask(
            task_id="t-corrupt",
            tenant_id="pilot-tenant",
            status=TaskStatus.canceled,
            created_at=1.0,
            updated_at=2.0,
            source_file_name="a.dxf",
            source_content_sha256="ab",
            idempotency_key="idem-corrupt",
            trace_id="tr",
        )
    )
    other = store.put(
        ReviewReuseTask(
            task_id="t-ok",
            tenant_id="pilot-tenant",
            status=TaskStatus.evidence_ready,
            created_at=1.0,
            updated_at=2.0,
            source_file_name="b.dxf",
            source_content_sha256="cd",
            trace_id="tr",
        )
    )
    hashed_corrupt = (
        root / tenant_dir_key("pilot-tenant") / "tasks" / f"{canceled.task_id}.json"
    )
    legacy_dir = root / "pilot-tenant" / "tasks"
    legacy_dir.mkdir(parents=True)
    running = canceled.model_copy(update={"status": TaskStatus.running})
    (legacy_dir / f"{canceled.task_id}.json").write_text(
        json.dumps(running.model_dump(mode="json")), encoding="utf-8"
    )
    only_legacy = ReviewReuseTask(
        task_id="t-legacy-only",
        tenant_id="pilot-tenant",
        status=TaskStatus.evidence_ready,
        created_at=1.0,
        updated_at=2.0,
        source_file_name="c.dxf",
        source_content_sha256="ef",
        trace_id="tr",
    )
    (legacy_dir / f"{only_legacy.task_id}.json").write_text(
        json.dumps(only_legacy.model_dump(mode="json")), encoding="utf-8"
    )
    hashed_corrupt.write_text("{not-json", encoding="utf-8")

    assert store.get("pilot-tenant", canceled.task_id) is None
    assert store.get_by_idempotency("pilot-tenant", "idem-corrupt") is None
    assert store.get("pilot-tenant", other.task_id) is not None
    assert store.get("pilot-tenant", only_legacy.task_id) is not None
    listed = {task.task_id: task.status for task in store.list_for_tenant("pilot-tenant")}
    assert canceled.task_id not in listed
    assert listed[other.task_id] == TaskStatus.evidence_ready
    assert listed[only_legacy.task_id] == TaskStatus.evidence_ready


def test_hashed_tenant_mismatch_does_not_resurrect_legacy(tmp_path: Path) -> None:
    """A present hashed file with the wrong tenant_id must not fall through."""
    root = tmp_path / "tasks"
    store = FilesystemReviewReuseStore(root)
    canceled = store.put(
        ReviewReuseTask(
            task_id="t-mismatch",
            tenant_id="pilot-tenant",
            status=TaskStatus.canceled,
            created_at=1.0,
            updated_at=2.0,
            source_file_name="a.dxf",
            source_content_sha256="ab",
            trace_id="tr",
        )
    )
    hashed_path = (
        root / tenant_dir_key("pilot-tenant") / "tasks" / f"{canceled.task_id}.json"
    )
    payload = json.loads(hashed_path.read_text(encoding="utf-8"))
    payload["tenant_id"] = "other-tenant"
    hashed_path.write_text(json.dumps(payload), encoding="utf-8")
    legacy_dir = root / "pilot-tenant" / "tasks"
    legacy_dir.mkdir(parents=True)
    running = canceled.model_copy(update={"status": TaskStatus.running})
    (legacy_dir / f"{canceled.task_id}.json").write_text(
        json.dumps(running.model_dump(mode="json")), encoding="utf-8"
    )

    assert store.get("pilot-tenant", canceled.task_id) is None
    listed = store.list_for_tenant("pilot-tenant")
    assert all(task.task_id != canceled.task_id for task in listed)


def test_legacy_idempotency_does_not_skip_hashed_task(tmp_path: Path) -> None:
    """Stale leftover mapping must not win if the hashed task file exists."""
    root = tmp_path / "tasks"
    store = FilesystemReviewReuseStore(root)
    canceled = store.put(
        ReviewReuseTask(
            task_id="t-idem-stale",
            tenant_id="pilot-tenant",
            status=TaskStatus.canceled,
            created_at=1.0,
            updated_at=2.0,
            source_file_name="a.dxf",
            source_content_sha256="ab",
            idempotency_key="idem-stale-map",
            trace_id="tr",
        )
    )
    hashed_dir = root / tenant_dir_key("pilot-tenant")
    (hashed_dir / "idempotency.json").write_text("{not-json", encoding="utf-8")
    legacy_dir = root / "pilot-tenant"
    (legacy_dir / "tasks").mkdir(parents=True)
    running = canceled.model_copy(update={"status": TaskStatus.running})
    (legacy_dir / "tasks" / f"{canceled.task_id}.json").write_text(
        json.dumps(running.model_dump(mode="json")), encoding="utf-8"
    )
    (legacy_dir / "idempotency.json").write_text(
        json.dumps({"idem-stale-map": canceled.task_id}), encoding="utf-8"
    )

    loaded = store.get_by_idempotency("pilot-tenant", "idem-stale-map")
    assert loaded is not None
    assert loaded.status == TaskStatus.canceled


def test_filesystem_put_skips_task_scan_when_tenant_meta_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Valid tenant_meta must not glob/parse every task JSON on put."""
    root = tmp_path / "tasks"
    store = FilesystemReviewReuseStore(root)
    first = store.put(_running_task("pilot-tenant", "t1"))
    scans = {"n": 0}
    orig = store._existing_dir_tenants

    def _count(tenant_dir: Path) -> list:
        scans["n"] += 1
        return orig(tenant_dir)

    monkeypatch.setattr(store, "_existing_dir_tenants", _count)
    store.put(first.model_copy(update={"status": TaskStatus.canceled}))
    assert scans["n"] == 0


def test_filesystem_put_scans_tasks_when_tenant_meta_missing(tmp_path: Path) -> None:
    from src.core.review_reuse.service import ReviewReuseError

    root = tmp_path / "tasks"
    hashed = root / tenant_dir_key("pilot-tenant")
    tasks = hashed / "tasks"
    tasks.mkdir(parents=True)
    (tasks / "foreign.json").write_text(
        json.dumps(
            {
                "task_id": "foreign",
                "tenant_id": "other-tenant",
                "status": "evidence_ready",
                "created_at": 1.0,
                "updated_at": 1.0,
                "source_file_name": "a.dxf",
                "source_content_sha256": "ab",
                "trace_id": "tr",
            }
        ),
        encoding="utf-8",
    )
    store = FilesystemReviewReuseStore(root)
    svc = ReviewReuseService(store)
    with pytest.raises(ReviewReuseError) as ei:
        svc.create_task(
            tenant_id="pilot-tenant",
            file_name="p.dxf",
            file_bytes=b"x",
        )
    assert ei.value.code == "store_conflict"


def test_filesystem_store_skips_rewriting_valid_tenant_meta(tmp_path: Path) -> None:
    root = tmp_path / "tasks"
    store = FilesystemReviewReuseStore(root)
    svc = ReviewReuseService(store)
    seed = [
        {
            "candidate_id": "c1",
            "state": "similar",
            "scores": {"geometric": 0.8, "semantic": 0.7},
            "methods": ["precision-l4"],
        }
    ]
    first = svc.create_task(
        tenant_id="pilot-tenant",
        file_name="p.dxf",
        file_bytes=b"dxf-bytes",
        seed_candidates=seed,
    )
    meta = root / tenant_dir_key("pilot-tenant") / "tenant_meta.json"
    first_mtime = meta.stat().st_mtime
    canceled = first.model_copy(update={"status": TaskStatus.canceled})
    store.put(canceled)
    assert json.loads(meta.read_text(encoding="utf-8"))["tenant_id"] == "pilot-tenant"
    assert meta.stat().st_mtime == first_mtime


def test_filesystem_get_rejects_mismatched_tenant_payload(tmp_path: Path) -> None:
    store = FilesystemReviewReuseStore(tmp_path / "tasks")
    svc = ReviewReuseService(store)
    task = svc.create_task(
        tenant_id="tenant-a",
        file_name="p.dxf",
        file_bytes=b"dxf-bytes",
        seed_candidates=[
            {
                "candidate_id": "c1",
                "state": "similar",
                "scores": {"geometric": 0.8, "semantic": 0.7},
                "methods": ["precision-l4"],
            }
        ],
    )
    # Poison the JSON tenant_id; get() must not return it for tenant-a.
    from src.core.review_reuse.store import tenant_dir_key

    path = (
        tmp_path
        / "tasks"
        / tenant_dir_key("tenant-a")
        / "tasks"
        / f"{task.task_id}.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    data["tenant_id"] = "tenant-b"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert store.get("tenant-a", task.task_id) is None


def test_map_raw_hits_preserves_scores() -> None:
    out = map_raw_hits_to_candidates(
        [{"candidate_id": "x", "state": "different", "scores": {"geometric": 0.1}}],
        content_sha="ab",
        file_name="f.dxf",
    )
    assert out[0].state == CandidateState.different
    assert out[0].scores.get("geometric") == 0.1
