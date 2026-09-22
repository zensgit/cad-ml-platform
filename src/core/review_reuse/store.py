"""Tenant-isolated task stores: memory (default) and filesystem (durable)."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from .models import ReviewReuseTask

_STORE_LOCK_NAME = ".review_reuse.lock"

ENV_STORE = "REVIEW_REUSE_STORE"
ENV_STORE_DIR = "REVIEW_REUSE_STORE_DIR"
_TRUE_BACKENDS_FS = frozenset({"fs", "file", "filesystem", "disk"})
_TENANT_META = "tenant_meta.json"
_UNREADABLE = "__unreadable__"


class OccupiedTenantDirError(RuntimeError):
    """Hashed write path already belongs to a different tenant."""

    def __init__(self, tenant_id: str, path: Path, occupants: List[str]) -> None:
        self.tenant_id = tenant_id
        self.path = path
        self.occupants = occupants
        super().__init__(
            f"hashed tenant dir {path} is occupied by {occupants!r}, "
            f"refusing write for {tenant_id!r}"
        )


class StoreLockUnavailableError(RuntimeError):
    """Filesystem store cannot take an inter-process lock on this platform."""


class CorruptIdempotencyIndexError(RuntimeError):
    """Hashed idempotency.json exists but cannot be read as a mapping."""

    def __init__(self, tenant_id: str, path: Path) -> None:
        self.tenant_id = tenant_id
        self.path = path
        super().__init__(f"hashed idempotency index {path} is unreadable")


def _import_optional(name: str) -> Any:
    try:
        return __import__(name)
    except ImportError:
        return None


def _is_msvcrt_contention(exc: OSError) -> bool:
    """True only for documented msvcrt/Windows lock-busy errors."""
    if exc.errno in (
        errno.EACCES,
        getattr(errno, "EDEADLK", None),
        getattr(errno, "EDEADLOCK", None),
    ):
        return True
    # ERROR_SHARING_VIOLATION / ERROR_LOCK_VIOLATION
    return getattr(exc, "winerror", None) in (32, 33)


def _msvcrt_lock(fd: int, exclusive: bool, msvcrt: Any) -> None:
    """Byte-range lock on the dedicated store lockfile (Windows)."""
    if os.fstat(fd).st_size < 1:
        os.write(fd, b"\0")
    os.lseek(fd, 0, os.SEEK_SET)
    if exclusive:
        while True:
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return
            except OSError as exc:
                if not _is_msvcrt_contention(exc):
                    raise
                time.sleep(0.01)
                os.lseek(fd, 0, os.SEEK_SET)
    else:
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


def _flock(fd: int, exclusive: bool) -> None:
    """Inter-process lock; fail closed if fcntl and msvcrt are both missing."""
    fcntl_mod = _import_optional("fcntl")
    if fcntl_mod is not None:
        fcntl_mod.flock(
            fd, fcntl_mod.LOCK_EX if exclusive else fcntl_mod.LOCK_UN
        )
        return
    msvcrt_mod = _import_optional("msvcrt")
    if msvcrt_mod is None:
        raise StoreLockUnavailableError(
            "FilesystemReviewReuseStore requires an inter-process lock "
            "(fcntl or msvcrt); neither is available"
        )
    _msvcrt_lock(fd, exclusive, msvcrt_mod)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` via a unique tmp file in the same directory, then replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        tmp_path.replace(path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


class _StoreFileLock:
    """Reentrant thread lock plus exclusive flock for multi-worker FS stores.

    ``threading.RLock`` alone is per-process; gunicorn/uvicorn workers each
    have their own, so cancel/decision/idempotent create can clobber each
    other. flock on a store-root lockfile serializes those read-modify-write
    paths across processes (``msvcrt.locking`` on Windows). Nested methods
    (update_atomically → get/put) re-enter on the same thread without
    dropping the flock. Missing both backends fails closed.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._thread = threading.RLock()
        self._local = threading.local()
        self._fd: Optional[int] = None
        self._fd_pid: Optional[int] = None
        self._fd_guard = threading.Lock()

    def _ensure_fd(self) -> int:
        with self._fd_guard:
            pid = os.getpid()
            # Prefork workers inherit the parent's open-file description;
            # flock on that shared description is not exclusive across
            # children. Reopen when the PID changes.
            if self._fd is not None and self._fd_pid != pid:
                try:
                    os.close(self._fd)
                except OSError:
                    pass
                self._fd = None
                self._fd_pid = None
            if self._fd is None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                flags = os.O_RDWR | os.O_CREAT
                if hasattr(os, "O_CLOEXEC"):
                    flags |= os.O_CLOEXEC
                self._fd = os.open(str(self._path), flags, 0o644)
                self._fd_pid = pid
            return self._fd

    def __enter__(self) -> "_StoreFileLock":
        self._thread.acquire()
        try:
            depth = getattr(self._local, "depth", 0)
            if depth == 0:
                _flock(self._ensure_fd(), exclusive=True)
            self._local.depth = depth + 1
        except Exception:
            self._thread.release()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        depth = getattr(self._local, "depth", 1) - 1
        self._local.depth = depth
        try:
            if depth == 0 and self._fd is not None:
                _flock(self._fd, exclusive=False)
        finally:
            self._thread.release()
        return False


def tenant_dir_key(tenant_id: str) -> str:
    """Stable non-colliding directory name (sha256 prefix)."""
    return hashlib.sha256((tenant_id or "").encode("utf-8")).hexdigest()[:24]


def _legacy_safe_tenant(tenant_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in tenant_id)[
        :128
    ] or "unknown"


def read_tenant_meta_id(tenant_dir: Path) -> Optional[str]:
    """Return original tenant_id from sidecar, if present."""
    path = tenant_dir / _TENANT_META
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if isinstance(data, dict):
        tid = data.get("tenant_id")
        if isinstance(tid, str) and tid:
            return tid
    return None


class ReviewReuseStoreProtocol(Protocol):
    def put(self, task: ReviewReuseTask) -> ReviewReuseTask: ...

    def get(self, tenant_id: str, task_id: str) -> Optional[ReviewReuseTask]: ...

    def get_by_idempotency(self, tenant_id: str, key: str) -> Optional[ReviewReuseTask]: ...

    def put_new_idempotent(self, task: ReviewReuseTask) -> ReviewReuseTask: ...

    def list_for_tenant(self, tenant_id: str) -> List[ReviewReuseTask]: ...

    def update_atomically(
        self,
        tenant_id: str,
        task_id: str,
        updater: Callable[[Optional[ReviewReuseTask]], ReviewReuseTask],
    ) -> ReviewReuseTask: ...


class InMemoryReviewReuseStore:
    """Process-local store (default; not multi-process durable)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tasks: Dict[str, Dict[str, ReviewReuseTask]] = {}
        self._idem: Dict[str, Dict[str, str]] = {}

    def put(self, task: ReviewReuseTask) -> ReviewReuseTask:
        with self._lock:
            bucket = self._tasks.setdefault(task.tenant_id, {})
            bucket[task.task_id] = task.model_copy(deep=True)
            if task.idempotency_key:
                self._idem.setdefault(task.tenant_id, {})[task.idempotency_key] = (
                    task.task_id
                )
            return task

    def get(self, tenant_id: str, task_id: str) -> Optional[ReviewReuseTask]:
        with self._lock:
            found = self._tasks.get(tenant_id, {}).get(task_id)
            return found.model_copy(deep=True) if found is not None else None

    def get_by_idempotency(self, tenant_id: str, key: str) -> Optional[ReviewReuseTask]:
        with self._lock:
            tid = self._idem.get(tenant_id, {}).get(key)
            if not tid:
                return None
            found = self._tasks.get(tenant_id, {}).get(tid)
            return found.model_copy(deep=True) if found is not None else None

    def put_new_idempotent(self, task: ReviewReuseTask) -> ReviewReuseTask:
        with self._lock:
            if task.idempotency_key:
                existing = self.get_by_idempotency(task.tenant_id, task.idempotency_key)
                if existing is not None:
                    return existing
            return self.put(task)

    def update_atomically(
        self,
        tenant_id: str,
        task_id: str,
        updater: Callable[[Optional[ReviewReuseTask]], ReviewReuseTask],
    ) -> ReviewReuseTask:
        with self._lock:
            current = self.get(tenant_id, task_id)
            updated = updater(current)
            return self.put(updated)

    def list_for_tenant(self, tenant_id: str) -> List[ReviewReuseTask]:
        with self._lock:
            return [
                t.model_copy(deep=True)
                for t in self._tasks.get(tenant_id, {}).values()
            ]


class FilesystemReviewReuseStore:
    """JSON-on-disk store for restart-safe pilot single-node deployments.

    Layout::

        {root}/{sha256(tenant_id)[:24]}/tasks/{task_id}.json
        {root}/{sha256(tenant_id)[:24]}/idempotency.json
        {root}/{sha256(tenant_id)[:24]}/tenant_meta.json
        {root}/.review_reuse.lock

    Writes use a store-root flock so multi-worker processes cannot clobber
    cancel/decision/idempotent create. Unique tmp names avoid ``.tmp`` races.
    Legacy sanitized ``{root}/{safe_tenant}/`` dirs are still read (get/list)
    so existing stores keep working; new writes always use the hash dir so
    ``a/b`` and ``a_b`` cannot collide.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = _StoreFileLock(self._root / _STORE_LOCK_NAME)

    def _hashed_dir(self, tenant_id: str) -> Path:
        return self._root / tenant_dir_key(tenant_id)

    def _legacy_dir(self, tenant_id: str) -> Path:
        return self._root / _legacy_safe_tenant(tenant_id)

    def _existing_dir_tenants(self, tenant_dir: Path) -> List[str]:
        """Identities already recorded in an on-disk tenant directory."""
        found: List[str] = []
        seen = set()

        def _add(tid: str) -> None:
            if tid not in seen:
                seen.add(tid)
                found.append(tid)

        meta_path = tenant_dir / _TENANT_META
        if meta_path.is_file():
            meta = read_tenant_meta_id(tenant_dir)
            if meta:
                _add(meta)
            else:
                # Sidecar present but unreadable/invalid: fail closed.
                _add(_UNREADABLE)
        tasks_dir = tenant_dir / "tasks"
        if not tasks_dir.is_dir():
            return found
        for path in tasks_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, ValueError):
                _add(_UNREADABLE)
                continue
            if not isinstance(data, dict):
                _add(_UNREADABLE)
                continue
            tid = data.get("tenant_id")
            if isinstance(tid, str) and tid:
                _add(tid)
            else:
                _add(_UNREADABLE)
        return found

    def _ensure_write_dir(self, tenant_id: str) -> Path:
        d = self._hashed_dir(tenant_id)
        if d.exists():
            meta_path = d / _TENANT_META
            if meta_path.is_file():
                # Valid sidecar is the write fast path; do not rescan
                # every task JSON on put/lease.
                meta = read_tenant_meta_id(d)
                if not meta:
                    raise OccupiedTenantDirError(tenant_id, d, [_UNREADABLE])
                if meta != tenant_id:
                    raise OccupiedTenantDirError(tenant_id, d, [meta])
            else:
                occupants = self._existing_dir_tenants(d)
                foreign = [tid for tid in occupants if tid != tenant_id]
                if foreign:
                    raise OccupiedTenantDirError(tenant_id, d, foreign)
        (d / "tasks").mkdir(parents=True, exist_ok=True)
        self._write_meta(d, tenant_id)
        return d

    def _write_meta(self, tenant_dir: Path, tenant_id: str) -> None:
        if read_tenant_meta_id(tenant_dir) == tenant_id:
            return
        path = tenant_dir / _TENANT_META
        payload = json.dumps({"tenant_id": tenant_id}, ensure_ascii=False)
        _atomic_write_text(path, payload)

    def _read_dirs(self, tenant_id: str) -> List[Path]:
        hashed = self._hashed_dir(tenant_id)
        legacy = self._legacy_dir(tenant_id)
        out: List[Path] = []
        if hashed.is_dir():
            out.append(hashed)
        if legacy != hashed and legacy.is_dir():
            out.append(legacy)
        return out

    def _task_path(self, tenant_dir: Path, task_id: str) -> Path:
        safe_tid = "".join(c if c.isalnum() or c in "-_" else "_" for c in task_id)
        return tenant_dir / "tasks" / f"{safe_tid}.json"

    def _idem_path(self, tenant_dir: Path) -> Path:
        return tenant_dir / "idempotency.json"

    def _parse_task_file(self, path: Path) -> Optional[ReviewReuseTask]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ReviewReuseTask.model_validate(data)
        except (OSError, json.JSONDecodeError, ValueError):
            return None

    def _hashed_task_if_present(
        self, hashed: Path, tenant_id: str, task_id: str
    ) -> tuple[bool, Optional[ReviewReuseTask]]:
        """Hashed file, if present, is authoritative (including junk/mismatch)."""
        path = self._task_path(hashed, task_id)
        if not path.exists():
            return False, None
        task = self._parse_task_file(path)
        if task is None or task.tenant_id != tenant_id:
            return True, None
        return True, task

    def _try_load_idem(self, tenant_dir: Path) -> Optional[Dict[str, str]]:
        """Return mapping, empty if missing, None if present but invalid."""
        path = self._idem_path(tenant_dir)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        mapping: Dict[str, str] = {}
        for key, value in data.items():
            if not isinstance(key, str) or not key:
                return None
            if not isinstance(value, str) or not value:
                return None
            mapping[key] = value
        return mapping

    def _load_idem(self, tenant_dir: Path) -> Dict[str, str]:
        loaded = self._try_load_idem(tenant_dir)
        return loaded if loaded is not None else {}

    def _save_idem(self, tenant_dir: Path, mapping: Dict[str, str]) -> None:
        path = self._idem_path(tenant_dir)
        _atomic_write_text(
            path, json.dumps(mapping, ensure_ascii=False, indent=0)
        )

    def put(self, task: ReviewReuseTask) -> ReviewReuseTask:
        with self._lock:
            tenant_dir = self._ensure_write_dir(task.tenant_id)
            idem: Optional[Dict[str, str]] = None
            if task.idempotency_key:
                # Validate before replacing the task file so cancel/decision
                # cannot persist while the caller sees store_conflict.
                idem = self._try_load_idem(tenant_dir)
                if idem is None:
                    raise CorruptIdempotencyIndexError(
                        task.tenant_id, self._idem_path(tenant_dir)
                    )
            path = self._task_path(tenant_dir, task.task_id)
            payload = task.model_dump(mode="json")
            _atomic_write_text(
                path, json.dumps(payload, ensure_ascii=False, indent=0)
            )
            if task.idempotency_key:
                if idem is None:
                    raise CorruptIdempotencyIndexError(
                        task.tenant_id, self._idem_path(tenant_dir)
                    )
                idem[task.idempotency_key] = task.task_id
                self._save_idem(tenant_dir, idem)
            return task

    def get(self, tenant_id: str, task_id: str) -> Optional[ReviewReuseTask]:
        with self._lock:
            hashed = self._hashed_dir(tenant_id)
            present, task = self._hashed_task_if_present(hashed, tenant_id, task_id)
            if present:
                return task
            for tenant_dir in self._read_dirs(tenant_id):
                if tenant_dir == hashed:
                    continue
                path = self._task_path(tenant_dir, task_id)
                if not path.exists():
                    continue
                loaded = self._parse_task_file(path)
                if loaded is not None and loaded.tenant_id == tenant_id:
                    return loaded
            return None

    def get_by_idempotency(self, tenant_id: str, key: str) -> Optional[ReviewReuseTask]:
        with self._lock:
            hashed = self._hashed_dir(tenant_id)
            hashed_idem = self._try_load_idem(hashed)
            if hashed_idem is None:
                # Present-but-junk hashed index is corruption, not a miss.
                raise CorruptIdempotencyIndexError(
                    tenant_id, self._idem_path(hashed)
                )
            hashed_tid = hashed_idem.get(key)
            if hashed_tid:
                present, task = self._hashed_task_if_present(
                    hashed, tenant_id, hashed_tid
                )
                if present:
                    return task
            for tenant_dir in self._read_dirs(tenant_id):
                if tenant_dir == hashed:
                    continue
                tid = self._load_idem(tenant_dir).get(key)
                if not tid:
                    continue
                present, task = self._hashed_task_if_present(hashed, tenant_id, tid)
                if present:
                    return task
                path = self._task_path(tenant_dir, tid)
                if not path.exists():
                    continue
                loaded = self._parse_task_file(path)
                if loaded is not None and loaded.tenant_id == tenant_id:
                    return loaded
            return None

    def put_new_idempotent(self, task: ReviewReuseTask) -> ReviewReuseTask:
        with self._lock:
            if task.idempotency_key:
                existing = self.get_by_idempotency(task.tenant_id, task.idempotency_key)
                if existing is not None:
                    return existing
            return self.put(task)

    def update_atomically(
        self,
        tenant_id: str,
        task_id: str,
        updater: Callable[[Optional[ReviewReuseTask]], ReviewReuseTask],
    ) -> ReviewReuseTask:
        with self._lock:
            current = self.get(tenant_id, task_id)
            updated = updater(current)
            return self.put(updated)

    def list_for_tenant(self, tenant_id: str) -> List[ReviewReuseTask]:
        with self._lock:
            seen: Dict[str, ReviewReuseTask] = {}
            blocked: set[str] = set()
            hashed = self._hashed_dir(tenant_id)
            for tenant_dir in self._read_dirs(tenant_id):
                tasks_dir = tenant_dir / "tasks"
                if not tasks_dir.exists():
                    continue
                for path in tasks_dir.glob("*.json"):
                    if path.stem in blocked:
                        continue
                    task = self._parse_task_file(path)
                    if tenant_dir == hashed:
                        # Present hashed file is authoritative, including
                        # junk or a mismatched tenant_id.
                        blocked.add(path.stem)
                        if (
                            task is not None
                            and task.tenant_id == tenant_id
                            and task.task_id not in seen
                        ):
                            seen[task.task_id] = task
                        continue
                    if task is None:
                        continue
                    if task.tenant_id == tenant_id and task.task_id not in seen:
                        seen[task.task_id] = task
            return list(seen.values())


# Back-compat alias used by existing tests/service imports.
ReviewReuseStore = InMemoryReviewReuseStore


def create_review_reuse_store() -> ReviewReuseStoreProtocol:
    backend = os.getenv(ENV_STORE, "memory").strip().lower()
    if backend in _TRUE_BACKENDS_FS:
        root = os.getenv(ENV_STORE_DIR, "data/review_reuse_tasks")
        return FilesystemReviewReuseStore(root)
    return InMemoryReviewReuseStore()
