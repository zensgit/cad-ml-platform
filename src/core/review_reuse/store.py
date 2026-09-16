"""Tenant-isolated task stores: memory (default) and filesystem (durable)."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol

from .models import ReviewReuseTask

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

    Legacy sanitized ``{root}/{safe_tenant}/`` dirs are still read (get/list)
    so existing stores keep working; new writes always use the hash dir so
    ``a/b`` and ``a_b`` cannot collide.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._lock = threading.RLock()
        self._root.mkdir(parents=True, exist_ok=True)

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

        meta = read_tenant_meta_id(tenant_dir)
        if meta:
            _add(meta)
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
        occupants = self._existing_dir_tenants(d) if d.exists() else []
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
        fd, tmp_name = tempfile.mkstemp(
            dir=str(tenant_dir), prefix=".tenant_meta.", suffix=".tmp"
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            tmp_path.replace(path)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

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

    def _load_idem(self, tenant_dir: Path) -> Dict[str, str]:
        path = self._idem_path(tenant_dir)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return dict(data) if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_idem(self, tenant_dir: Path, mapping: Dict[str, str]) -> None:
        path = self._idem_path(tenant_dir)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(mapping, ensure_ascii=False, indent=0), encoding="utf-8")
        tmp.replace(path)

    def put(self, task: ReviewReuseTask) -> ReviewReuseTask:
        with self._lock:
            tenant_dir = self._ensure_write_dir(task.tenant_id)
            path = self._task_path(tenant_dir, task.task_id)
            tmp = path.with_suffix(".tmp")
            payload = task.model_dump(mode="json")
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=0), encoding="utf-8"
            )
            tmp.replace(path)
            if task.idempotency_key:
                idem = self._load_idem(tenant_dir)
                idem[task.idempotency_key] = task.task_id
                self._save_idem(tenant_dir, idem)
            return task

    def get(self, tenant_id: str, task_id: str) -> Optional[ReviewReuseTask]:
        with self._lock:
            for tenant_dir in self._read_dirs(tenant_id):
                path = self._task_path(tenant_dir, task_id)
                if not path.exists():
                    continue
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    task = ReviewReuseTask.model_validate(data)
                except (OSError, json.JSONDecodeError, ValueError):
                    continue
                if task.tenant_id == tenant_id:
                    return task
            return None

    def get_by_idempotency(self, tenant_id: str, key: str) -> Optional[ReviewReuseTask]:
        with self._lock:
            for tenant_dir in self._read_dirs(tenant_id):
                tid = self._load_idem(tenant_dir).get(key)
                if not tid:
                    continue
                path = self._task_path(tenant_dir, tid)
                if not path.exists():
                    continue
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    task = ReviewReuseTask.model_validate(data)
                except (OSError, json.JSONDecodeError, ValueError):
                    continue
                if task.tenant_id == tenant_id:
                    return task
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
            for tenant_dir in self._read_dirs(tenant_id):
                tasks_dir = tenant_dir / "tasks"
                if not tasks_dir.exists():
                    continue
                for path in tasks_dir.glob("*.json"):
                    try:
                        data = json.loads(path.read_text(encoding="utf-8"))
                        task = ReviewReuseTask.model_validate(data)
                    except (OSError, json.JSONDecodeError, ValueError):
                        continue
                    if task.tenant_id == tenant_id and task.task_id not in seen:
                        # Hashed dir is visited first; keep that copy over a
                        # stale leftover in the legacy layout.
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
