"""Tenant-isolated task stores: memory (default) and filesystem (durable)."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Protocol

from .models import ReviewReuseTask

ENV_STORE = "REVIEW_REUSE_STORE"
ENV_STORE_DIR = "REVIEW_REUSE_STORE_DIR"
_TRUE_BACKENDS_FS = frozenset({"fs", "file", "filesystem", "disk"})


class ReviewReuseStoreProtocol(Protocol):
    def put(self, task: ReviewReuseTask) -> ReviewReuseTask: ...

    def get(self, tenant_id: str, task_id: str) -> Optional[ReviewReuseTask]: ...

    def get_by_idempotency(self, tenant_id: str, key: str) -> Optional[ReviewReuseTask]: ...

    def list_for_tenant(self, tenant_id: str) -> List[ReviewReuseTask]: ...


class InMemoryReviewReuseStore:
    """Process-local store (default; not multi-process durable)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tasks: Dict[str, Dict[str, ReviewReuseTask]] = {}
        self._idem: Dict[str, Dict[str, str]] = {}

    def put(self, task: ReviewReuseTask) -> ReviewReuseTask:
        with self._lock:
            bucket = self._tasks.setdefault(task.tenant_id, {})
            bucket[task.task_id] = task
            if task.idempotency_key:
                self._idem.setdefault(task.tenant_id, {})[task.idempotency_key] = (
                    task.task_id
                )
            return task

    def get(self, tenant_id: str, task_id: str) -> Optional[ReviewReuseTask]:
        with self._lock:
            return self._tasks.get(tenant_id, {}).get(task_id)

    def get_by_idempotency(self, tenant_id: str, key: str) -> Optional[ReviewReuseTask]:
        with self._lock:
            tid = self._idem.get(tenant_id, {}).get(key)
            if not tid:
                return None
            return self._tasks.get(tenant_id, {}).get(tid)

    def list_for_tenant(self, tenant_id: str) -> List[ReviewReuseTask]:
        with self._lock:
            return list(self._tasks.get(tenant_id, {}).values())


class FilesystemReviewReuseStore:
    """JSON-on-disk store for restart-safe pilot single-node deployments.

    Layout::

        {root}/{tenant_id}/tasks/{task_id}.json
        {root}/{tenant_id}/idempotency.json
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._lock = threading.RLock()
        self._root.mkdir(parents=True, exist_ok=True)

    def _tenant_dir(self, tenant_id: str) -> Path:
        # path-safe tenant segment
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in tenant_id)[
            :128
        ] or "unknown"
        d = self._root / safe
        (d / "tasks").mkdir(parents=True, exist_ok=True)
        return d

    def _task_path(self, tenant_id: str, task_id: str) -> Path:
        safe_tid = "".join(c if c.isalnum() or c in "-_" else "_" for c in task_id)
        return self._tenant_dir(tenant_id) / "tasks" / f"{safe_tid}.json"

    def _idem_path(self, tenant_id: str) -> Path:
        return self._tenant_dir(tenant_id) / "idempotency.json"

    def _load_idem(self, tenant_id: str) -> Dict[str, str]:
        path = self._idem_path(tenant_id)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return dict(data) if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_idem(self, tenant_id: str, mapping: Dict[str, str]) -> None:
        path = self._idem_path(tenant_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(mapping, ensure_ascii=False, indent=0), encoding="utf-8")
        tmp.replace(path)

    def put(self, task: ReviewReuseTask) -> ReviewReuseTask:
        with self._lock:
            path = self._task_path(task.tenant_id, task.task_id)
            tmp = path.with_suffix(".tmp")
            payload = task.model_dump(mode="json")
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=0), encoding="utf-8"
            )
            tmp.replace(path)
            if task.idempotency_key:
                idem = self._load_idem(task.tenant_id)
                idem[task.idempotency_key] = task.task_id
                self._save_idem(task.tenant_id, idem)
            return task

    def get(self, tenant_id: str, task_id: str) -> Optional[ReviewReuseTask]:
        with self._lock:
            path = self._task_path(tenant_id, task_id)
            if not path.exists():
                return None
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return ReviewReuseTask.model_validate(data)
            except (OSError, json.JSONDecodeError, ValueError):
                return None

    def get_by_idempotency(self, tenant_id: str, key: str) -> Optional[ReviewReuseTask]:
        with self._lock:
            tid = self._load_idem(tenant_id).get(key)
            if not tid:
                return None
            return self.get(tenant_id, tid)

    def list_for_tenant(self, tenant_id: str) -> List[ReviewReuseTask]:
        with self._lock:
            tasks_dir = self._tenant_dir(tenant_id) / "tasks"
            out: List[ReviewReuseTask] = []
            if not tasks_dir.exists():
                return out
            for path in tasks_dir.glob("*.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    task = ReviewReuseTask.model_validate(data)
                    if task.tenant_id == tenant_id:
                        out.append(task)
                except (OSError, json.JSONDecodeError, ValueError):
                    continue
            return out


# Back-compat alias used by existing tests/service imports.
ReviewReuseStore = InMemoryReviewReuseStore


def create_review_reuse_store() -> ReviewReuseStoreProtocol:
    backend = os.getenv(ENV_STORE, "memory").strip().lower()
    if backend in _TRUE_BACKENDS_FS:
        root = os.getenv(ENV_STORE_DIR, "data/review_reuse_tasks")
        return FilesystemReviewReuseStore(root)
    return InMemoryReviewReuseStore()
