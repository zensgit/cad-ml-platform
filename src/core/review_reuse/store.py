"""Tenant-isolated in-memory task store (MVP; not multi-process durable)."""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

from .models import ReviewReuseTask


class ReviewReuseStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        # tenant_id -> task_id -> task
        self._tasks: Dict[str, Dict[str, ReviewReuseTask]] = {}
        # tenant_id -> idempotency_key -> task_id
        self._idem: Dict[str, Dict[str, str]] = {}

    def put(self, task: ReviewReuseTask) -> ReviewReuseTask:
        with self._lock:
            bucket = self._tasks.setdefault(task.tenant_id, {})
            bucket[task.task_id] = task
            if task.idempotency_key:
                self._idem.setdefault(task.tenant_id, {})[task.idempotency_key] = task.task_id
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
