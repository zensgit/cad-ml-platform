"""Tests for listing Dedup2D jobs via Redis backend.

Regression: Phase 4 Day 6 job listing should include recently finished jobs within TTL,
not only active (pending/in_progress) jobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pytest

from src.core.dedupcad_2d_jobs import Dedup2DJobStatus
from src.core.dedupcad_2d_jobs_redis import Dedup2DRedisJobConfig, list_dedup2d_jobs_for_tenant


@dataclass
class _FakeRedisPool:
    """Minimal async stub to simulate the Redis operations used by list_dedup2d_jobs_for_tenant."""

    zrevrange_calls: List[Tuple[str, int, int]]
    zrem_calls: List[Tuple[str, str]]
    zrevrange_return: Dict[str, List[bytes]]
    hgetall_return: Dict[str, Dict[bytes, bytes]]

    async def zrevrange(self, key: str, start: int, stop: int) -> List[bytes]:
        self.zrevrange_calls.append((key, start, stop))
        return list(self.zrevrange_return.get(key, []))[start : stop + 1]

    async def hgetall(self, key: str) -> Dict[bytes, bytes]:
        return dict(self.hgetall_return.get(key, {}))

    async def zrem(self, key: str, member: str) -> int:
        self.zrem_calls.append((key, member))
        return 1


@pytest.mark.asyncio
async def test_list_includes_completed_jobs_within_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = Dedup2DRedisJobConfig(
        redis_url="redis://example.invalid:6379/0",
        key_prefix="dedup2d",
        queue_name="dedup2d:queue",
        ttl_seconds=3600,
        max_jobs=200,
        job_timeout_seconds=300,
    )

    tenant_id = "tenant_1"
    tenant_jobs_key = f"{cfg.key_prefix}:tenant:{tenant_id}:jobs"
    job_id_completed = "job_completed"
    job_id_pending = "job_pending"
    job_id_expired = "job_expired"

    def _job_key(job_id: str) -> str:
        return f"{cfg.key_prefix}:job:{job_id}"

    pool = _FakeRedisPool(
        zrevrange_calls=[],
        zrem_calls=[],
        zrevrange_return={
            tenant_jobs_key: [
                job_id_completed.encode("utf-8"),
                job_id_pending.encode("utf-8"),
                job_id_expired.encode("utf-8"),
            ],
        },
        hgetall_return={
            _job_key(job_id_completed): {
                b"job_id": job_id_completed.encode("utf-8"),
                b"tenant_id": tenant_id.encode("utf-8"),
                b"status": Dedup2DJobStatus.COMPLETED.value.encode("utf-8"),
                b"created_at": b"200.0",
                b"started_at": b"201.0",
                b"finished_at": b"202.0",
                b"error": b"",
            },
            _job_key(job_id_pending): {
                b"job_id": job_id_pending.encode("utf-8"),
                b"tenant_id": tenant_id.encode("utf-8"),
                b"status": Dedup2DJobStatus.PENDING.value.encode("utf-8"),
                b"created_at": b"100.0",
                b"started_at": b"",
                b"finished_at": b"",
                b"error": b"",
            },
            # job_id_expired simulates missing/expired job hash -> hgetall returns empty dict
        },
    )

    async def _fake_get_pool(_: Any = None) -> _FakeRedisPool:
        return pool

    monkeypatch.setattr(
        "src.core.dedupcad_2d_jobs_redis.get_dedup2d_redis_pool",
        _fake_get_pool,
    )

    jobs = await list_dedup2d_jobs_for_tenant(tenant_id, limit=10, cfg=cfg)

    assert [j.job_id for j in jobs] == [job_id_completed, job_id_pending]
    assert jobs[0].status == Dedup2DJobStatus.COMPLETED
    assert jobs[1].status == Dedup2DJobStatus.PENDING
    # Expired job should be pruned from index
    assert (tenant_jobs_key, job_id_expired) in pool.zrem_calls


@pytest.mark.asyncio
async def test_list_status_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = Dedup2DRedisJobConfig(
        redis_url="redis://example.invalid:6379/0",
        key_prefix="dedup2d",
        queue_name="dedup2d:queue",
        ttl_seconds=3600,
        max_jobs=200,
        job_timeout_seconds=300,
    )

    tenant_id = "tenant_1"
    tenant_jobs_key = f"{cfg.key_prefix}:tenant:{tenant_id}:jobs"
    job_id_completed = "job_completed"
    job_id_failed = "job_failed"

    def _job_key(job_id: str) -> str:
        return f"{cfg.key_prefix}:job:{job_id}"

    pool = _FakeRedisPool(
        zrevrange_calls=[],
        zrem_calls=[],
        zrevrange_return={
            tenant_jobs_key: [
                job_id_failed.encode("utf-8"),
                job_id_completed.encode("utf-8"),
            ],
        },
        hgetall_return={
            _job_key(job_id_failed): {
                b"job_id": job_id_failed.encode("utf-8"),
                b"tenant_id": tenant_id.encode("utf-8"),
                b"status": Dedup2DJobStatus.FAILED.value.encode("utf-8"),
                b"created_at": b"300.0",
                b"started_at": b"301.0",
                b"finished_at": b"302.0",
                b"error": b"boom",
            },
            _job_key(job_id_completed): {
                b"job_id": job_id_completed.encode("utf-8"),
                b"tenant_id": tenant_id.encode("utf-8"),
                b"status": Dedup2DJobStatus.COMPLETED.value.encode("utf-8"),
                b"created_at": b"200.0",
                b"started_at": b"201.0",
                b"finished_at": b"202.0",
                b"error": b"",
            },
        },
    )

    async def _fake_get_pool(_: Any = None) -> _FakeRedisPool:
        return pool

    monkeypatch.setattr(
        "src.core.dedupcad_2d_jobs_redis.get_dedup2d_redis_pool",
        _fake_get_pool,
    )

    jobs = await list_dedup2d_jobs_for_tenant(
        tenant_id, status=Dedup2DJobStatus.COMPLETED, limit=10, cfg=cfg
    )
    assert [j.job_id for j in jobs] == [job_id_completed]
