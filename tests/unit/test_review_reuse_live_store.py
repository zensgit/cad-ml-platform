"""Live dedup mapping + filesystem durable store tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.core.review_reuse.dedup_adapter import (
    ENV_LIVE_DEDUP,
    map_raw_hits_to_candidates,
    recall_candidates,
    set_live_recall_hook,
)
from src.core.review_reuse.dedup_live import vision_response_to_hits
from src.core.review_reuse.models import CandidateState, TaskStatus
from src.core.review_reuse.service import ReviewReuseService
from src.core.review_reuse.store import (
    ENV_STORE,
    ENV_STORE_DIR,
    FilesystemReviewReuseStore,
    InMemoryReviewReuseStore,
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
    assert hits[0]["scores"]["geometric"] == 0.95
    assert "precision-l4" in hits[0]["methods"]
    assert hits[1]["state"] == "similar"
    # Visual similarity must not be copied into geometric (strategy §3.3).
    assert hits[1]["scores"]["geometric"] is None
    assert hits[1]["scores"]["semantic"] == 0.85


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
    """Live hook must ask vision for geometric/L4 and keep that score."""
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
    assert hits[0]["scores"]["geometric"] == 0.94
    assert hits[0]["scores"]["semantic"] == 0.91
    assert "precision-l4" in hits[0]["methods"]


def test_create_task_live_geometric_not_vision_only(
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
        assert cand.scores.get("geometric") == 0.88
        assert "vision_only_unverified" not in cand.rejection_reasons
        assert "precision-l4" in (cand.verification.get("methods") or [])
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


def test_create_store_factory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(ENV_STORE, "memory")
    assert isinstance(create_review_reuse_store(), InMemoryReviewReuseStore)
    monkeypatch.setenv(ENV_STORE, "filesystem")
    monkeypatch.setenv(ENV_STORE_DIR, str(tmp_path / "fs"))
    store = create_review_reuse_store()
    assert isinstance(store, FilesystemReviewReuseStore)


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
