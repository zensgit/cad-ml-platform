"""Live dedup mapping + filesystem durable store tests."""

from __future__ import annotations

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
    store1.close()
    store2 = FilesystemReviewReuseStore(tmp_path / "tasks", read_only=True)
    try:
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
    finally:
        store2.close()


def test_create_store_factory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(ENV_STORE, "memory")
    assert isinstance(create_review_reuse_store(), InMemoryReviewReuseStore)
    monkeypatch.setenv(ENV_STORE, "filesystem")
    monkeypatch.setenv(ENV_STORE_DIR, str(tmp_path / "fs"))
    store = create_review_reuse_store()
    try:
        assert isinstance(store, FilesystemReviewReuseStore)
    finally:
        store.close()  # type: ignore[attr-defined]


def test_map_raw_hits_preserves_scores() -> None:
    out = map_raw_hits_to_candidates(
        [{"candidate_id": "x", "state": "different", "scores": {"geometric": 0.1}}],
        content_sha="ab",
        file_name="f.dxf",
    )
    assert out[0].state == CandidateState.different
    assert out[0].scores.get("geometric") == 0.1
