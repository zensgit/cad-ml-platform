"""Tests: batch similarity when Faiss backend is unavailable (fallback / degraded mode)."""

from __future__ import annotations

from typing import Dict, List

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from src.core import similarity
from src.core.similarity import reset_default_store
from src.main import app


def test_batch_similarity_faiss_unavailable_degraded_flag(monkeypatch: MonkeyPatch) -> None:
    original_vectors: Dict[str, List[float]] = similarity._VECTOR_STORE.copy()
    original_meta = similarity._VECTOR_META.copy()
    similarity._VECTOR_STORE.clear()
    similarity._VECTOR_META.clear()
    similarity._VECTOR_STORE.update({"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0]})
    similarity._VECTOR_META.update(
        {
            "a": {"material": "steel", "complexity": "low", "format": "stp"},
            "b": {"material": "steel", "complexity": "low", "format": "stp"},
        }
    )

    monkeypatch.setenv("VECTOR_STORE_BACKEND", "faiss")
    store = similarity.InMemoryVectorStore()
    store._fallback_from = "faiss"
    store._requested_backend = "faiss"
    store._backend = "memory"
    store._available = False
    monkeypatch.setattr(similarity, "get_vector_store", lambda *_args, **_kwargs: store)
    reset_default_store()

    try:
        payload = {"ids": ["a"], "top_k": 1}
        with TestClient(app) as client:
            response = client.post("/api/v1/vectors/similarity/batch", json=payload)
            assert response.status_code == 200
            data = response.json()
        assert data["fallback"] is True
        assert data["degraded"] is True
        assert data["successful"] == 1
        assert data["failed"] == 0
        assert data["items"][0]["status"] == "success"
    finally:
        similarity._VECTOR_STORE.clear()
        similarity._VECTOR_META.clear()
        similarity._VECTOR_STORE.update(original_vectors)
        similarity._VECTOR_META.update(original_meta)
        reset_default_store()


def test_batch_similarity_degraded_forces_fallback(monkeypatch: MonkeyPatch) -> None:
    original_vectors: Dict[str, List[float]] = similarity._VECTOR_STORE.copy()
    original_meta = similarity._VECTOR_META.copy()
    similarity._VECTOR_STORE.clear()
    similarity._VECTOR_META.clear()
    similarity._VECTOR_STORE.update({"a": [1.0, 0.0, 0.0]})
    similarity._VECTOR_META.update(
        {
            "a": {"material": "steel", "complexity": "low", "format": "stp"},
        }
    )

    class DummyStore:
        def query(self, vector, top_k=10):
            return []

    def fake_get_vector_store(*_args, **_kwargs):
        return DummyStore()

    def fake_get_degraded_mode_info():
        return {
            "degraded": True,
            "reason": "unit-test",
            "degraded_at": 0.0,
            "degraded_duration_seconds": 1.0,
            "history": [],
            "history_count": 1,
        }

    monkeypatch.setenv("VECTOR_STORE_BACKEND", "faiss")
    monkeypatch.setattr(similarity, "get_vector_store", fake_get_vector_store)
    monkeypatch.setattr(similarity, "get_degraded_mode_info", fake_get_degraded_mode_info)
    reset_default_store()

    try:
        payload = {"ids": ["a"], "top_k": 1}
        with TestClient(app) as client:
            response = client.post("/api/v1/vectors/similarity/batch", json=payload)
            assert response.status_code == 200
            data = response.json()
        assert data["fallback"] is True
        assert data["degraded"] is True
        assert data["successful"] == 1
        assert data["failed"] == 0
        assert data["items"][0]["status"] == "success"
    finally:
        similarity._VECTOR_STORE.clear()
        similarity._VECTOR_META.clear()
        similarity._VECTOR_STORE.update(original_vectors)
        similarity._VECTOR_META.update(original_meta)
        reset_default_store()
