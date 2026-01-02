from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from src.core.dedupcad_precision import (
    GeomJsonStore,
    GeomJsonStoreConfig,
    HybridGeomJsonStore,
    RedisGeomJsonStore,
    RedisGeomJsonStoreConfig,
    create_geom_store,
)


class _FakeRedisClient:
    def __init__(self) -> None:
        self._data: Dict[str, bytes] = {}

    def exists(self, key: str) -> int:
        return 1 if key in self._data else 0

    def get(self, key: str) -> Optional[bytes]:
        return self._data.get(key)

    def set(self, key: str, value: bytes) -> None:
        self._data[key] = value

    def setex(self, key: str, ttl: int, value: bytes) -> None:  # noqa: ARG002
        self._data[key] = value


def _valid_file_hash(char: str = "a") -> str:
    return char * 64


def test_create_geom_store_defaults_to_filesystem(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("DEDUPCAD_GEOM_STORE_BACKEND", "filesystem")
    monkeypatch.setenv("DEDUPCAD_GEOM_STORE_DIR", str(tmp_path))
    create_geom_store.cache_clear()

    store = create_geom_store()
    assert isinstance(store, GeomJsonStore)


def test_redis_geom_store_roundtrip(monkeypatch: pytest.MonkeyPatch):
    from src.core.dedupcad_precision import store as store_mod

    fake_client = _FakeRedisClient()

    class _FakeRedisModule:
        class Redis:
            @staticmethod
            def from_url(url: str) -> Any:  # noqa: ARG004
                return fake_client

    monkeypatch.setattr(store_mod, "redis", _FakeRedisModule)

    file_hash = _valid_file_hash("b")
    geom = {"meta": {"drawing_number": "A-001"}, "layers": [{"name": "L1"}]}

    store = RedisGeomJsonStore(
        RedisGeomJsonStoreConfig(redis_url="redis://fake", key_prefix="dedup2d", ttl_seconds=0)
    )
    store.save(file_hash, geom)
    loaded = store.load(file_hash)
    assert loaded == geom


def test_hybrid_geom_store_fallback_to_redis(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from src.core.dedupcad_precision import store as store_mod

    fake_client = _FakeRedisClient()

    class _FakeRedisModule:
        class Redis:
            @staticmethod
            def from_url(url: str) -> Any:  # noqa: ARG004
                return fake_client

    monkeypatch.setattr(store_mod, "redis", _FakeRedisModule)

    file_hash = _valid_file_hash("c")
    geom = {"meta": {"drawing_number": "B-002"}, "layers": [{"name": "L2"}]}

    local_store = GeomJsonStore(GeomJsonStoreConfig(base_dir=tmp_path))

    redis_store = RedisGeomJsonStore(
        RedisGeomJsonStoreConfig(redis_url="redis://fake", key_prefix="dedup2d", ttl_seconds=0)
    )
    redis_store.save(file_hash, geom)

    hybrid = HybridGeomJsonStore(
        local_store=local_store,
        redis_store=redis_store,
        populate_local_on_read=True,
    )

    assert hybrid.load(file_hash) == geom
    assert local_store.exists(file_hash)
