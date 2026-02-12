from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Union

from .verifier import PrecisionVerifier

_FILE_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

logger = logging.getLogger(__name__)


class GeomJsonStoreProtocol(Protocol):
    def exists(self, file_hash: str) -> bool:
        ...

    def load(self, file_hash: str) -> Optional[Dict[str, Any]]:
        ...

    def save(self, file_hash: str, geom_json: Dict[str, Any]) -> Optional[Path]:
        ...


@dataclass(frozen=True)
class GeomJsonStoreConfig:
    base_dir: Path = Path("data/dedup_geom")
    file_suffix: str = ".v2.json"

    @classmethod
    def from_env(cls) -> "GeomJsonStoreConfig":
        return cls(
            base_dir=Path(os.getenv("DEDUPCAD_GEOM_STORE_DIR", str(cls.base_dir))),
            file_suffix=os.getenv("DEDUPCAD_GEOM_STORE_SUFFIX", cls.file_suffix),
        )


class GeomJsonStore:
    """Filesystem-backed store for v2 geometry JSON keyed by `file_hash`."""

    def __init__(self, config: Optional[GeomJsonStoreConfig] = None) -> None:
        self.config = config or GeomJsonStoreConfig.from_env()
        self.config.base_dir.mkdir(parents=True, exist_ok=True)

    def path_for_hash(self, file_hash: str) -> Path:
        if not _FILE_HASH_RE.match(file_hash):
            raise ValueError("Invalid file_hash; expected 64-char lowercase hex sha256")
        return self.config.base_dir / f"{file_hash}{self.config.file_suffix}"

    def exists(self, file_hash: str) -> bool:
        return self.path_for_hash(file_hash).exists()

    def load(self, file_hash: str) -> Optional[Dict[str, Any]]:
        path = self.path_for_hash(file_hash)
        if not path.exists():
            return None
        return PrecisionVerifier.load_json_bytes(path.read_bytes())

    def save(self, file_hash: str, geom_json: Dict[str, Any]) -> Path:
        path = self.path_for_hash(file_hash)
        content = PrecisionVerifier.canonical_json_bytes(geom_json)

        tmp_fd: Optional[int] = None
        tmp_path: Optional[str] = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent), prefix=f".{file_hash}.", suffix=".tmp"
            )
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(content)
            Path(tmp_path).replace(path)
        finally:
            if tmp_path is not None and Path(tmp_path).exists():
                Path(tmp_path).unlink(missing_ok=True)
        return path


try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    redis = None  # type: ignore


@dataclass(frozen=True)
class RedisGeomJsonStoreConfig:
    redis_url: str = "redis://localhost:6379/0"
    key_prefix: str = "dedup2d"
    ttl_seconds: int = 0

    @classmethod
    def from_env(cls) -> "RedisGeomJsonStoreConfig":
        redis_url = (
            os.getenv("DEDUPCAD_GEOM_STORE_REDIS_URL")
            or os.getenv("DEDUP2D_REDIS_URL")
            or os.getenv("REDIS_URL")
            or cls.redis_url
        )
        key_prefix = (
            os.getenv("DEDUPCAD_GEOM_STORE_REDIS_KEY_PREFIX")
            or os.getenv("DEDUP2D_REDIS_KEY_PREFIX")
            or cls.key_prefix
        )
        ttl_seconds = int(os.getenv("DEDUPCAD_GEOM_STORE_REDIS_TTL_SECONDS", str(cls.ttl_seconds)))
        return cls(
            redis_url=str(redis_url).strip() or cls.redis_url,
            key_prefix=str(key_prefix).strip() or cls.key_prefix,
            ttl_seconds=max(0, ttl_seconds),
        )


class RedisGeomJsonStore:
    """Redis-backed store for v2 geometry JSON keyed by `file_hash`.

    Keys:
      - `{key_prefix}:geom:{file_hash}` -> canonical JSON bytes
    """

    def __init__(
        self,
        config: Optional[RedisGeomJsonStoreConfig] = None,
        *,
        client: Any = None,
    ) -> None:
        self.config = config or RedisGeomJsonStoreConfig.from_env()
        if client is not None:
            self.client = client
        else:
            if redis is None:
                raise RuntimeError("redis package is not available")
            self.client = redis.Redis.from_url(self.config.redis_url)

    def _key_for_hash(self, file_hash: str) -> str:
        if not _FILE_HASH_RE.match(file_hash):
            raise ValueError("Invalid file_hash; expected 64-char lowercase hex sha256")
        return f"{self.config.key_prefix}:geom:{file_hash}"

    def exists(self, file_hash: str) -> bool:
        key = self._key_for_hash(file_hash)
        return bool(int(self.client.exists(key)))

    def load(self, file_hash: str) -> Optional[Dict[str, Any]]:
        key = self._key_for_hash(file_hash)
        raw = self.client.get(key)
        if raw is None:
            return None
        if isinstance(raw, str):
            raw_bytes = raw.encode("utf-8")
        else:
            raw_bytes = bytes(raw)
        return PrecisionVerifier.load_json_bytes(raw_bytes)

    def save(self, file_hash: str, geom_json: Dict[str, Any]) -> Optional[Path]:
        key = self._key_for_hash(file_hash)
        payload = PrecisionVerifier.canonical_json_bytes(geom_json)
        ttl = int(self.config.ttl_seconds or 0)
        if ttl > 0:
            self.client.setex(key, ttl, payload)
        else:
            self.client.set(key, payload)
        return None


class HybridGeomJsonStore:
    """Hybrid store: filesystem primary + Redis fallback."""

    def __init__(
        self,
        *,
        local_store: Optional[GeomJsonStore] = None,
        redis_store: Optional[RedisGeomJsonStore] = None,
        populate_local_on_read: bool = False,
    ) -> None:
        self.local_store = local_store or GeomJsonStore()
        self.redis_store = redis_store or RedisGeomJsonStore()
        self.populate_local_on_read = bool(populate_local_on_read)

    def exists(self, file_hash: str) -> bool:
        return self.local_store.exists(file_hash) or self.redis_store.exists(file_hash)

    def load(self, file_hash: str) -> Optional[Dict[str, Any]]:
        value = self.local_store.load(file_hash)
        if value is not None:
            return value

        value = self.redis_store.load(file_hash)
        if value is None:
            return None

        if self.populate_local_on_read:
            try:
                self.local_store.save(file_hash, value)
            except Exception:
                logger.debug(
                    "geom_store_hybrid_populate_local_failed",
                    extra={"file_hash": file_hash},
                    exc_info=True,
                )
        return value

    def save(self, file_hash: str, geom_json: Dict[str, Any]) -> Optional[Path]:
        local_exc: Optional[Exception] = None
        redis_exc: Optional[Exception] = None
        local_path: Optional[Path] = None

        try:
            local_path = self.local_store.save(file_hash, geom_json)
        except Exception as e:  # noqa: BLE001 - best-effort hybrid behavior
            local_exc = e
        try:
            self.redis_store.save(file_hash, geom_json)
        except Exception as e:  # noqa: BLE001 - best-effort hybrid behavior
            redis_exc = e

        if local_exc and redis_exc:
            raise local_exc
        if local_exc:
            logger.debug(
                "geom_store_hybrid_local_save_failed",
                extra={"file_hash": file_hash, "error": str(local_exc)},
                exc_info=True,
            )
        if redis_exc:
            logger.debug(
                "geom_store_hybrid_redis_save_failed",
                extra={"file_hash": file_hash, "error": str(redis_exc)},
                exc_info=True,
            )
        return local_path


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v == "":
        return default
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    logger.warning("invalid_bool_env", extra={"name": name, "value": raw})
    return default


@lru_cache(maxsize=1)
def create_geom_store() -> GeomJsonStoreProtocol:
    backend = os.getenv("DEDUPCAD_GEOM_STORE_BACKEND", "filesystem").strip().lower() or "filesystem"
    if backend == "filesystem":
        return GeomJsonStore()
    if backend == "redis":
        return RedisGeomJsonStore()
    if backend == "hybrid":
        populate_local = _env_bool("DEDUPCAD_GEOM_STORE_HYBRID_POPULATE_LOCAL", default=False)
        return HybridGeomJsonStore(populate_local_on_read=populate_local)
    raise ValueError(f"Unknown DEDUPCAD_GEOM_STORE_BACKEND: {backend}")
