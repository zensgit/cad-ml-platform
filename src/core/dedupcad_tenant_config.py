from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.cache import delete_cache, get_cache, set_cache

DEFAULT_TENANT_CONFIG_DIR = Path("data/dedup_tenant_configs")


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TenantConfigStoreConfig:
    dir_path: Path = DEFAULT_TENANT_CONFIG_DIR
    cache_ttl_seconds: int = 3600
    file_prefix: str = "dedup2d"

    @classmethod
    def from_env(cls) -> "TenantConfigStoreConfig":
        dir_path = Path(os.getenv("DEDUPCAD_TENANT_CONFIG_DIR", str(DEFAULT_TENANT_CONFIG_DIR)))
        ttl = int(os.getenv("DEDUPCAD_TENANT_CONFIG_CACHE_TTL_SECONDS", str(cls.cache_ttl_seconds)))
        prefix = os.getenv("DEDUPCAD_TENANT_CONFIG_PREFIX", cls.file_prefix)
        return cls(dir_path=dir_path, cache_ttl_seconds=ttl, file_prefix=prefix)


class TenantDedup2DConfigStore:
    """Per-tenant (X-API-Key) config store for 2D dedup defaults.

    Storage:
      - Persistent on disk under `DEDUPCAD_TENANT_CONFIG_DIR`
      - Cached in Redis/in-memory via src.utils.cache
    """

    def __init__(self, config: Optional[TenantConfigStoreConfig] = None) -> None:
        self.config = config or TenantConfigStoreConfig.from_env()

    @staticmethod
    def tenant_id(api_key: str) -> str:
        return _sha256_hex(api_key)[:16]

    def _cache_key(self, api_key: str) -> str:
        return f"{self.config.file_prefix}:tenant_config:{self.tenant_id(api_key)}"

    def _file_path(self, api_key: str) -> Path:
        tid = self.tenant_id(api_key)
        return self.config.dir_path / f"{self.config.file_prefix}_{tid}.json"

    async def get(self, api_key: str) -> Optional[Dict[str, Any]]:
        if not api_key:
            raise ValueError("api_key is empty")
        cache_key = self._cache_key(api_key)
        cached = await get_cache(cache_key)
        if isinstance(cached, dict) and cached:
            return cached

        path = self._file_path(api_key)
        if not path.exists():
            return None
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(obj, dict):
            return None
        await set_cache(cache_key, obj, ttl_seconds=int(self.config.cache_ttl_seconds))
        return obj

    async def set(self, api_key: str, config_obj: Dict[str, Any]) -> None:
        if not api_key:
            raise ValueError("api_key is empty")
        if not isinstance(config_obj, dict):
            raise ValueError("config_obj must be a dict")
        path = self._file_path(api_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(config_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        await set_cache(
            self._cache_key(api_key),
            config_obj,
            ttl_seconds=int(self.config.cache_ttl_seconds),
        )

    async def delete(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("api_key is empty")
        path = self._file_path(api_key)
        try:
            if path.exists():
                path.unlink()
        finally:
            await delete_cache(self._cache_key(api_key))
