"""Client for dedupcad-vision (2D dedup/search service).

This module provides a small HTTP client wrapper used by API routes to call the
separate `dedupcad-vision` service (default http://localhost:58001).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import os

import httpx


@dataclass(frozen=True)
class DedupCadVisionConfig:
    base_url: str = "http://localhost:58001"
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> "DedupCadVisionConfig":
        return cls(
            base_url=os.getenv("DEDUPCAD_VISION_URL", cls.base_url).rstrip("/"),
            timeout_seconds=float(os.getenv("DEDUPCAD_VISION_TIMEOUT_SECONDS", str(cls.timeout_seconds))),
        )


class DedupCadVisionClient:
    def __init__(self, config: Optional[DedupCadVisionConfig] = None) -> None:
        self.config = config or DedupCadVisionConfig.from_env()

    async def health(self) -> Dict[str, Any]:
        async with httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=httpx.Timeout(self.config.timeout_seconds),
        ) as client:
            resp = await client.get("/health")
            resp.raise_for_status()
            return resp.json()

    async def rebuild_indexes(self) -> Dict[str, Any]:
        """Trigger a full (re)build of vision-side L1/L2 indexes."""
        async with httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=httpx.Timeout(self.config.timeout_seconds),
        ) as client:
            resp = await client.post("/api/v2/index/rebuild")
            resp.raise_for_status()
            return resp.json()

    async def search_2d(
        self,
        *,
        file_name: str,
        file_bytes: bytes,
        content_type: str,
        mode: str = "balanced",
        max_results: int = 50,
        compute_diff: bool = True,
        enable_ml: bool = False,
        enable_geometric: bool = False,
    ) -> Dict[str, Any]:
        files = {"file": (file_name, file_bytes, content_type)}
        data = {
            "mode": mode,
            "max_results": str(max_results),
            "compute_diff": "true" if compute_diff else "false",
            "enable_ml": "true" if enable_ml else "false",
            "enable_geometric": "true" if enable_geometric else "false",
        }

        async with httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=httpx.Timeout(self.config.timeout_seconds),
        ) as client:
            resp = await client.post("/api/v2/search", files=files, data=data)
            resp.raise_for_status()
            return resp.json()

    async def index_add_2d(
        self,
        *,
        file_name: str,
        file_bytes: bytes,
        content_type: str,
        user_name: str,
        upload_to_s3: bool = True,
    ) -> Dict[str, Any]:
        files = {"file": (file_name, file_bytes, content_type)}
        params = {
            "user_name": user_name,
            "upload_to_s3": "true" if upload_to_s3 else "false",
        }

        async with httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=httpx.Timeout(self.config.timeout_seconds),
        ) as client:
            resp = await client.post("/api/index/add", files=files, params=params)
            resp.raise_for_status()
            return resp.json()
