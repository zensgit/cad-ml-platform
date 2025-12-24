from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

logger = logging.getLogger(__name__)

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


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


def _sanitize_file_name(name: str) -> str:
    base = Path(str(name or "")).name.strip() or "upload.bin"
    sanitized = _SAFE_NAME_RE.sub("_", base).strip("._-") or "upload.bin"
    return sanitized[:200]


@dataclass(frozen=True)
class Dedup2DFileRef:
    """Reference to an uploaded file, stored outside Redis."""

    backend: str  # local|s3
    path: Optional[str] = None  # local relative path (under DEDUP2D_FILE_STORAGE_DIR)
    bucket: Optional[str] = None  # s3 bucket
    key: Optional[str] = None  # s3 object key

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"backend": self.backend}
        if self.path:
            out["path"] = self.path
        if self.bucket:
            out["bucket"] = self.bucket
        if self.key:
            out["key"] = self.key
        return out

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Dedup2DFileRef":
        if not isinstance(raw, dict):
            raise ValueError("file_ref must be a dict")
        backend = str(raw.get("backend") or "").strip().lower()
        if backend not in {"local", "s3"}:
            raise ValueError(f"Unknown file_ref backend: {backend}")
        path = str(raw.get("path") or "").strip() or None
        bucket = str(raw.get("bucket") or "").strip() or None
        key = str(raw.get("key") or "").strip() or None
        if backend == "local" and not path:
            raise ValueError("local file_ref requires path")
        if backend == "s3" and (not bucket or not key):
            raise ValueError("s3 file_ref requires bucket and key")
        return cls(backend=backend, path=path, bucket=bucket, key=key)


@dataclass(frozen=True)
class Dedup2DFileStorageConfig:
    backend: str  # local|s3
    local_dir: Path
    cleanup_on_finish: bool
    retention_seconds: int  # 0 = no auto-GC, >0 = files older than this are eligible for GC
    # S3/MinIO
    s3_bucket: str
    s3_prefix: str
    s3_endpoint: Optional[str]
    s3_region: Optional[str]

    @classmethod
    def from_env(cls) -> "Dedup2DFileStorageConfig":
        backend = os.getenv("DEDUP2D_FILE_STORAGE", "local").strip().lower() or "local"
        local_dir = Path(os.getenv("DEDUP2D_FILE_STORAGE_DIR", "data/dedup2d_uploads"))
        cleanup_default = backend == "local"
        cleanup_on_finish = _env_bool(
            "DEDUP2D_FILE_STORAGE_CLEANUP_ON_FINISH",
            default=cleanup_default,
        )
        # Retention for GC (0 = disabled, default 1 hour = 3600)
        retention_raw = os.getenv("DEDUP2D_FILE_STORAGE_RETENTION_SECONDS", "3600").strip()
        try:
            retention_seconds = max(0, int(retention_raw))
        except ValueError:
            retention_seconds = 3600
        s3_bucket = str(os.getenv("DEDUP2D_S3_BUCKET", "")).strip()
        s3_prefix = str(os.getenv("DEDUP2D_S3_PREFIX", "dedup2d/uploads")).strip().strip("/")
        s3_endpoint = str(os.getenv("DEDUP2D_S3_ENDPOINT", "")).strip() or None
        s3_region = str(os.getenv("DEDUP2D_S3_REGION", "")).strip() or None
        return cls(
            backend=backend,
            local_dir=local_dir,
            cleanup_on_finish=bool(cleanup_on_finish),
            retention_seconds=retention_seconds,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            s3_endpoint=s3_endpoint,
            s3_region=s3_region,
        )


class Dedup2DFileStorageProtocol(Protocol):
    config: Dedup2DFileStorageConfig

    async def save_bytes(
        self,
        *,
        job_id: str,
        file_name: str,
        content_type: str,
        data: bytes,
    ) -> Dedup2DFileRef:
        ...

    async def load_bytes(self, file_ref: Dedup2DFileRef) -> bytes:
        ...

    async def delete(self, file_ref: Dedup2DFileRef) -> None:
        ...


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd: Optional[int] = None
    tmp_path: Optional[str] = None
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(data)
        Path(tmp_path).replace(path)
    finally:
        if tmp_path is not None:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


class LocalDedup2DFileStorage:
    def __init__(self, config: Optional[Dedup2DFileStorageConfig] = None) -> None:
        self.config = config or Dedup2DFileStorageConfig.from_env()
        self.base_dir = Path(self.config.local_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _abs_path(self, relative_path: str) -> Path:
        rel = Path(str(relative_path))
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Invalid local file_ref path")
        base = self.base_dir.resolve()
        full = (self.base_dir / rel).resolve()
        try:
            full.relative_to(base)
        except Exception as e:
            raise ValueError("Invalid local file_ref path") from e
        return full

    async def save_bytes(
        self,
        *,
        job_id: str,
        file_name: str,
        content_type: str,  # noqa: ARG002
        data: bytes,
    ) -> Dedup2DFileRef:
        safe_name = _sanitize_file_name(file_name)
        key = f"{job_id}/{uuid.uuid4().hex}_{safe_name}"
        path = self._abs_path(key)
        await asyncio.to_thread(_write_bytes_atomic, path, data)
        return Dedup2DFileRef(backend="local", path=key)

    async def load_bytes(self, file_ref: Dedup2DFileRef) -> bytes:
        if file_ref.backend != "local" or not file_ref.path:
            raise ValueError("file_ref backend mismatch (expected local)")
        path = self._abs_path(file_ref.path)
        return await asyncio.to_thread(path.read_bytes)

    async def delete(self, file_ref: Dedup2DFileRef) -> None:
        if file_ref.backend != "local" or not file_ref.path:
            return
        try:
            path = self._abs_path(file_ref.path)
        except Exception:
            return
        try:
            await asyncio.to_thread(path.unlink, True)
        except TypeError:
            # Python <3.8 compatibility; should not happen here but keep safe.
            await asyncio.to_thread(path.unlink)
        except FileNotFoundError:
            return


try:
    import boto3  # type: ignore
except Exception:  # pragma: no cover - optional
    boto3 = None  # type: ignore


class S3Dedup2DFileStorage:
    def __init__(self, config: Optional[Dedup2DFileStorageConfig] = None) -> None:
        self.config = config or Dedup2DFileStorageConfig.from_env()
        if boto3 is None:
            raise RuntimeError("boto3 is required for DEDUP2D_FILE_STORAGE=s3")
        if not self.config.s3_bucket:
            raise ValueError("DEDUP2D_S3_BUCKET is required for DEDUP2D_FILE_STORAGE=s3")
        self.bucket = self.config.s3_bucket
        self.prefix = self.config.s3_prefix.strip().strip("/")
        self.client = boto3.client(
            "s3",
            endpoint_url=self.config.s3_endpoint,
            region_name=self.config.s3_region,
        )

    async def save_bytes(
        self,
        *,
        job_id: str,
        file_name: str,
        content_type: str,
        data: bytes,
    ) -> Dedup2DFileRef:
        safe_name = _sanitize_file_name(file_name)
        key = f"{self.prefix}/{job_id}/{uuid.uuid4().hex}_{safe_name}"

        def _put() -> None:
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                ContentType=content_type or "application/octet-stream",
            )

        await asyncio.to_thread(_put)
        return Dedup2DFileRef(backend="s3", bucket=self.bucket, key=key)

    async def load_bytes(self, file_ref: Dedup2DFileRef) -> bytes:
        if file_ref.backend != "s3" or not file_ref.bucket or not file_ref.key:
            raise ValueError("file_ref backend mismatch (expected s3)")

        def _get() -> bytes:
            resp = self.client.get_object(Bucket=file_ref.bucket, Key=file_ref.key)
            body = resp.get("Body")
            return body.read() if body is not None else b""

        return await asyncio.to_thread(_get)

    async def delete(self, file_ref: Dedup2DFileRef) -> None:
        if file_ref.backend != "s3" or not file_ref.bucket or not file_ref.key:
            return

        def _del() -> None:
            self.client.delete_object(Bucket=file_ref.bucket, Key=file_ref.key)

        try:
            await asyncio.to_thread(_del)
        except Exception:
            logger.debug(
                "dedup2d_file_storage_delete_failed",
                extra={"bucket": file_ref.bucket, "key": file_ref.key},
                exc_info=True,
            )


def create_dedup2d_file_storage(
    config: Optional[Dedup2DFileStorageConfig] = None,
) -> Dedup2DFileStorageProtocol:
    cfg = config or Dedup2DFileStorageConfig.from_env()
    backend = (cfg.backend or "").strip().lower()
    if backend == "local":
        return LocalDedup2DFileStorage(cfg)
    if backend == "s3":
        return S3Dedup2DFileStorage(cfg)
    raise ValueError(f"Unknown DEDUP2D_FILE_STORAGE: {cfg.backend}")


__all__ = [
    "Dedup2DFileRef",
    "Dedup2DFileStorageConfig",
    "Dedup2DFileStorageProtocol",
    "LocalDedup2DFileStorage",
    "S3Dedup2DFileStorage",
    "create_dedup2d_file_storage",
]
