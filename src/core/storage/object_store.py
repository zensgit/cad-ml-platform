"""S3-Compatible File Storage.

Provides:
- S3-compatible object storage interface
- Support for AWS S3, MinIO, and local filesystem
- Presigned URL generation
- Multipart upload support
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import mimetypes
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, BinaryIO, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class StorageBackend(str, Enum):
    """Storage backend types."""
    S3 = "s3"
    MINIO = "minio"
    LOCAL = "local"
    MEMORY = "memory"


@dataclass
class StorageConfig:
    """Storage configuration."""
    backend: StorageBackend = StorageBackend.LOCAL
    bucket: str = "default"
    # S3/MinIO settings
    endpoint_url: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    region: str = "us-east-1"
    use_ssl: bool = True
    # Local settings
    base_path: str = "/tmp/storage"
    # General settings
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: Optional[List[str]] = None


@dataclass
class ObjectMetadata:
    """Metadata for a stored object."""
    key: str
    size: int
    content_type: str
    etag: Optional[str] = None
    last_modified: Optional[datetime] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class MultipartUpload:
    """Multipart upload state."""
    upload_id: str
    key: str
    bucket: str
    parts: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


class StorageClient(ABC):
    """Abstract base class for storage clients."""

    @abstractmethod
    async def put_object(
        self,
        key: str,
        data: Union[bytes, BinaryIO],
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> ObjectMetadata:
        """Upload an object."""
        pass

    @abstractmethod
    async def get_object(self, key: str) -> bytes:
        """Download an object."""
        pass

    @abstractmethod
    async def delete_object(self, key: str) -> bool:
        """Delete an object."""
        pass

    @abstractmethod
    async def head_object(self, key: str) -> Optional[ObjectMetadata]:
        """Get object metadata without downloading."""
        pass

    @abstractmethod
    async def list_objects(
        self,
        prefix: str = "",
        max_keys: int = 1000,
        continuation_token: Optional[str] = None,
    ) -> tuple[List[ObjectMetadata], Optional[str]]:
        """List objects with optional prefix."""
        pass

    @abstractmethod
    async def generate_presigned_url(
        self,
        key: str,
        expires_in: int = 3600,
        method: str = "GET",
    ) -> str:
        """Generate a presigned URL for object access."""
        pass

    @abstractmethod
    async def copy_object(self, source_key: str, dest_key: str) -> ObjectMetadata:
        """Copy an object."""
        pass


class InMemoryStorage(StorageClient):
    """In-memory storage for testing."""

    def __init__(self, config: StorageConfig):
        self.config = config
        self._objects: Dict[str, bytes] = {}
        self._metadata: Dict[str, ObjectMetadata] = {}
        self._multipart_uploads: Dict[str, MultipartUpload] = {}

    async def put_object(
        self,
        key: str,
        data: Union[bytes, BinaryIO],
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> ObjectMetadata:
        if isinstance(data, (io.IOBase, BinaryIO)):
            data = data.read()

        if len(data) > self.config.max_file_size:
            raise ValueError(f"File size exceeds maximum: {self.config.max_file_size}")

        etag = hashlib.md5(data).hexdigest()
        content_type = content_type or mimetypes.guess_type(key)[0] or "application/octet-stream"

        obj_meta = ObjectMetadata(
            key=key,
            size=len(data),
            content_type=content_type,
            etag=etag,
            last_modified=datetime.utcnow(),
            metadata=metadata or {},
        )

        self._objects[key] = data
        self._metadata[key] = obj_meta
        return obj_meta

    async def get_object(self, key: str) -> bytes:
        if key not in self._objects:
            raise FileNotFoundError(f"Object not found: {key}")
        return self._objects[key]

    async def delete_object(self, key: str) -> bool:
        if key in self._objects:
            del self._objects[key]
            del self._metadata[key]
            return True
        return False

    async def head_object(self, key: str) -> Optional[ObjectMetadata]:
        return self._metadata.get(key)

    async def list_objects(
        self,
        prefix: str = "",
        max_keys: int = 1000,
        continuation_token: Optional[str] = None,
    ) -> tuple[List[ObjectMetadata], Optional[str]]:
        matching = [
            meta for key, meta in self._metadata.items()
            if key.startswith(prefix)
        ]
        matching.sort(key=lambda m: m.key)

        start_index = 0
        if continuation_token:
            try:
                start_index = int(continuation_token)
            except ValueError:
                pass

        end_index = start_index + max_keys
        result = matching[start_index:end_index]
        next_token = str(end_index) if end_index < len(matching) else None

        return result, next_token

    async def generate_presigned_url(
        self,
        key: str,
        expires_in: int = 3600,
        method: str = "GET",
    ) -> str:
        # For in-memory storage, return a mock URL
        return f"memory://{self.config.bucket}/{key}?expires={expires_in}&method={method}"

    async def copy_object(self, source_key: str, dest_key: str) -> ObjectMetadata:
        if source_key not in self._objects:
            raise FileNotFoundError(f"Source object not found: {source_key}")

        data = self._objects[source_key]
        source_meta = self._metadata[source_key]

        return await self.put_object(
            dest_key,
            data,
            source_meta.content_type,
            source_meta.metadata,
        )

    # Multipart upload support
    async def create_multipart_upload(self, key: str) -> str:
        upload_id = str(uuid.uuid4())
        self._multipart_uploads[upload_id] = MultipartUpload(
            upload_id=upload_id,
            key=key,
            bucket=self.config.bucket,
        )
        return upload_id

    async def upload_part(
        self,
        key: str,
        upload_id: str,
        part_number: int,
        data: bytes,
    ) -> Dict[str, Any]:
        if upload_id not in self._multipart_uploads:
            raise ValueError(f"Upload not found: {upload_id}")

        etag = hashlib.md5(data).hexdigest()
        part = {
            "part_number": part_number,
            "etag": etag,
            "data": data,
        }
        self._multipart_uploads[upload_id].parts.append(part)

        return {"etag": etag, "part_number": part_number}

    async def complete_multipart_upload(
        self,
        key: str,
        upload_id: str,
        parts: List[Dict[str, Any]],
    ) -> ObjectMetadata:
        if upload_id not in self._multipart_uploads:
            raise ValueError(f"Upload not found: {upload_id}")

        upload = self._multipart_uploads[upload_id]

        # Sort parts by part number and concatenate
        sorted_parts = sorted(upload.parts, key=lambda p: p["part_number"])
        data = b"".join(p["data"] for p in sorted_parts)

        # Store as regular object
        meta = await self.put_object(key, data)

        # Clean up
        del self._multipart_uploads[upload_id]

        return meta

    async def abort_multipart_upload(self, key: str, upload_id: str) -> bool:
        if upload_id in self._multipart_uploads:
            del self._multipart_uploads[upload_id]
            return True
        return False


class LocalFileStorage(StorageClient):
    """Local filesystem storage."""

    def __init__(self, config: StorageConfig):
        self.config = config
        self._base_path = Path(config.base_path) / config.bucket
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._multipart_dir = self._base_path / ".multipart"
        self._multipart_dir.mkdir(exist_ok=True)

    def _get_path(self, key: str) -> Path:
        """Get filesystem path for a key."""
        return self._base_path / key

    async def put_object(
        self,
        key: str,
        data: Union[bytes, BinaryIO],
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> ObjectMetadata:
        if isinstance(data, (io.IOBase, BinaryIO)):
            data = data.read()

        if len(data) > self.config.max_file_size:
            raise ValueError(f"File size exceeds maximum: {self.config.max_file_size}")

        file_path = self._get_path(key)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        await asyncio.to_thread(file_path.write_bytes, data)

        # Write metadata
        meta_path = file_path.with_suffix(file_path.suffix + ".meta")
        meta_content = {
            "content_type": content_type or mimetypes.guess_type(key)[0] or "application/octet-stream",
            "metadata": metadata or {},
        }
        import json
        await asyncio.to_thread(meta_path.write_text, json.dumps(meta_content))

        etag = hashlib.md5(data).hexdigest()

        return ObjectMetadata(
            key=key,
            size=len(data),
            content_type=meta_content["content_type"],
            etag=etag,
            last_modified=datetime.utcnow(),
            metadata=metadata or {},
        )

    async def get_object(self, key: str) -> bytes:
        file_path = self._get_path(key)
        if not file_path.exists():
            raise FileNotFoundError(f"Object not found: {key}")
        return await asyncio.to_thread(file_path.read_bytes)

    async def delete_object(self, key: str) -> bool:
        file_path = self._get_path(key)
        meta_path = file_path.with_suffix(file_path.suffix + ".meta")

        deleted = False
        if file_path.exists():
            await asyncio.to_thread(file_path.unlink)
            deleted = True
        if meta_path.exists():
            await asyncio.to_thread(meta_path.unlink)

        return deleted

    async def head_object(self, key: str) -> Optional[ObjectMetadata]:
        file_path = self._get_path(key)
        if not file_path.exists():
            return None

        stat = await asyncio.to_thread(file_path.stat)
        data = await asyncio.to_thread(file_path.read_bytes)
        etag = hashlib.md5(data).hexdigest()

        # Read metadata
        meta_path = file_path.with_suffix(file_path.suffix + ".meta")
        content_type = "application/octet-stream"
        metadata = {}

        if meta_path.exists():
            import json
            meta_content = json.loads(await asyncio.to_thread(meta_path.read_text))
            content_type = meta_content.get("content_type", content_type)
            metadata = meta_content.get("metadata", {})

        return ObjectMetadata(
            key=key,
            size=stat.st_size,
            content_type=content_type,
            etag=etag,
            last_modified=datetime.fromtimestamp(stat.st_mtime),
            metadata=metadata,
        )

    async def list_objects(
        self,
        prefix: str = "",
        max_keys: int = 1000,
        continuation_token: Optional[str] = None,
    ) -> tuple[List[ObjectMetadata], Optional[str]]:
        results = []

        for path in self._base_path.rglob("*"):
            if path.is_file() and not path.suffix.endswith(".meta"):
                key = str(path.relative_to(self._base_path))
                if key.startswith(prefix):
                    meta = await self.head_object(key)
                    if meta:
                        results.append(meta)

        results.sort(key=lambda m: m.key)

        start_index = 0
        if continuation_token:
            try:
                start_index = int(continuation_token)
            except ValueError:
                pass

        end_index = start_index + max_keys
        result = results[start_index:end_index]
        next_token = str(end_index) if end_index < len(results) else None

        return result, next_token

    async def generate_presigned_url(
        self,
        key: str,
        expires_in: int = 3600,
        method: str = "GET",
    ) -> str:
        # For local storage, return a file:// URL
        file_path = self._get_path(key)
        return f"file://{file_path.absolute()}"

    async def copy_object(self, source_key: str, dest_key: str) -> ObjectMetadata:
        data = await self.get_object(source_key)
        source_meta = await self.head_object(source_key)

        return await self.put_object(
            dest_key,
            data,
            source_meta.content_type if source_meta else None,
            source_meta.metadata if source_meta else None,
        )


class S3Storage(StorageClient):
    """AWS S3 compatible storage (S3, MinIO)."""

    def __init__(self, config: StorageConfig):
        self.config = config
        self._client = None

    async def _get_client(self):
        """Get or create the S3 client."""
        if self._client is None:
            try:
                import aioboto3
            except ImportError:
                raise ImportError("aioboto3 required for S3 storage. Install with: pip install aioboto3")

            session = aioboto3.Session()
            self._client = await session.client(
                "s3",
                endpoint_url=self.config.endpoint_url,
                aws_access_key_id=self.config.access_key,
                aws_secret_access_key=self.config.secret_key,
                region_name=self.config.region,
                use_ssl=self.config.use_ssl,
            ).__aenter__()
        return self._client

    async def put_object(
        self,
        key: str,
        data: Union[bytes, BinaryIO],
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> ObjectMetadata:
        client = await self._get_client()

        if isinstance(data, bytes):
            body = io.BytesIO(data)
        else:
            body = data

        content_type = content_type or mimetypes.guess_type(key)[0] or "application/octet-stream"

        kwargs = {
            "Bucket": self.config.bucket,
            "Key": key,
            "Body": body,
            "ContentType": content_type,
        }
        if metadata:
            kwargs["Metadata"] = metadata

        response = await client.put_object(**kwargs)

        body.seek(0)
        data_bytes = body.read() if hasattr(body, "read") else data

        return ObjectMetadata(
            key=key,
            size=len(data_bytes),
            content_type=content_type,
            etag=response.get("ETag", "").strip('"'),
            last_modified=datetime.utcnow(),
            metadata=metadata or {},
        )

    async def get_object(self, key: str) -> bytes:
        client = await self._get_client()
        response = await client.get_object(Bucket=self.config.bucket, Key=key)
        async with response["Body"] as stream:
            return await stream.read()

    async def delete_object(self, key: str) -> bool:
        client = await self._get_client()
        await client.delete_object(Bucket=self.config.bucket, Key=key)
        return True

    async def head_object(self, key: str) -> Optional[ObjectMetadata]:
        client = await self._get_client()
        try:
            response = await client.head_object(Bucket=self.config.bucket, Key=key)
            return ObjectMetadata(
                key=key,
                size=response["ContentLength"],
                content_type=response.get("ContentType", "application/octet-stream"),
                etag=response.get("ETag", "").strip('"'),
                last_modified=response.get("LastModified"),
                metadata=response.get("Metadata", {}),
            )
        except Exception:
            return None

    async def list_objects(
        self,
        prefix: str = "",
        max_keys: int = 1000,
        continuation_token: Optional[str] = None,
    ) -> tuple[List[ObjectMetadata], Optional[str]]:
        client = await self._get_client()

        kwargs = {
            "Bucket": self.config.bucket,
            "Prefix": prefix,
            "MaxKeys": max_keys,
        }
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token

        response = await client.list_objects_v2(**kwargs)

        results = []
        for obj in response.get("Contents", []):
            results.append(ObjectMetadata(
                key=obj["Key"],
                size=obj["Size"],
                content_type="application/octet-stream",  # Need head_object for actual type
                etag=obj.get("ETag", "").strip('"'),
                last_modified=obj.get("LastModified"),
            ))

        next_token = response.get("NextContinuationToken")
        return results, next_token

    async def generate_presigned_url(
        self,
        key: str,
        expires_in: int = 3600,
        method: str = "GET",
    ) -> str:
        client = await self._get_client()
        method_name = "get_object" if method == "GET" else "put_object"

        url = await client.generate_presigned_url(
            method_name,
            Params={"Bucket": self.config.bucket, "Key": key},
            ExpiresIn=expires_in,
        )
        return url

    async def copy_object(self, source_key: str, dest_key: str) -> ObjectMetadata:
        client = await self._get_client()

        copy_source = {"Bucket": self.config.bucket, "Key": source_key}
        response = await client.copy_object(
            Bucket=self.config.bucket,
            Key=dest_key,
            CopySource=copy_source,
        )

        # Get metadata for the new object
        return await self.head_object(dest_key)

    async def create_multipart_upload(self, key: str, content_type: Optional[str] = None) -> str:
        client = await self._get_client()
        response = await client.create_multipart_upload(
            Bucket=self.config.bucket,
            Key=key,
            ContentType=content_type or "application/octet-stream",
        )
        return response["UploadId"]

    async def upload_part(
        self,
        key: str,
        upload_id: str,
        part_number: int,
        data: bytes,
    ) -> Dict[str, Any]:
        client = await self._get_client()
        response = await client.upload_part(
            Bucket=self.config.bucket,
            Key=key,
            UploadId=upload_id,
            PartNumber=part_number,
            Body=data,
        )
        return {"ETag": response["ETag"], "PartNumber": part_number}

    async def complete_multipart_upload(
        self,
        key: str,
        upload_id: str,
        parts: List[Dict[str, Any]],
    ) -> ObjectMetadata:
        client = await self._get_client()
        await client.complete_multipart_upload(
            Bucket=self.config.bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )
        return await self.head_object(key)

    async def abort_multipart_upload(self, key: str, upload_id: str) -> bool:
        client = await self._get_client()
        await client.abort_multipart_upload(
            Bucket=self.config.bucket,
            Key=key,
            UploadId=upload_id,
        )
        return True


def create_storage_client(config: StorageConfig) -> StorageClient:
    """Create a storage client based on configuration."""
    if config.backend == StorageBackend.MEMORY:
        return InMemoryStorage(config)
    elif config.backend == StorageBackend.LOCAL:
        return LocalFileStorage(config)
    elif config.backend in (StorageBackend.S3, StorageBackend.MINIO):
        return S3Storage(config)
    else:
        raise ValueError(f"Unsupported storage backend: {config.backend}")


# Global storage client
_storage_client: Optional[StorageClient] = None


def get_storage_client() -> StorageClient:
    """Get the global storage client."""
    global _storage_client
    if _storage_client is None:
        # Default to local storage
        config = StorageConfig(backend=StorageBackend.LOCAL)
        _storage_client = create_storage_client(config)
    return _storage_client


def configure_storage(config: StorageConfig) -> StorageClient:
    """Configure and set the global storage client."""
    global _storage_client
    _storage_client = create_storage_client(config)
    return _storage_client
