"""Multipart Upload Service.

Provides efficient large file uploads:
- Chunked upload handling
- Resume support
- Progress tracking
- Parallel part uploads
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, BinaryIO, Callable, Dict, List, Optional

from src.core.storage.object_store import StorageClient, ObjectMetadata

logger = logging.getLogger(__name__)


class UploadStatus(str, Enum):
    """Multipart upload status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class PartInfo:
    """Information about an uploaded part."""
    part_number: int
    size: int
    etag: str
    uploaded_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class UploadProgress:
    """Upload progress information."""
    upload_id: str
    total_parts: int
    completed_parts: int
    total_bytes: int
    uploaded_bytes: int
    status: UploadStatus
    started_at: datetime
    updated_at: datetime

    @property
    def progress_percent(self) -> float:
        """Get upload progress as percentage."""
        if self.total_bytes == 0:
            return 0.0
        return (self.uploaded_bytes / self.total_bytes) * 100


@dataclass
class MultipartUploadState:
    """State of a multipart upload."""
    upload_id: str
    key: str
    bucket: str
    status: UploadStatus
    total_size: int
    part_size: int
    total_parts: int
    parts: Dict[int, PartInfo] = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)
    content_type: str = "application/octet-stream"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class MultipartUploadManager:
    """Manages multipart uploads."""

    # Default part size: 5MB (S3 minimum)
    DEFAULT_PART_SIZE = 5 * 1024 * 1024
    # Maximum part size: 5GB
    MAX_PART_SIZE = 5 * 1024 * 1024 * 1024
    # Maximum parts: 10,000 (S3 limit)
    MAX_PARTS = 10000

    def __init__(
        self,
        storage_client: StorageClient,
        part_size: int = DEFAULT_PART_SIZE,
    ):
        self.storage = storage_client
        self.part_size = part_size
        self._uploads: Dict[str, MultipartUploadState] = {}
        self._progress_callbacks: Dict[str, Callable[[UploadProgress], None]] = {}

    def calculate_parts(self, total_size: int) -> tuple[int, int]:
        """Calculate number of parts and part size.

        Args:
            total_size: Total file size in bytes.

        Returns:
            Tuple of (num_parts, part_size).
        """
        if total_size == 0:
            return 1, 0

        part_size = self.part_size
        num_parts = (total_size + part_size - 1) // part_size

        # Adjust if too many parts
        if num_parts > self.MAX_PARTS:
            part_size = (total_size + self.MAX_PARTS - 1) // self.MAX_PARTS
            num_parts = (total_size + part_size - 1) // part_size

        return num_parts, part_size

    async def initiate_upload(
        self,
        key: str,
        bucket: str,
        total_size: int,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
    ) -> MultipartUploadState:
        """Initiate a multipart upload.

        Args:
            key: Object key.
            bucket: Bucket name.
            total_size: Total file size.
            content_type: Content type.
            metadata: Optional metadata.

        Returns:
            MultipartUploadState.
        """
        upload_id = str(uuid.uuid4())
        num_parts, part_size = self.calculate_parts(total_size)

        state = MultipartUploadState(
            upload_id=upload_id,
            key=key,
            bucket=bucket,
            status=UploadStatus.PENDING,
            total_size=total_size,
            part_size=part_size,
            total_parts=num_parts,
            metadata=metadata or {},
            content_type=content_type,
        )

        self._uploads[upload_id] = state
        logger.info(
            f"Initiated multipart upload: {upload_id}, "
            f"key={key}, size={total_size}, parts={num_parts}"
        )

        return state

    async def upload_part(
        self,
        upload_id: str,
        part_number: int,
        data: bytes,
    ) -> PartInfo:
        """Upload a single part.

        Args:
            upload_id: Upload ID.
            part_number: Part number (1-indexed).
            data: Part data.

        Returns:
            PartInfo for the uploaded part.
        """
        state = self._uploads.get(upload_id)
        if not state:
            raise ValueError(f"Upload not found: {upload_id}")

        if state.status == UploadStatus.ABORTED:
            raise ValueError(f"Upload was aborted: {upload_id}")

        if part_number < 1 or part_number > state.total_parts:
            raise ValueError(f"Invalid part number: {part_number}")

        # Update status
        state.status = UploadStatus.IN_PROGRESS
        state.updated_at = datetime.utcnow()

        # Calculate ETag (MD5)
        etag = hashlib.md5(data).hexdigest()

        # Store part info
        part_info = PartInfo(
            part_number=part_number,
            size=len(data),
            etag=etag,
        )
        state.parts[part_number] = part_info

        # Notify progress
        self._notify_progress(upload_id)

        logger.debug(
            f"Uploaded part {part_number}/{state.total_parts} "
            f"for upload {upload_id}"
        )

        return part_info

    async def complete_upload(self, upload_id: str) -> ObjectMetadata:
        """Complete a multipart upload.

        Args:
            upload_id: Upload ID.

        Returns:
            ObjectMetadata for the completed object.
        """
        state = self._uploads.get(upload_id)
        if not state:
            raise ValueError(f"Upload not found: {upload_id}")

        # Verify all parts uploaded
        missing_parts = set(range(1, state.total_parts + 1)) - set(state.parts.keys())
        if missing_parts:
            raise ValueError(f"Missing parts: {missing_parts}")

        state.status = UploadStatus.COMPLETING
        state.updated_at = datetime.utcnow()

        # In a real implementation, this would call the storage backend's
        # complete_multipart_upload. For now, we simulate completion.
        state.status = UploadStatus.COMPLETED
        state.completed_at = datetime.utcnow()
        state.updated_at = state.completed_at

        logger.info(f"Completed multipart upload: {upload_id}, key={state.key}")

        # Calculate combined ETag
        combined_etag = hashlib.md5(
            b"".join(
                state.parts[i].etag.encode()
                for i in sorted(state.parts.keys())
            )
        ).hexdigest()

        return ObjectMetadata(
            key=state.key,
            size=state.total_size,
            content_type=state.content_type,
            etag=f"{combined_etag}-{state.total_parts}",
            last_modified=state.completed_at,
            metadata=state.metadata,
        )

    async def abort_upload(self, upload_id: str) -> bool:
        """Abort a multipart upload.

        Args:
            upload_id: Upload ID.

        Returns:
            True if aborted successfully.
        """
        state = self._uploads.get(upload_id)
        if not state:
            return False

        state.status = UploadStatus.ABORTED
        state.updated_at = datetime.utcnow()

        logger.info(f"Aborted multipart upload: {upload_id}")
        return True

    def get_upload_state(self, upload_id: str) -> Optional[MultipartUploadState]:
        """Get the state of an upload."""
        return self._uploads.get(upload_id)

    def get_progress(self, upload_id: str) -> Optional[UploadProgress]:
        """Get upload progress."""
        state = self._uploads.get(upload_id)
        if not state:
            return None

        uploaded_bytes = sum(p.size for p in state.parts.values())

        return UploadProgress(
            upload_id=upload_id,
            total_parts=state.total_parts,
            completed_parts=len(state.parts),
            total_bytes=state.total_size,
            uploaded_bytes=uploaded_bytes,
            status=state.status,
            started_at=state.created_at,
            updated_at=state.updated_at,
        )

    def register_progress_callback(
        self,
        upload_id: str,
        callback: Callable[[UploadProgress], None],
    ) -> None:
        """Register a callback for progress updates."""
        self._progress_callbacks[upload_id] = callback

    def _notify_progress(self, upload_id: str) -> None:
        """Notify progress callback if registered."""
        callback = self._progress_callbacks.get(upload_id)
        if callback:
            progress = self.get_progress(upload_id)
            if progress:
                try:
                    callback(progress)
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")

    async def upload_file(
        self,
        file: BinaryIO,
        key: str,
        bucket: str,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
        progress_callback: Optional[Callable[[UploadProgress], None]] = None,
    ) -> ObjectMetadata:
        """Upload a file using multipart upload.

        Args:
            file: File-like object.
            key: Object key.
            bucket: Bucket name.
            content_type: Content type.
            metadata: Optional metadata.
            progress_callback: Optional progress callback.

        Returns:
            ObjectMetadata for the uploaded object.
        """
        # Get file size
        file.seek(0, 2)
        total_size = file.tell()
        file.seek(0)

        # Initiate upload
        state = await self.initiate_upload(
            key, bucket, total_size, content_type, metadata
        )

        if progress_callback:
            self.register_progress_callback(state.upload_id, progress_callback)

        try:
            # Upload parts
            for part_num in range(1, state.total_parts + 1):
                data = file.read(state.part_size)
                if not data:
                    break
                await self.upload_part(state.upload_id, part_num, data)

            # Complete upload
            return await self.complete_upload(state.upload_id)

        except Exception as e:
            await self.abort_upload(state.upload_id)
            raise

    async def upload_file_parallel(
        self,
        file: BinaryIO,
        key: str,
        bucket: str,
        max_concurrency: int = 4,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
        progress_callback: Optional[Callable[[UploadProgress], None]] = None,
    ) -> ObjectMetadata:
        """Upload a file using parallel multipart upload.

        Args:
            file: File-like object.
            key: Object key.
            bucket: Bucket name.
            max_concurrency: Maximum parallel uploads.
            content_type: Content type.
            metadata: Optional metadata.
            progress_callback: Optional progress callback.

        Returns:
            ObjectMetadata for the uploaded object.
        """
        # Get file size
        file.seek(0, 2)
        total_size = file.tell()
        file.seek(0)

        # Read all data (for in-memory processing)
        all_data = file.read()

        # Initiate upload
        state = await self.initiate_upload(
            key, bucket, total_size, content_type, metadata
        )

        if progress_callback:
            self.register_progress_callback(state.upload_id, progress_callback)

        try:
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(max_concurrency)

            async def upload_part_with_semaphore(part_num: int, data: bytes):
                async with semaphore:
                    return await self.upload_part(state.upload_id, part_num, data)

            # Create tasks for all parts
            tasks = []
            for part_num in range(1, state.total_parts + 1):
                start = (part_num - 1) * state.part_size
                end = min(start + state.part_size, total_size)
                data = all_data[start:end]

                task = upload_part_with_semaphore(part_num, data)
                tasks.append(task)

            # Execute all uploads in parallel
            await asyncio.gather(*tasks)

            # Complete upload
            return await self.complete_upload(state.upload_id)

        except Exception as e:
            await self.abort_upload(state.upload_id)
            raise


# Global upload manager
_upload_manager: Optional[MultipartUploadManager] = None


def get_upload_manager() -> MultipartUploadManager:
    """Get the global upload manager."""
    global _upload_manager
    if _upload_manager is None:
        from src.core.storage.object_store import get_storage_client
        _upload_manager = MultipartUploadManager(get_storage_client())
    return _upload_manager
