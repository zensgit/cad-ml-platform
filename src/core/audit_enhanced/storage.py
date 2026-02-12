"""Audit Record Storage Backends.

Provides persistent storage for audit records:
- In-memory storage (testing)
- File-based storage (simple deployment)
- Database interface (production)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.audit_enhanced.record import (
    AuditCategory,
    AuditOutcome,
    AuditRecord,
    AuditSeverity,
    ChainMetadata,
    RecordIntegrity,
)

logger = logging.getLogger(__name__)


class AuditStorage(ABC):
    """Abstract base class for audit storage."""

    @abstractmethod
    async def store(self, record: AuditRecord) -> bool:
        """Store a single audit record."""
        pass

    @abstractmethod
    async def store_batch(self, records: List[AuditRecord]) -> int:
        """Store multiple records. Returns count of successfully stored."""
        pass

    @abstractmethod
    async def get(self, record_id: str) -> Optional[AuditRecord]:
        """Retrieve a record by ID."""
        pass

    @abstractmethod
    async def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        categories: Optional[List[AuditCategory]] = None,
        severities: Optional[List[AuditSeverity]] = None,
        outcomes: Optional[List[AuditOutcome]] = None,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        action: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditRecord]:
        """Query records with filters."""
        pass

    @abstractmethod
    async def count(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        categories: Optional[List[AuditCategory]] = None,
    ) -> int:
        """Count records matching criteria."""
        pass

    @abstractmethod
    async def delete_before(self, before_time: datetime) -> int:
        """Delete records before specified time. Returns count deleted."""
        pass

    @abstractmethod
    async def get_chain_records(
        self,
        start_hash: Optional[str] = None,
        limit: int = 1000,
    ) -> List[AuditRecord]:
        """Get records in chain order for verification."""
        pass


class InMemoryAuditStorage(AuditStorage):
    """In-memory audit storage for testing."""

    def __init__(self, max_records: int = 100000):
        self._records: Dict[str, AuditRecord] = {}
        self._ordered_ids: List[str] = []
        self._max_records = max_records
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def store(self, record: AuditRecord) -> bool:
        async with self._get_lock():
            # Enforce max records
            if len(self._records) >= self._max_records:
                # Remove oldest
                oldest_id = self._ordered_ids.pop(0)
                del self._records[oldest_id]

            self._records[record.record_id] = record
            self._ordered_ids.append(record.record_id)
            return True

    async def store_batch(self, records: List[AuditRecord]) -> int:
        count = 0
        for record in records:
            if await self.store(record):
                count += 1
        return count

    async def get(self, record_id: str) -> Optional[AuditRecord]:
        async with self._get_lock():
            return self._records.get(record_id)

    async def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        categories: Optional[List[AuditCategory]] = None,
        severities: Optional[List[AuditSeverity]] = None,
        outcomes: Optional[List[AuditOutcome]] = None,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        action: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditRecord]:
        async with self._get_lock():
            results = []

            for record_id in reversed(self._ordered_ids):
                record = self._records[record_id]

                # Apply filters
                if start_time and record.timestamp < start_time:
                    continue
                if end_time and record.timestamp > end_time:
                    continue
                if categories and record.category not in categories:
                    continue
                if severities and record.severity not in severities:
                    continue
                if outcomes and record.outcome not in outcomes:
                    continue
                if user_id and (not record.context or record.context.user_id != user_id):
                    continue
                if resource_type and record.resource_type != resource_type:
                    continue
                if resource_id and record.resource_id != resource_id:
                    continue
                if action and record.action != action:
                    continue
                if tags and not all(t in record.tags for t in tags):
                    continue

                results.append(record)

            # Apply pagination
            return results[offset:offset + limit]

    async def count(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        categories: Optional[List[AuditCategory]] = None,
    ) -> int:
        records = await self.query(
            start_time=start_time,
            end_time=end_time,
            categories=categories,
            limit=self._max_records,
        )
        return len(records)

    async def delete_before(self, before_time: datetime) -> int:
        async with self._get_lock():
            to_delete = []
            for record_id, record in self._records.items():
                if record.timestamp < before_time:
                    to_delete.append(record_id)

            for record_id in to_delete:
                del self._records[record_id]
                self._ordered_ids.remove(record_id)

            return len(to_delete)

    async def get_chain_records(
        self,
        start_hash: Optional[str] = None,
        limit: int = 1000,
    ) -> List[AuditRecord]:
        async with self._get_lock():
            if start_hash:
                # Find starting point
                start_idx = 0
                for i, record_id in enumerate(self._ordered_ids):
                    if self._records[record_id].record_hash == start_hash:
                        start_idx = i
                        break
                ids = self._ordered_ids[start_idx:start_idx + limit]
            else:
                ids = self._ordered_ids[:limit]

            return [self._records[rid] for rid in ids]


class FileAuditStorage(AuditStorage):
    """File-based audit storage."""

    def __init__(
        self,
        directory: str,
        records_per_file: int = 10000,
        integrity: Optional[RecordIntegrity] = None,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.records_per_file = records_per_file
        self._integrity = integrity or RecordIntegrity()
        self._current_file_records = 0
        self._current_file_path: Optional[Path] = None
        self._lock: Optional[asyncio.Lock] = None
        self._index: Dict[str, str] = {}  # record_id -> file_path

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _get_current_file(self) -> Path:
        """Get or create current file for writing."""
        if (
            self._current_file_path is None
            or self._current_file_records >= self.records_per_file
        ):
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            self._current_file_path = self.directory / f"audit_{timestamp}.jsonl"
            self._current_file_records = 0
        return self._current_file_path

    async def store(self, record: AuditRecord) -> bool:
        async with self._get_lock():
            try:
                file_path = self._get_current_file()
                with open(file_path, "a") as f:
                    f.write(json.dumps(record.to_dict()) + "\n")

                self._current_file_records += 1
                self._index[record.record_id] = str(file_path)
                return True
            except Exception as e:
                logger.error(f"Failed to store audit record: {e}")
                return False

    async def store_batch(self, records: List[AuditRecord]) -> int:
        count = 0
        for record in records:
            if await self.store(record):
                count += 1
        return count

    async def get(self, record_id: str) -> Optional[AuditRecord]:
        async with self._get_lock():
            # Check index first
            if record_id in self._index:
                return await self._read_from_file(self._index[record_id], record_id)

            # Search all files
            for file_path in sorted(self.directory.glob("audit_*.jsonl"), reverse=True):
                record = await self._read_from_file(str(file_path), record_id)
                if record:
                    self._index[record_id] = str(file_path)
                    return record

            return None

    async def _read_from_file(
        self,
        file_path: str,
        record_id: str,
    ) -> Optional[AuditRecord]:
        """Read specific record from file."""
        try:
            with open(file_path, "r") as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data.get("record_id") == record_id:
                        return AuditRecord.from_dict(data)
        except Exception as e:
            logger.error(f"Error reading from {file_path}: {e}")
        return None

    async def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        categories: Optional[List[AuditCategory]] = None,
        severities: Optional[List[AuditSeverity]] = None,
        outcomes: Optional[List[AuditOutcome]] = None,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        action: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditRecord]:
        async with self._get_lock():
            results = []
            skipped = 0

            # Sort files by name (timestamp) in reverse order
            for file_path in sorted(self.directory.glob("audit_*.jsonl"), reverse=True):
                try:
                    with open(file_path, "r") as f:
                        for line in f:
                            data = json.loads(line.strip())
                            record = AuditRecord.from_dict(data)

                            # Apply filters
                            if not self._matches_filters(
                                record, start_time, end_time, categories,
                                severities, outcomes, user_id, resource_type,
                                resource_id, action, tags
                            ):
                                continue

                            if skipped < offset:
                                skipped += 1
                                continue

                            results.append(record)
                            if len(results) >= limit:
                                return results

                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")

            return results

    def _matches_filters(
        self,
        record: AuditRecord,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        categories: Optional[List[AuditCategory]],
        severities: Optional[List[AuditSeverity]],
        outcomes: Optional[List[AuditOutcome]],
        user_id: Optional[str],
        resource_type: Optional[str],
        resource_id: Optional[str],
        action: Optional[str],
        tags: Optional[List[str]],
    ) -> bool:
        """Check if record matches all filters."""
        if start_time and record.timestamp < start_time:
            return False
        if end_time and record.timestamp > end_time:
            return False
        if categories and record.category not in categories:
            return False
        if severities and record.severity not in severities:
            return False
        if outcomes and record.outcome not in outcomes:
            return False
        if user_id and (not record.context or record.context.user_id != user_id):
            return False
        if resource_type and record.resource_type != resource_type:
            return False
        if resource_id and record.resource_id != resource_id:
            return False
        if action and record.action != action:
            return False
        if tags and not all(t in record.tags for t in tags):
            return False
        return True

    async def count(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        categories: Optional[List[AuditCategory]] = None,
    ) -> int:
        count = 0
        for file_path in self.directory.glob("audit_*.jsonl"):
            try:
                with open(file_path, "r") as f:
                    for line in f:
                        data = json.loads(line.strip())
                        record = AuditRecord.from_dict(data)
                        if self._matches_filters(
                            record, start_time, end_time, categories,
                            None, None, None, None, None, None, None
                        ):
                            count += 1
            except Exception:
                continue
        return count

    async def delete_before(self, before_time: datetime) -> int:
        async with self._get_lock():
            deleted = 0
            files_to_check = list(self.directory.glob("audit_*.jsonl"))

            for file_path in files_to_check:
                try:
                    records_to_keep = []
                    with open(file_path, "r") as f:
                        for line in f:
                            data = json.loads(line.strip())
                            timestamp = datetime.fromisoformat(data["timestamp"])
                            if timestamp >= before_time:
                                records_to_keep.append(line)
                            else:
                                deleted += 1
                                record_id = data.get("record_id")
                                if record_id in self._index:
                                    del self._index[record_id]

                    if records_to_keep:
                        with open(file_path, "w") as f:
                            f.writelines(records_to_keep)
                    else:
                        file_path.unlink()
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")

            return deleted

    async def get_chain_records(
        self,
        start_hash: Optional[str] = None,
        limit: int = 1000,
    ) -> List[AuditRecord]:
        """Get records in chain order."""
        records = []
        found_start = start_hash is None

        for file_path in sorted(self.directory.glob("audit_*.jsonl")):
            try:
                with open(file_path, "r") as f:
                    for line in f:
                        data = json.loads(line.strip())
                        record = AuditRecord.from_dict(data)

                        if not found_start:
                            if record.record_hash == start_hash:
                                found_start = True
                            else:
                                continue

                        records.append(record)
                        if len(records) >= limit:
                            return records
            except Exception:
                continue

        return records


class RotatingFileStorage(AuditStorage):
    """File storage with automatic rotation and compression."""

    def __init__(
        self,
        directory: str,
        max_file_size_mb: int = 100,
        max_files: int = 100,
        compress_old: bool = True,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.max_files = max_files
        self.compress_old = compress_old
        self._base_storage = FileAuditStorage(directory)
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def store(self, record: AuditRecord) -> bool:
        await self._check_rotation()
        return await self._base_storage.store(record)

    async def store_batch(self, records: List[AuditRecord]) -> int:
        await self._check_rotation()
        return await self._base_storage.store_batch(records)

    async def _check_rotation(self) -> None:
        """Check if rotation is needed."""
        async with self._get_lock():
            current_file = self._base_storage._get_current_file()
            if current_file.exists():
                size = current_file.stat().st_size
                if size >= self.max_file_size:
                    # Force new file
                    self._base_storage._current_file_records = self._base_storage.records_per_file

            # Check max files
            files = sorted(self.directory.glob("audit_*.jsonl"))
            while len(files) > self.max_files:
                oldest = files.pop(0)
                if self.compress_old:
                    await self._compress_file(oldest)
                oldest.unlink(missing_ok=True)

    async def _compress_file(self, file_path: Path) -> None:
        """Compress a file using gzip."""
        import gzip
        import shutil

        compressed_path = file_path.with_suffix(".jsonl.gz")
        try:
            with open(file_path, "rb") as f_in:
                with gzip.open(compressed_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            logger.info(f"Compressed {file_path} to {compressed_path}")
        except Exception as e:
            logger.error(f"Failed to compress {file_path}: {e}")

    async def get(self, record_id: str) -> Optional[AuditRecord]:
        return await self._base_storage.get(record_id)

    async def query(self, **kwargs) -> List[AuditRecord]:
        return await self._base_storage.query(**kwargs)

    async def count(self, **kwargs) -> int:
        return await self._base_storage.count(**kwargs)

    async def delete_before(self, before_time: datetime) -> int:
        return await self._base_storage.delete_before(before_time)

    async def get_chain_records(
        self,
        start_hash: Optional[str] = None,
        limit: int = 1000,
    ) -> List[AuditRecord]:
        return await self._base_storage.get_chain_records(start_hash, limit)
