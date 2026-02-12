"""ETL Data Sinks.

Provides data sink abstractions for ETL pipelines:
- File sinks (CSV, JSON, Parquet)
- Database sinks
- API sinks
- Memory sinks
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.core.etl.sources import Record

logger = logging.getLogger(__name__)


class SinkType(str, Enum):
    """Data sink types."""
    FILE = "file"
    DATABASE = "database"
    API = "api"
    MEMORY = "memory"


@dataclass
class SinkConfig:
    """Base sink configuration."""
    sink_type: SinkType
    name: str
    batch_size: int = 1000
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileSinkConfig(SinkConfig):
    """File sink configuration."""
    path: str = ""
    format: str = "csv"  # csv, json, jsonl
    encoding: str = "utf-8"
    delimiter: str = ","
    write_header: bool = True
    append: bool = False

    def __post_init__(self):
        self.sink_type = SinkType.FILE


@dataclass
class DatabaseSinkConfig(SinkConfig):
    """Database sink configuration."""
    connection_string: str = ""
    table: str = ""
    upsert_keys: Optional[List[str]] = None  # Keys for upsert operation
    create_table: bool = False

    def __post_init__(self):
        self.sink_type = SinkType.DATABASE


@dataclass
class APISinkConfig(SinkConfig):
    """API sink configuration."""
    url: str = ""
    method: str = "POST"
    headers: Dict[str, str] = field(default_factory=dict)
    batch_endpoint: bool = True  # Send records in batches

    def __post_init__(self):
        self.sink_type = SinkType.API


@dataclass
class WriteResult:
    """Result of a write operation."""
    written: int
    failed: int
    errors: List[Dict[str, Any]] = field(default_factory=list)


class DataSink(ABC):
    """Abstract base class for data sinks."""

    def __init__(self, config: SinkConfig):
        self.config = config
        self._is_open = False

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    async def open(self) -> None:
        """Open the data sink."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the data sink."""
        pass

    @abstractmethod
    async def write(self, records: List[Record]) -> WriteResult:
        """Write records to the sink."""
        pass

    async def write_batch(self, records: List[Record]) -> WriteResult:
        """Write records in batches."""
        total_written = 0
        total_failed = 0
        all_errors = []

        for i in range(0, len(records), self.config.batch_size):
            batch = records[i:i + self.config.batch_size]
            result = await self.write(batch)
            total_written += result.written
            total_failed += result.failed
            all_errors.extend(result.errors)

        return WriteResult(
            written=total_written,
            failed=total_failed,
            errors=all_errors,
        )

    async def __aenter__(self) -> "DataSink":
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()


class MemorySink(DataSink):
    """In-memory data sink for testing."""

    def __init__(self, config: SinkConfig):
        super().__init__(config)
        self.records: List[Record] = []

    async def open(self) -> None:
        self._is_open = True
        self.records = []

    async def close(self) -> None:
        self._is_open = False

    async def write(self, records: List[Record]) -> WriteResult:
        if not self._is_open:
            raise RuntimeError("Sink not open")

        self.records.extend(records)
        return WriteResult(written=len(records), failed=0)

    def get_data(self) -> List[Dict[str, Any]]:
        """Get written data."""
        return [r.data for r in self.records]


class CSVFileSink(DataSink):
    """CSV file data sink."""

    def __init__(self, config: FileSinkConfig):
        super().__init__(config)
        self._file: Optional[io.TextIOWrapper] = None
        self._writer: Optional[csv.DictWriter] = None
        self._fieldnames: Optional[List[str]] = None
        self._header_written = False

    async def open(self) -> None:
        path = Path(self.config.path)
        path.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if self.config.append else "w"
        self._file = await asyncio.to_thread(
            open, path, mode, encoding=self.config.encoding, newline=""
        )
        self._is_open = True
        self._header_written = self.config.append  # Skip header if appending

    async def close(self) -> None:
        if self._file:
            await asyncio.to_thread(self._file.close)
        self._is_open = False

    async def write(self, records: List[Record]) -> WriteResult:
        if not self._is_open or not self._file:
            raise RuntimeError("Sink not open")

        if not records:
            return WriteResult(written=0, failed=0)

        # Initialize writer with fieldnames from first record
        if self._writer is None:
            self._fieldnames = list(records[0].data.keys())
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=self._fieldnames,
                delimiter=self.config.delimiter,
            )

            if self.config.write_header and not self._header_written:
                self._writer.writeheader()
                self._header_written = True

        written = 0
        errors = []

        for record in records:
            try:
                # Filter to only known fieldnames
                row = {k: v for k, v in record.data.items() if k in self._fieldnames}
                self._writer.writerow(row)
                written += 1
            except Exception as e:
                errors.append({
                    "offset": record.offset,
                    "error": str(e),
                })

        await asyncio.to_thread(self._file.flush)
        return WriteResult(written=written, failed=len(errors), errors=errors)


class JSONFileSink(DataSink):
    """JSON/JSONL file data sink."""

    def __init__(self, config: FileSinkConfig):
        super().__init__(config)
        self._records: List[Dict[str, Any]] = []

    async def open(self) -> None:
        self._is_open = True
        self._records = []

        if self.config.append and self.config.format == "json":
            # Load existing data for JSON format
            path = Path(self.config.path)
            if path.exists():
                content = await asyncio.to_thread(
                    path.read_text, encoding=self.config.encoding
                )
                self._records = json.loads(content)

    async def close(self) -> None:
        if self._is_open:
            await self._flush()
        self._is_open = False

    async def _flush(self) -> None:
        """Flush records to file."""
        path = Path(self.config.path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.config.format == "jsonl":
            mode = "a" if self.config.append else "w"
            content = "\n".join(json.dumps(r) for r in self._records) + "\n"
            await asyncio.to_thread(path.open(mode).write, content)
        else:
            content = json.dumps(self._records, indent=2)
            await asyncio.to_thread(path.write_text, content, encoding=self.config.encoding)

    async def write(self, records: List[Record]) -> WriteResult:
        if not self._is_open:
            raise RuntimeError("Sink not open")

        for record in records:
            self._records.append(record.data)

        # For JSONL, we can write incrementally
        if self.config.format == "jsonl":
            path = Path(self.config.path)
            path.parent.mkdir(parents=True, exist_ok=True)

            mode = "a" if self.config.append or path.exists() else "w"
            content = "\n".join(json.dumps(r.data) for r in records) + "\n"

            async def write_content():
                with open(path, mode, encoding=self.config.encoding) as f:
                    f.write(content)

            await asyncio.to_thread(write_content)
            self._records = []  # Clear since we've written

        return WriteResult(written=len(records), failed=0)


class APISink(DataSink):
    """HTTP API data sink."""

    def __init__(self, config: APISinkConfig):
        super().__init__(config)

    async def open(self) -> None:
        self._is_open = True

    async def close(self) -> None:
        self._is_open = False

    async def write(self, records: List[Record]) -> WriteResult:
        if not self._is_open:
            raise RuntimeError("Sink not open")

        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required for API sink. Install with: pip install httpx")

        written = 0
        errors = []

        async with httpx.AsyncClient() as client:
            if self.config.batch_endpoint:
                # Send all records in one request
                data = [r.data for r in records]
                try:
                    if self.config.method == "POST":
                        response = await client.post(
                            self.config.url,
                            headers=self.config.headers,
                            json=data,
                        )
                    else:
                        response = await client.request(
                            self.config.method,
                            self.config.url,
                            headers=self.config.headers,
                            json=data,
                        )
                    response.raise_for_status()
                    written = len(records)
                except Exception as e:
                    errors.append({"error": str(e)})
            else:
                # Send each record individually
                for record in records:
                    try:
                        response = await client.post(
                            self.config.url,
                            headers=self.config.headers,
                            json=record.data,
                        )
                        response.raise_for_status()
                        written += 1
                    except Exception as e:
                        errors.append({
                            "offset": record.offset,
                            "error": str(e),
                        })

        return WriteResult(written=written, failed=len(errors), errors=errors)


def create_sink(config: SinkConfig) -> DataSink:
    """Create a data sink from configuration."""
    if config.sink_type == SinkType.MEMORY:
        return MemorySink(config)

    if config.sink_type == SinkType.FILE:
        file_config = config
        if file_config.format in ("csv", "tsv"):
            return CSVFileSink(file_config)
        elif file_config.format in ("json", "jsonl"):
            return JSONFileSink(file_config)
        else:
            raise ValueError(f"Unsupported file format: {file_config.format}")

    if config.sink_type == SinkType.API:
        return APISink(config)

    raise ValueError(f"Unsupported sink type: {config.sink_type}")
