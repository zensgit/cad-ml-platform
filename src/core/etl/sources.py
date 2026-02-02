"""ETL Data Sources.

Provides data source abstractions for ETL pipelines:
- File sources (CSV, JSON, Parquet)
- Database sources
- API sources
- Stream sources
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
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class SourceType(str, Enum):
    """Data source types."""
    FILE = "file"
    DATABASE = "database"
    API = "api"
    STREAM = "stream"
    MEMORY = "memory"


@dataclass
class SourceConfig:
    """Base source configuration."""
    source_type: SourceType
    name: str
    batch_size: int = 1000
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileSourceConfig(SourceConfig):
    """File source configuration."""
    path: str = ""
    format: str = "csv"  # csv, json, jsonl, parquet
    encoding: str = "utf-8"
    delimiter: str = ","
    has_header: bool = True

    def __post_init__(self):
        self.source_type = SourceType.FILE


@dataclass
class DatabaseSourceConfig(SourceConfig):
    """Database source configuration."""
    connection_string: str = ""
    query: str = ""
    table: Optional[str] = None

    def __post_init__(self):
        self.source_type = SourceType.DATABASE


@dataclass
class APISourceConfig(SourceConfig):
    """API source configuration."""
    url: str = ""
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, str] = field(default_factory=dict)
    pagination_key: Optional[str] = None
    data_path: Optional[str] = None  # JSON path to data array

    def __post_init__(self):
        self.source_type = SourceType.API


@dataclass
class Record:
    """A data record from a source."""
    data: Dict[str, Any]
    source: str
    offset: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataSource(ABC):
    """Abstract base class for data sources."""

    def __init__(self, config: SourceConfig):
        self.config = config
        self._is_open = False

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    async def open(self) -> None:
        """Open the data source."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the data source."""
        pass

    @abstractmethod
    async def read_batch(self, batch_size: Optional[int] = None) -> List[Record]:
        """Read a batch of records."""
        pass

    async def read_all(self) -> List[Record]:
        """Read all records."""
        records = []
        while True:
            batch = await self.read_batch()
            if not batch:
                break
            records.extend(batch)
        return records

    async def __aiter__(self) -> AsyncIterator[Record]:
        """Iterate over records."""
        while True:
            batch = await self.read_batch()
            if not batch:
                break
            for record in batch:
                yield record

    async def __aenter__(self) -> "DataSource":
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()


class MemorySource(DataSource):
    """In-memory data source for testing."""

    def __init__(self, config: SourceConfig, data: List[Dict[str, Any]]):
        super().__init__(config)
        self._data = data
        self._offset = 0

    async def open(self) -> None:
        self._is_open = True
        self._offset = 0

    async def close(self) -> None:
        self._is_open = False

    async def read_batch(self, batch_size: Optional[int] = None) -> List[Record]:
        if not self._is_open:
            raise RuntimeError("Source not open")

        batch_size = batch_size or self.config.batch_size
        batch = self._data[self._offset:self._offset + batch_size]

        records = [
            Record(
                data=item,
                source=self.name,
                offset=self._offset + i,
            )
            for i, item in enumerate(batch)
        ]

        self._offset += len(batch)
        return records


class CSVFileSource(DataSource):
    """CSV file data source."""

    def __init__(self, config: FileSourceConfig):
        super().__init__(config)
        self._file: Optional[io.TextIOWrapper] = None
        self._reader: Optional[csv.DictReader] = None
        self._offset = 0

    async def open(self) -> None:
        path = Path(self.config.path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        self._file = await asyncio.to_thread(
            open, path, "r", encoding=self.config.encoding
        )
        self._reader = csv.DictReader(
            self._file,
            delimiter=self.config.delimiter,
        )
        self._is_open = True
        self._offset = 0

    async def close(self) -> None:
        if self._file:
            await asyncio.to_thread(self._file.close)
        self._is_open = False

    async def read_batch(self, batch_size: Optional[int] = None) -> List[Record]:
        if not self._is_open or not self._reader:
            raise RuntimeError("Source not open")

        batch_size = batch_size or self.config.batch_size
        records = []

        for _ in range(batch_size):
            try:
                row = next(self._reader)
                records.append(Record(
                    data=dict(row),
                    source=self.name,
                    offset=self._offset,
                ))
                self._offset += 1
            except StopIteration:
                break

        return records


class JSONFileSource(DataSource):
    """JSON/JSONL file data source."""

    def __init__(self, config: FileSourceConfig):
        super().__init__(config)
        self._data: List[Dict[str, Any]] = []
        self._offset = 0

    async def open(self) -> None:
        path = Path(self.config.path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = await asyncio.to_thread(
            path.read_text, encoding=self.config.encoding
        )

        if self.config.format == "jsonl":
            # JSON Lines format
            self._data = [
                json.loads(line)
                for line in content.strip().split("\n")
                if line.strip()
            ]
        else:
            # Regular JSON
            data = json.loads(content)
            if isinstance(data, list):
                self._data = data
            else:
                self._data = [data]

        self._is_open = True
        self._offset = 0

    async def close(self) -> None:
        self._data = []
        self._is_open = False

    async def read_batch(self, batch_size: Optional[int] = None) -> List[Record]:
        if not self._is_open:
            raise RuntimeError("Source not open")

        batch_size = batch_size or self.config.batch_size
        batch = self._data[self._offset:self._offset + batch_size]

        records = [
            Record(
                data=item,
                source=self.name,
                offset=self._offset + i,
            )
            for i, item in enumerate(batch)
        ]

        self._offset += len(batch)
        return records


class APISource(DataSource):
    """HTTP API data source."""

    def __init__(self, config: APISourceConfig):
        super().__init__(config)
        self._data: List[Dict[str, Any]] = []
        self._offset = 0
        self._next_page: Optional[str] = None

    async def open(self) -> None:
        await self._fetch_data()
        self._is_open = True
        self._offset = 0

    async def _fetch_data(self) -> None:
        """Fetch data from API."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required for API source. Install with: pip install httpx")

        async with httpx.AsyncClient() as client:
            url = self._next_page or self.config.url

            if self.config.method == "GET":
                response = await client.get(
                    url,
                    headers=self.config.headers,
                    params=self.config.params,
                )
            else:
                response = await client.post(
                    url,
                    headers=self.config.headers,
                    json=self.config.params,
                )

            response.raise_for_status()
            data = response.json()

            # Extract data from response
            if self.config.data_path:
                # Navigate to data path (e.g., "results.items")
                for key in self.config.data_path.split("."):
                    data = data[key]

            if isinstance(data, list):
                self._data.extend(data)
            else:
                self._data.append(data)

            # Handle pagination
            if self.config.pagination_key:
                self._next_page = response.json().get(self.config.pagination_key)

    async def close(self) -> None:
        self._data = []
        self._is_open = False

    async def read_batch(self, batch_size: Optional[int] = None) -> List[Record]:
        if not self._is_open:
            raise RuntimeError("Source not open")

        batch_size = batch_size or self.config.batch_size

        # Fetch more data if needed and pagination is available
        while len(self._data) - self._offset < batch_size and self._next_page:
            await self._fetch_data()

        batch = self._data[self._offset:self._offset + batch_size]

        records = [
            Record(
                data=item,
                source=self.name,
                offset=self._offset + i,
            )
            for i, item in enumerate(batch)
        ]

        self._offset += len(batch)
        return records


def create_source(config: SourceConfig) -> DataSource:
    """Create a data source from configuration."""
    if config.source_type == SourceType.MEMORY:
        return MemorySource(config, [])

    if config.source_type == SourceType.FILE:
        file_config = config
        if file_config.format in ("csv", "tsv"):
            return CSVFileSource(file_config)
        elif file_config.format in ("json", "jsonl"):
            return JSONFileSource(file_config)
        else:
            raise ValueError(f"Unsupported file format: {file_config.format}")

    if config.source_type == SourceType.API:
        return APISource(config)

    raise ValueError(f"Unsupported source type: {config.source_type}")
