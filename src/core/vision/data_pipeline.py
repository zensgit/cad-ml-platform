"""Data pipeline and ETL processes for Vision Provider system.

This module provides data pipeline features including:
- ETL (Extract, Transform, Load) processes
- Data transformations
- Batch processing
- Pipeline orchestration
- Data quality checks
"""

import asyncio
import json
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from .base import VisionDescription, VisionProvider


class PipelineStageType(Enum):
    """Pipeline stage types."""

    EXTRACT = "extract"
    TRANSFORM = "transform"
    LOAD = "load"
    VALIDATE = "validate"
    FILTER = "filter"
    AGGREGATE = "aggregate"
    ENRICH = "enrich"
    CUSTOM = "custom"


class PipelineStatus(Enum):
    """Pipeline status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class DataFormat(Enum):
    """Data formats."""

    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    AVRO = "avro"
    BYTES = "bytes"
    DICT = "dict"


T = TypeVar("T")


@dataclass
class DataRecord:
    """Data record."""

    record_id: str
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    format: DataFormat = DataFormat.DICT

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "record_id": self.record_id,
            "data": self.data,
            "metadata": dict(self.metadata),
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "format": self.format.value,
        }


@dataclass
class BatchResult:
    """Batch processing result."""

    batch_id: str
    total_records: int
    processed_records: int
    failed_records: int
    start_time: datetime
    end_time: Optional[datetime] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Get success rate."""
        if self.total_records == 0:
            return 0.0
        return (self.processed_records - self.failed_records) / self.total_records

    @property
    def duration(self) -> Optional[timedelta]:
        """Get duration."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_id": self.batch_id,
            "total_records": self.total_records,
            "processed_records": self.processed_records,
            "failed_records": self.failed_records,
            "success_rate": self.success_rate,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration.total_seconds() if self.duration else None,
            "errors": list(self.errors),
            "metrics": dict(self.metrics),
        }


class DataSource(ABC):
    """Abstract data source."""

    @abstractmethod
    def read(self) -> Iterator[DataRecord]:
        """Read data records."""
        pass

    @abstractmethod
    def read_batch(self, batch_size: int) -> List[DataRecord]:
        """Read batch of records."""
        pass


class DataSink(ABC):
    """Abstract data sink."""

    @abstractmethod
    def write(self, record: DataRecord) -> bool:
        """Write single record."""
        pass

    @abstractmethod
    def write_batch(self, records: List[DataRecord]) -> int:
        """Write batch of records."""
        pass

    @abstractmethod
    def flush(self) -> None:
        """Flush pending writes."""
        pass


class InMemoryDataSource(DataSource):
    """In-memory data source."""

    def __init__(self, records: Optional[List[DataRecord]] = None) -> None:
        """Initialize source."""
        self._records = list(records) if records else []
        self._index = 0
        self._lock = threading.Lock()

    def add_record(self, record: DataRecord) -> None:
        """Add record."""
        with self._lock:
            self._records.append(record)

    def add_records(self, records: List[DataRecord]) -> None:
        """Add records."""
        with self._lock:
            self._records.extend(records)

    def read(self) -> Iterator[DataRecord]:
        """Read data records."""
        with self._lock:
            for record in self._records:
                yield record

    def read_batch(self, batch_size: int) -> List[DataRecord]:
        """Read batch of records."""
        with self._lock:
            start = self._index
            end = min(self._index + batch_size, len(self._records))
            self._index = end
            return self._records[start:end]

    def reset(self) -> None:
        """Reset read position."""
        with self._lock:
            self._index = 0

    def count(self) -> int:
        """Get record count."""
        with self._lock:
            return len(self._records)


class InMemoryDataSink(DataSink):
    """In-memory data sink."""

    def __init__(self) -> None:
        """Initialize sink."""
        self._records: List[DataRecord] = []
        self._lock = threading.Lock()

    def write(self, record: DataRecord) -> bool:
        """Write single record."""
        with self._lock:
            self._records.append(record)
            return True

    def write_batch(self, records: List[DataRecord]) -> int:
        """Write batch of records."""
        with self._lock:
            self._records.extend(records)
            return len(records)

    def flush(self) -> None:
        """Flush pending writes."""
        pass  # No buffering in memory sink

    def get_records(self) -> List[DataRecord]:
        """Get all records."""
        with self._lock:
            return list(self._records)

    def count(self) -> int:
        """Get record count."""
        with self._lock:
            return len(self._records)

    def clear(self) -> None:
        """Clear all records."""
        with self._lock:
            self._records.clear()


class Transformer(ABC):
    """Abstract transformer."""

    @abstractmethod
    def transform(self, record: DataRecord) -> Optional[DataRecord]:
        """Transform single record."""
        pass


class MapTransformer(Transformer):
    """Map transformer using a function."""

    def __init__(self, func: Callable[[Any], Any]) -> None:
        """Initialize transformer."""
        self._func = func

    def transform(self, record: DataRecord) -> Optional[DataRecord]:
        """Transform record."""
        try:
            transformed_data = self._func(record.data)
            return DataRecord(
                record_id=record.record_id,
                data=transformed_data,
                metadata=record.metadata,
                timestamp=datetime.now(),
                source=record.source,
                format=record.format,
            )
        except Exception:
            return None


class FilterTransformer(Transformer):
    """Filter transformer."""

    def __init__(self, predicate: Callable[[Any], bool]) -> None:
        """Initialize transformer."""
        self._predicate = predicate

    def transform(self, record: DataRecord) -> Optional[DataRecord]:
        """Filter record."""
        if self._predicate(record.data):
            return record
        return None


class FieldExtractTransformer(Transformer):
    """Extract specific fields from record."""

    def __init__(self, fields: List[str]) -> None:
        """Initialize transformer."""
        self._fields = fields

    def transform(self, record: DataRecord) -> Optional[DataRecord]:
        """Extract fields."""
        if not isinstance(record.data, dict):
            return None

        extracted = {k: record.data.get(k) for k in self._fields if k in record.data}
        return DataRecord(
            record_id=record.record_id,
            data=extracted,
            metadata=record.metadata,
            timestamp=datetime.now(),
            source=record.source,
            format=DataFormat.DICT,
        )


class EnrichTransformer(Transformer):
    """Enrich records with additional data."""

    def __init__(self, enrichment: Dict[str, Any]) -> None:
        """Initialize transformer."""
        self._enrichment = enrichment

    def transform(self, record: DataRecord) -> Optional[DataRecord]:
        """Enrich record."""
        if isinstance(record.data, dict):
            enriched = {**record.data, **self._enrichment}
        else:
            enriched = {"original": record.data, **self._enrichment}

        return DataRecord(
            record_id=record.record_id,
            data=enriched,
            metadata={**record.metadata, "enriched": True},
            timestamp=datetime.now(),
            source=record.source,
            format=DataFormat.DICT,
        )


@dataclass
class PipelineStage:
    """Pipeline stage."""

    stage_id: str
    stage_type: PipelineStageType
    name: str
    transformer: Optional[Transformer] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def process(self, record: DataRecord) -> Optional[DataRecord]:
        """Process record through stage."""
        if self.transformer:
            return self.transformer.transform(record)
        return record


@dataclass
class PipelineConfig:
    """Pipeline configuration."""

    pipeline_id: str
    name: str
    stages: List[PipelineStage] = field(default_factory=list)
    batch_size: int = 100
    max_retries: int = 3
    retry_delay: float = 1.0
    parallel_processing: bool = False
    max_workers: int = 4
    error_handling: str = "skip"  # skip, stop, retry


class Pipeline:
    """Data pipeline."""

    def __init__(
        self,
        config: PipelineConfig,
        source: Optional[DataSource] = None,
        sink: Optional[DataSink] = None,
    ) -> None:
        """Initialize pipeline."""
        self._config = config
        self._source = source
        self._sink = sink
        self._status = PipelineStatus.PENDING
        self._metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()

    @property
    def status(self) -> PipelineStatus:
        """Get pipeline status."""
        return self._status

    @property
    def config(self) -> PipelineConfig:
        """Get pipeline config."""
        return self._config

    def set_source(self, source: DataSource) -> None:
        """Set data source."""
        self._source = source

    def set_sink(self, sink: DataSink) -> None:
        """Set data sink."""
        self._sink = sink

    def add_stage(self, stage: PipelineStage) -> None:
        """Add pipeline stage."""
        self._config.stages.append(stage)

    def process_record(self, record: DataRecord) -> Optional[DataRecord]:
        """Process single record through all stages."""
        current = record
        for stage in self._config.stages:
            result = stage.process(current)
            if result is None:
                return None
            current = result
        return current

    def run(self) -> BatchResult:
        """Run pipeline."""
        if self._source is None:
            raise ValueError("No data source configured")

        with self._lock:
            self._status = PipelineStatus.RUNNING

        batch_id = str(uuid.uuid4())
        result = BatchResult(
            batch_id=batch_id,
            total_records=0,
            processed_records=0,
            failed_records=0,
            start_time=datetime.now(),
        )

        try:
            for record in self._source.read():
                result.total_records += 1

                try:
                    processed = self.process_record(record)
                    if processed is not None:
                        if self._sink:
                            self._sink.write(processed)
                        result.processed_records += 1
                    else:
                        result.failed_records += 1

                except Exception as e:
                    result.failed_records += 1
                    result.errors.append(
                        {
                            "record_id": record.record_id,
                            "error": str(e),
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

                    if self._config.error_handling == "stop":
                        break

            if self._sink:
                self._sink.flush()

            with self._lock:
                self._status = PipelineStatus.COMPLETED

        except Exception as e:
            with self._lock:
                self._status = PipelineStatus.FAILED
            result.errors.append(
                {
                    "error": str(e),
                    "type": "pipeline_error",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        result.end_time = datetime.now()
        return result

    async def run_async(self) -> BatchResult:
        """Run pipeline asynchronously."""
        return await asyncio.get_event_loop().run_in_executor(None, self.run)

    def run_batch(self, batch_size: Optional[int] = None) -> BatchResult:
        """Run pipeline in batches."""
        if self._source is None:
            raise ValueError("No data source configured")

        batch_size = batch_size or self._config.batch_size

        with self._lock:
            self._status = PipelineStatus.RUNNING

        batch_id = str(uuid.uuid4())
        result = BatchResult(
            batch_id=batch_id,
            total_records=0,
            processed_records=0,
            failed_records=0,
            start_time=datetime.now(),
        )

        try:
            while True:
                batch = self._source.read_batch(batch_size)
                if not batch:
                    break

                for record in batch:
                    result.total_records += 1

                    try:
                        processed = self.process_record(record)
                        if processed is not None:
                            if self._sink:
                                self._sink.write(processed)
                            result.processed_records += 1
                        else:
                            result.failed_records += 1

                    except Exception as e:
                        result.failed_records += 1
                        result.errors.append(
                            {
                                "record_id": record.record_id,
                                "error": str(e),
                            }
                        )

            if self._sink:
                self._sink.flush()

            with self._lock:
                self._status = PipelineStatus.COMPLETED

        except Exception as e:
            with self._lock:
                self._status = PipelineStatus.FAILED
            result.errors.append({"error": str(e), "type": "pipeline_error"})

        result.end_time = datetime.now()
        return result

    def cancel(self) -> None:
        """Cancel pipeline."""
        with self._lock:
            self._status = PipelineStatus.CANCELLED

    def pause(self) -> None:
        """Pause pipeline."""
        with self._lock:
            self._status = PipelineStatus.PAUSED

    def resume(self) -> None:
        """Resume pipeline."""
        with self._lock:
            if self._status == PipelineStatus.PAUSED:
                self._status = PipelineStatus.RUNNING


class PipelineBuilder:
    """Pipeline builder."""

    def __init__(self, name: str) -> None:
        """Initialize builder."""
        self._pipeline_id = str(uuid.uuid4())
        self._name = name
        self._stages: List[PipelineStage] = []
        self._source: Optional[DataSource] = None
        self._sink: Optional[DataSink] = None
        self._batch_size = 100
        self._max_retries = 3
        self._error_handling = "skip"

    def from_source(self, source: DataSource) -> "PipelineBuilder":
        """Set source."""
        self._source = source
        return self

    def to_sink(self, sink: DataSink) -> "PipelineBuilder":
        """Set sink."""
        self._sink = sink
        return self

    def add_stage(
        self,
        name: str,
        stage_type: PipelineStageType,
        transformer: Optional[Transformer] = None,
    ) -> "PipelineBuilder":
        """Add stage."""
        stage = PipelineStage(
            stage_id=str(uuid.uuid4()),
            stage_type=stage_type,
            name=name,
            transformer=transformer,
        )
        self._stages.append(stage)
        return self

    def map(self, func: Callable[[Any], Any], name: str = "map") -> "PipelineBuilder":
        """Add map transformation."""
        return self.add_stage(
            name=name,
            stage_type=PipelineStageType.TRANSFORM,
            transformer=MapTransformer(func),
        )

    def filter(self, predicate: Callable[[Any], bool], name: str = "filter") -> "PipelineBuilder":
        """Add filter transformation."""
        return self.add_stage(
            name=name,
            stage_type=PipelineStageType.FILTER,
            transformer=FilterTransformer(predicate),
        )

    def extract_fields(self, fields: List[str], name: str = "extract") -> "PipelineBuilder":
        """Add field extraction."""
        return self.add_stage(
            name=name,
            stage_type=PipelineStageType.TRANSFORM,
            transformer=FieldExtractTransformer(fields),
        )

    def enrich(self, enrichment: Dict[str, Any], name: str = "enrich") -> "PipelineBuilder":
        """Add enrichment."""
        return self.add_stage(
            name=name,
            stage_type=PipelineStageType.ENRICH,
            transformer=EnrichTransformer(enrichment),
        )

    def with_batch_size(self, batch_size: int) -> "PipelineBuilder":
        """Set batch size."""
        self._batch_size = batch_size
        return self

    def with_retries(self, max_retries: int) -> "PipelineBuilder":
        """Set max retries."""
        self._max_retries = max_retries
        return self

    def on_error(self, handling: str) -> "PipelineBuilder":
        """Set error handling strategy."""
        self._error_handling = handling
        return self

    def build(self) -> Pipeline:
        """Build pipeline."""
        config = PipelineConfig(
            pipeline_id=self._pipeline_id,
            name=self._name,
            stages=self._stages,
            batch_size=self._batch_size,
            max_retries=self._max_retries,
            error_handling=self._error_handling,
        )
        return Pipeline(config, self._source, self._sink)


class ETLPipeline:
    """ETL (Extract, Transform, Load) pipeline."""

    def __init__(
        self,
        name: str,
        extractor: Optional[Callable[[], Iterator[Any]]] = None,
        transformer: Optional[Callable[[Any], Any]] = None,
        loader: Optional[Callable[[Any], bool]] = None,
    ) -> None:
        """Initialize ETL pipeline."""
        self._name = name
        self._extractor = extractor
        self._transformer = transformer
        self._loader = loader
        self._status = PipelineStatus.PENDING

    def set_extractor(self, extractor: Callable[[], Iterator[Any]]) -> None:
        """Set extractor."""
        self._extractor = extractor

    def set_transformer(self, transformer: Callable[[Any], Any]) -> None:
        """Set transformer."""
        self._transformer = transformer

    def set_loader(self, loader: Callable[[Any], bool]) -> None:
        """Set loader."""
        self._loader = loader

    def run(self) -> BatchResult:
        """Run ETL pipeline."""
        if self._extractor is None:
            raise ValueError("No extractor configured")

        self._status = PipelineStatus.RUNNING
        batch_id = str(uuid.uuid4())

        result = BatchResult(
            batch_id=batch_id,
            total_records=0,
            processed_records=0,
            failed_records=0,
            start_time=datetime.now(),
        )

        try:
            for data in self._extractor():
                result.total_records += 1

                try:
                    # Transform
                    if self._transformer:
                        transformed = self._transformer(data)
                    else:
                        transformed = data

                    # Load
                    if self._loader:
                        success = self._loader(transformed)
                        if success:
                            result.processed_records += 1
                        else:
                            result.failed_records += 1
                    else:
                        result.processed_records += 1

                except Exception as e:
                    result.failed_records += 1
                    result.errors.append(
                        {
                            "error": str(e),
                            "data": str(data)[:100],
                        }
                    )

            self._status = PipelineStatus.COMPLETED

        except Exception as e:
            self._status = PipelineStatus.FAILED
            result.errors.append({"error": str(e), "type": "etl_error"})

        result.end_time = datetime.now()
        return result


class BatchProcessor:
    """Batch processor for processing data in batches."""

    def __init__(
        self,
        processor: Callable[[List[Any]], List[Any]],
        batch_size: int = 100,
    ) -> None:
        """Initialize batch processor."""
        self._processor = processor
        self._batch_size = batch_size
        self._buffer: List[Any] = []
        self._lock = threading.Lock()

    def add(self, item: Any) -> Optional[List[Any]]:
        """Add item to batch."""
        with self._lock:
            self._buffer.append(item)
            if len(self._buffer) >= self._batch_size:
                batch = list(self._buffer)
                self._buffer.clear()
                return self._processor(batch)
            return None

    def flush(self) -> Optional[List[Any]]:
        """Flush remaining items."""
        with self._lock:
            if self._buffer:
                batch = list(self._buffer)
                self._buffer.clear()
                return self._processor(batch)
            return None

    def process_all(self, items: List[Any]) -> List[Any]:
        """Process all items in batches."""
        results = []
        for i in range(0, len(items), self._batch_size):
            batch = items[i : i + self._batch_size]
            batch_results = self._processor(batch)
            results.extend(batch_results)
        return results


class VisionPipelineProvider(VisionProvider):
    """Vision provider with data pipeline integration."""

    def __init__(
        self,
        provider: VisionProvider,
        pre_pipeline: Optional[Pipeline] = None,
        post_pipeline: Optional[Pipeline] = None,
    ) -> None:
        """Initialize provider."""
        self._provider = provider
        self._pre_pipeline = pre_pipeline
        self._post_pipeline = post_pipeline
        self._results_sink = InMemoryDataSink()

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"pipeline_{self._provider.provider_name}"

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image with pipeline processing."""
        request_id = str(uuid.uuid4())

        # Pre-process through pipeline if configured
        if self._pre_pipeline:
            record = DataRecord(
                record_id=request_id,
                data={"image_data": image_data, "include_description": include_description},
                source="input",
            )
            source = InMemoryDataSource([record])
            self._pre_pipeline.set_source(source)
            self._pre_pipeline.run()

        # Analyze
        result = await self._provider.analyze_image(image_data, include_description)

        # Post-process through pipeline if configured
        if self._post_pipeline:
            result_record = DataRecord(
                record_id=request_id,
                data={
                    "summary": result.summary,
                    "details": result.details,
                    "confidence": result.confidence,
                },
                source="output",
            )
            source = InMemoryDataSource([result_record])
            self._post_pipeline.set_source(source)
            self._post_pipeline.set_sink(self._results_sink)
            self._post_pipeline.run()

        return result

    def get_results(self) -> List[DataRecord]:
        """Get processed results."""
        return self._results_sink.get_records()


def create_pipeline(name: str) -> PipelineBuilder:
    """Create pipeline builder.

    Args:
        name: Pipeline name

    Returns:
        Pipeline builder
    """
    return PipelineBuilder(name)


def create_etl_pipeline(
    name: str,
    extractor: Optional[Callable[[], Iterator[Any]]] = None,
    transformer: Optional[Callable[[Any], Any]] = None,
    loader: Optional[Callable[[Any], bool]] = None,
) -> ETLPipeline:
    """Create ETL pipeline.

    Args:
        name: Pipeline name
        extractor: Extractor function
        transformer: Transformer function
        loader: Loader function

    Returns:
        ETL pipeline
    """
    return ETLPipeline(name, extractor, transformer, loader)


def create_batch_processor(
    processor: Callable[[List[Any]], List[Any]],
    batch_size: int = 100,
) -> BatchProcessor:
    """Create batch processor.

    Args:
        processor: Batch processor function
        batch_size: Batch size

    Returns:
        Batch processor
    """
    return BatchProcessor(processor, batch_size)
