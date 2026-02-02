"""ETL Pipeline.

Provides end-to-end ETL pipeline execution:
- Pipeline definition
- Execution orchestration
- Error handling
- Metrics collection
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from src.core.etl.sources import DataSource, Record, SourceConfig
from src.core.etl.transforms import Transform, TransformResult
from src.core.etl.sinks import DataSink, SinkConfig, WriteResult

logger = logging.getLogger(__name__)


class PipelineStatus(str, Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineMetrics:
    """Metrics for pipeline execution."""
    records_read: int = 0
    records_transformed: int = 0
    records_written: int = 0
    records_dropped: int = 0
    records_failed: int = 0
    batches_processed: int = 0
    transform_errors: int = 0
    write_errors: int = 0


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    pipeline_id: str
    status: PipelineStatus
    metrics: PipelineMetrics
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    error: Optional[str] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    name: str
    batch_size: int = 1000
    max_errors: int = 100  # Stop after this many errors
    continue_on_error: bool = True
    parallel_transforms: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class Pipeline:
    """ETL Pipeline for data processing."""

    def __init__(
        self,
        source: DataSource,
        transforms: List[Transform],
        sink: DataSink,
        config: Optional[PipelineConfig] = None,
    ):
        """Initialize the pipeline.

        Args:
            source: Data source.
            transforms: List of transforms to apply.
            sink: Data sink.
            config: Pipeline configuration.
        """
        self.source = source
        self.transforms = transforms
        self.sink = sink
        self.config = config or PipelineConfig(name="default")

        self._pipeline_id = str(uuid.uuid4())
        self._status = PipelineStatus.PENDING
        self._metrics = PipelineMetrics()
        self._cancel_requested = False

        # Callbacks
        self._on_batch: Optional[Callable[[int, int], None]] = None
        self._on_error: Optional[Callable[[Dict[str, Any]], None]] = None

    @property
    def pipeline_id(self) -> str:
        return self._pipeline_id

    @property
    def status(self) -> PipelineStatus:
        return self._status

    @property
    def metrics(self) -> PipelineMetrics:
        return self._metrics

    def on_batch(self, callback: Callable[[int, int], None]) -> "Pipeline":
        """Register batch completion callback."""
        self._on_batch = callback
        return self

    def on_error(self, callback: Callable[[Dict[str, Any]], None]) -> "Pipeline":
        """Register error callback."""
        self._on_error = callback
        return self

    async def run(self) -> PipelineResult:
        """Execute the pipeline.

        Returns:
            PipelineResult with execution details.
        """
        started_at = datetime.utcnow()
        self._status = PipelineStatus.RUNNING
        self._metrics = PipelineMetrics()
        all_errors = []

        try:
            async with self.source, self.sink:
                await self._process_batches(all_errors)

            if self._cancel_requested:
                self._status = PipelineStatus.CANCELLED
            else:
                self._status = PipelineStatus.COMPLETED

        except Exception as e:
            logger.exception(f"Pipeline {self._pipeline_id} failed: {e}")
            self._status = PipelineStatus.FAILED
            all_errors.append({"type": "pipeline", "error": str(e)})

        completed_at = datetime.utcnow()
        duration_ms = (completed_at - started_at).total_seconds() * 1000

        return PipelineResult(
            pipeline_id=self._pipeline_id,
            status=self._status,
            metrics=self._metrics,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            error=all_errors[-1]["error"] if all_errors else None,
            errors=all_errors,
        )

    async def _process_batches(self, all_errors: List[Dict[str, Any]]) -> None:
        """Process all batches from source."""
        batch_num = 0

        while not self._cancel_requested:
            # Read batch
            records = await self.source.read_batch(self.config.batch_size)
            if not records:
                break

            self._metrics.records_read += len(records)
            batch_num += 1

            # Transform
            transformed = await self._apply_transforms(records)
            self._metrics.records_transformed += len(transformed.records)
            self._metrics.records_dropped += transformed.dropped
            self._metrics.transform_errors += len(transformed.errors)
            all_errors.extend(transformed.errors)

            # Check error limit
            if len(all_errors) > self.config.max_errors:
                logger.error(f"Pipeline exceeded max errors ({self.config.max_errors})")
                if not self.config.continue_on_error:
                    raise RuntimeError("Too many errors")
                break

            # Write
            if transformed.records:
                result = await self.sink.write(transformed.records)
                self._metrics.records_written += result.written
                self._metrics.records_failed += result.failed
                self._metrics.write_errors += len(result.errors)
                all_errors.extend(result.errors)

            self._metrics.batches_processed += 1

            # Callback
            if self._on_batch:
                self._on_batch(batch_num, len(records))

            # Report errors via callback
            if self._on_error:
                for error in transformed.errors:
                    self._on_error(error)

    async def _apply_transforms(self, records: List[Record]) -> TransformResult:
        """Apply all transforms to records."""
        current_records = records
        total_dropped = 0
        all_errors = []

        for transform in self.transforms:
            result = await transform.apply(current_records)
            current_records = result.records
            total_dropped += result.dropped
            all_errors.extend(result.errors)

            if not current_records:
                break

        return TransformResult(
            records=current_records,
            dropped=total_dropped,
            errors=all_errors,
        )

    def cancel(self) -> None:
        """Request pipeline cancellation."""
        self._cancel_requested = True


class PipelineBuilder:
    """Fluent builder for ETL pipelines."""

    def __init__(self, name: str = "default"):
        self._config = PipelineConfig(name=name)
        self._source: Optional[DataSource] = None
        self._transforms: List[Transform] = []
        self._sink: Optional[DataSink] = None

    def source(self, source: DataSource) -> "PipelineBuilder":
        """Set the data source."""
        self._source = source
        return self

    def transform(self, transform: Transform) -> "PipelineBuilder":
        """Add a transform."""
        self._transforms.append(transform)
        return self

    def transforms(self, *transforms: Transform) -> "PipelineBuilder":
        """Add multiple transforms."""
        self._transforms.extend(transforms)
        return self

    def sink(self, sink: DataSink) -> "PipelineBuilder":
        """Set the data sink."""
        self._sink = sink
        return self

    def batch_size(self, size: int) -> "PipelineBuilder":
        """Set batch size."""
        self._config.batch_size = size
        return self

    def max_errors(self, count: int) -> "PipelineBuilder":
        """Set maximum errors before stopping."""
        self._config.max_errors = count
        return self

    def continue_on_error(self, value: bool = True) -> "PipelineBuilder":
        """Set whether to continue on errors."""
        self._config.continue_on_error = value
        return self

    def build(self) -> Pipeline:
        """Build the pipeline."""
        if not self._source:
            raise ValueError("Source is required")
        if not self._sink:
            raise ValueError("Sink is required")

        return Pipeline(
            source=self._source,
            transforms=self._transforms,
            sink=self._sink,
            config=self._config,
        )


# Pipeline registry for scheduled execution
_pipelines: Dict[str, Pipeline] = {}


def register_pipeline(name: str, pipeline: Pipeline) -> None:
    """Register a pipeline for scheduled execution."""
    _pipelines[name] = pipeline


def get_pipeline(name: str) -> Optional[Pipeline]:
    """Get a registered pipeline."""
    return _pipelines.get(name)


def list_pipelines() -> List[str]:
    """List registered pipeline names."""
    return list(_pipelines.keys())
