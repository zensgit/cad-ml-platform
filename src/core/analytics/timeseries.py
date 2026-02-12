"""Time Series Data Store.

Provides time-series data storage and querying:
- Point-in-time metrics
- Time-based aggregations
- Efficient storage with downsampling
"""

from __future__ import annotations

import asyncio
import bisect
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AggregationType(str, Enum):
    """Aggregation function types."""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    FIRST = "first"
    LAST = "last"
    PERCENTILE_50 = "p50"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"


class Resolution(str, Enum):
    """Time resolution for data points."""
    SECOND = "1s"
    MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    HOUR = "1h"
    DAY = "1d"
    WEEK = "1w"

    def to_seconds(self) -> int:
        """Convert resolution to seconds."""
        mapping = {
            "1s": 1,
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "1d": 86400,
            "1w": 604800,
        }
        return mapping[self.value]


@dataclass
class DataPoint:
    """A single time-series data point."""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """A time series of data points."""
    metric_name: str
    points: List[DataPoint] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Result of a time series query."""
    series: List[MetricSeries]
    query_time_ms: float
    total_points: int


class TimeSeriesStore:
    """In-memory time series data store."""

    def __init__(
        self,
        max_points_per_series: int = 10000,
        retention_hours: int = 168,  # 7 days
    ):
        """Initialize the time series store.

        Args:
            max_points_per_series: Maximum points to keep per series.
            retention_hours: Data retention in hours.
        """
        self.max_points = max_points_per_series
        self.retention = timedelta(hours=retention_hours)

        # Storage: metric_name -> {tags_key -> [(timestamp, value)]}
        self._data: Dict[str, Dict[str, List[Tuple[datetime, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        """Lazily create lock."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _tags_key(self, tags: Dict[str, str]) -> str:
        """Create a hashable key from tags."""
        return "|".join(f"{k}={v}" for k, v in sorted(tags.items()))

    async def write(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Write a data point.

        Args:
            metric_name: Name of the metric.
            value: Metric value.
            timestamp: Point timestamp (defaults to now).
            tags: Optional tags for the series.
        """
        async with self._get_lock():
            timestamp = timestamp or datetime.utcnow()
            tags = tags or {}
            tags_key = self._tags_key(tags)

            series = self._data[metric_name][tags_key]

            # Insert in sorted order (tuple compares by first element by default)
            point = (timestamp, value)
            bisect.insort(series, point)

            # Enforce max points
            if len(series) > self.max_points:
                self._data[metric_name][tags_key] = series[-self.max_points:]

    async def write_batch(
        self,
        points: List[Tuple[str, float, Optional[datetime], Optional[Dict[str, str]]]],
    ) -> int:
        """Write multiple data points.

        Args:
            points: List of (metric_name, value, timestamp, tags) tuples.

        Returns:
            Number of points written.
        """
        count = 0
        for metric_name, value, timestamp, tags in points:
            await self.write(metric_name, value, timestamp, tags)
            count += 1
        return count

    async def query(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
        aggregation: Optional[AggregationType] = None,
        resolution: Optional[Resolution] = None,
    ) -> QueryResult:
        """Query time series data.

        Args:
            metric_name: Name of the metric.
            start_time: Query start time.
            end_time: Query end time (defaults to now).
            tags: Filter by tags.
            aggregation: Aggregation function.
            resolution: Time resolution for aggregation.

        Returns:
            QueryResult with matching series.
        """
        import time
        query_start = time.time()

        end_time = end_time or datetime.utcnow()

        async with self._get_lock():
            metric_data = self._data.get(metric_name, {})
            result_series = []
            total_points = 0

            for tags_key, points in metric_data.items():
                # Parse tags from key
                series_tags = {}
                if tags_key:
                    for pair in tags_key.split("|"):
                        if "=" in pair:
                            k, v = pair.split("=", 1)
                            series_tags[k] = v

                # Filter by tags if specified
                if tags:
                    if not all(series_tags.get(k) == v for k, v in tags.items()):
                        continue

                # Filter by time range
                filtered_points = [
                    DataPoint(timestamp=ts, value=val)
                    for ts, val in points
                    if start_time <= ts <= end_time
                ]

                if not filtered_points:
                    continue

                # Apply aggregation if specified
                if aggregation and resolution:
                    filtered_points = self._aggregate(
                        filtered_points, aggregation, resolution
                    )

                result_series.append(MetricSeries(
                    metric_name=metric_name,
                    points=filtered_points,
                    tags=series_tags,
                ))
                total_points += len(filtered_points)

        query_time_ms = (time.time() - query_start) * 1000

        return QueryResult(
            series=result_series,
            query_time_ms=query_time_ms,
            total_points=total_points,
        )

    def _aggregate(
        self,
        points: List[DataPoint],
        aggregation: AggregationType,
        resolution: Resolution,
    ) -> List[DataPoint]:
        """Aggregate points by time buckets."""
        if not points:
            return []

        bucket_seconds = resolution.to_seconds()
        buckets: Dict[datetime, List[float]] = defaultdict(list)

        for point in points:
            # Round down to bucket
            bucket_ts = datetime.fromtimestamp(
                (point.timestamp.timestamp() // bucket_seconds) * bucket_seconds
            )
            buckets[bucket_ts].append(point.value)

        result = []
        for bucket_ts in sorted(buckets.keys()):
            values = buckets[bucket_ts]
            agg_value = self._compute_aggregation(values, aggregation)
            result.append(DataPoint(timestamp=bucket_ts, value=agg_value))

        return result

    def _compute_aggregation(
        self,
        values: List[float],
        aggregation: AggregationType,
    ) -> float:
        """Compute aggregation on values."""
        if not values:
            return 0.0

        if aggregation == AggregationType.SUM:
            return sum(values)
        elif aggregation == AggregationType.AVG:
            return sum(values) / len(values)
        elif aggregation == AggregationType.MIN:
            return min(values)
        elif aggregation == AggregationType.MAX:
            return max(values)
        elif aggregation == AggregationType.COUNT:
            return float(len(values))
        elif aggregation == AggregationType.FIRST:
            return values[0]
        elif aggregation == AggregationType.LAST:
            return values[-1]
        elif aggregation == AggregationType.PERCENTILE_50:
            return self._percentile(values, 50)
        elif aggregation == AggregationType.PERCENTILE_95:
            return self._percentile(values, 95)
        elif aggregation == AggregationType.PERCENTILE_99:
            return self._percentile(values, 99)
        else:
            return sum(values) / len(values)

    def _percentile(self, values: List[float], p: int) -> float:
        """Calculate percentile."""
        sorted_values = sorted(values)
        index = (len(sorted_values) - 1) * p / 100
        lower = int(index)
        upper = lower + 1
        weight = index - lower

        if upper >= len(sorted_values):
            return sorted_values[-1]

        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

    async def delete(
        self,
        metric_name: str,
        before: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> int:
        """Delete data points.

        Args:
            metric_name: Name of the metric.
            before: Delete points before this time.
            tags: Filter by tags.

        Returns:
            Number of points deleted.
        """
        async with self._get_lock():
            deleted = 0

            if metric_name not in self._data:
                return 0

            if tags:
                tags_key = self._tags_key(tags)
                if tags_key in self._data[metric_name]:
                    if before:
                        original_len = len(self._data[metric_name][tags_key])
                        self._data[metric_name][tags_key] = [
                            (ts, val) for ts, val in self._data[metric_name][tags_key]
                            if ts >= before
                        ]
                        deleted = original_len - len(self._data[metric_name][tags_key])
                    else:
                        deleted = len(self._data[metric_name][tags_key])
                        del self._data[metric_name][tags_key]
            else:
                for tags_key, points in list(self._data[metric_name].items()):
                    if before:
                        original_len = len(points)
                        self._data[metric_name][tags_key] = [
                            (ts, val) for ts, val in points if ts >= before
                        ]
                        deleted += original_len - len(self._data[metric_name][tags_key])
                    else:
                        deleted += len(points)
                        del self._data[metric_name][tags_key]

            return deleted

    async def get_metrics(self) -> List[str]:
        """Get list of all metric names."""
        async with self._get_lock():
            return list(self._data.keys())

    async def get_tags(self, metric_name: str) -> List[Dict[str, str]]:
        """Get all tag combinations for a metric."""
        async with self._get_lock():
            result = []
            for tags_key in self._data.get(metric_name, {}).keys():
                tags = {}
                if tags_key:
                    for pair in tags_key.split("|"):
                        if "=" in pair:
                            k, v = pair.split("=", 1)
                            tags[k] = v
                result.append(tags)
            return result

    async def cleanup_old_data(self) -> int:
        """Remove data older than retention period."""
        cutoff = datetime.utcnow() - self.retention
        deleted = 0

        async with self._get_lock():
            for metric_name in list(self._data.keys()):
                for tags_key in list(self._data[metric_name].keys()):
                    original_len = len(self._data[metric_name][tags_key])
                    self._data[metric_name][tags_key] = [
                        (ts, val) for ts, val in self._data[metric_name][tags_key]
                        if ts >= cutoff
                    ]
                    deleted += original_len - len(self._data[metric_name][tags_key])

                    # Remove empty series
                    if not self._data[metric_name][tags_key]:
                        del self._data[metric_name][tags_key]

                # Remove empty metrics
                if not self._data[metric_name]:
                    del self._data[metric_name]

        return deleted


# Global time series store
_ts_store: Optional[TimeSeriesStore] = None


def get_timeseries_store() -> TimeSeriesStore:
    """Get the global time series store."""
    global _ts_store
    if _ts_store is None:
        _ts_store = TimeSeriesStore()
    return _ts_store
