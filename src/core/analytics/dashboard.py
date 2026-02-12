"""Dashboard Data Provider.

Provides data aggregation for dashboards:
- Widget data queries
- Real-time updates
- Cached aggregations
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from src.core.analytics.timeseries import (
    TimeSeriesStore,
    AggregationType,
    Resolution,
    QueryResult,
    get_timeseries_store,
)
from src.core.analytics.metrics import MetricRegistry, get_registry

logger = logging.getLogger(__name__)


class WidgetType(str, Enum):
    """Dashboard widget types."""
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    GAUGE = "gauge"
    COUNTER = "counter"
    TABLE = "table"
    STAT = "stat"
    HEATMAP = "heatmap"


class TimeRange(str, Enum):
    """Predefined time ranges."""
    LAST_5M = "5m"
    LAST_15M = "15m"
    LAST_1H = "1h"
    LAST_3H = "3h"
    LAST_6H = "6h"
    LAST_12H = "12h"
    LAST_24H = "24h"
    LAST_7D = "7d"
    LAST_30D = "30d"

    def to_timedelta(self) -> timedelta:
        """Convert to timedelta."""
        mapping = {
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "3h": timedelta(hours=3),
            "6h": timedelta(hours=6),
            "12h": timedelta(hours=12),
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
        }
        return mapping[self.value]

    def suggested_resolution(self) -> Resolution:
        """Get suggested resolution for time range."""
        mapping = {
            "5m": Resolution.SECOND,
            "15m": Resolution.SECOND,
            "1h": Resolution.MINUTE,
            "3h": Resolution.MINUTE,
            "6h": Resolution.FIVE_MINUTES,
            "12h": Resolution.FIVE_MINUTES,
            "24h": Resolution.FIFTEEN_MINUTES,
            "7d": Resolution.HOUR,
            "30d": Resolution.HOUR,
        }
        return mapping[self.value]


@dataclass
class WidgetConfig:
    """Configuration for a dashboard widget."""
    widget_id: str
    widget_type: WidgetType
    title: str
    metric_name: str
    aggregation: AggregationType = AggregationType.AVG
    tags: Dict[str, str] = field(default_factory=dict)
    group_by: Optional[List[str]] = None
    thresholds: Optional[Dict[str, float]] = None
    unit: str = ""
    decimals: int = 2
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WidgetData:
    """Data for rendering a widget."""
    widget_id: str
    widget_type: WidgetType
    title: str
    data: Any  # Type depends on widget_type
    unit: str = ""
    thresholds: Optional[Dict[str, float]] = None
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ChartData:
    """Data for chart widgets."""
    labels: List[str]  # X-axis labels (timestamps)
    datasets: List[Dict[str, Any]]  # {label, data, color, ...}


@dataclass
class StatData:
    """Data for stat widgets."""
    value: float
    previous_value: Optional[float] = None
    change_percent: Optional[float] = None
    trend: Optional[str] = None  # up, down, flat


@dataclass
class TableData:
    """Data for table widgets."""
    columns: List[Dict[str, str]]  # {key, label}
    rows: List[Dict[str, Any]]


class DashboardProvider:
    """Provides data for dashboard widgets."""

    def __init__(
        self,
        ts_store: Optional[TimeSeriesStore] = None,
        registry: Optional[MetricRegistry] = None,
    ):
        self.ts_store = ts_store or get_timeseries_store()
        self.registry = registry or get_registry()
        self._widgets: Dict[str, WidgetConfig] = {}
        self._cache: Dict[str, WidgetData] = {}
        self._cache_ttl = timedelta(seconds=30)

    def register_widget(self, config: WidgetConfig) -> None:
        """Register a widget configuration."""
        self._widgets[config.widget_id] = config

    def get_widget_config(self, widget_id: str) -> Optional[WidgetConfig]:
        """Get widget configuration."""
        return self._widgets.get(widget_id)

    async def get_widget_data(
        self,
        widget_id: str,
        time_range: TimeRange = TimeRange.LAST_1H,
        force_refresh: bool = False,
    ) -> Optional[WidgetData]:
        """Get data for a widget.

        Args:
            widget_id: Widget identifier.
            time_range: Time range for data.
            force_refresh: Skip cache.

        Returns:
            WidgetData or None if widget not found.
        """
        config = self._widgets.get(widget_id)
        if not config:
            return None

        # Check cache
        cache_key = f"{widget_id}:{time_range.value}"
        if not force_refresh and cache_key in self._cache:
            cached = self._cache[cache_key]
            if datetime.utcnow() - cached.updated_at < self._cache_ttl:
                return cached

        # Fetch fresh data
        data = await self._fetch_widget_data(config, time_range)
        self._cache[cache_key] = data
        return data

    async def _fetch_widget_data(
        self,
        config: WidgetConfig,
        time_range: TimeRange,
    ) -> WidgetData:
        """Fetch data for a widget."""
        start_time = datetime.utcnow() - time_range.to_timedelta()
        resolution = time_range.suggested_resolution()

        result = await self.ts_store.query(
            metric_name=config.metric_name,
            start_time=start_time,
            tags=config.tags,
            aggregation=config.aggregation,
            resolution=resolution,
        )

        # Format data based on widget type
        if config.widget_type in (WidgetType.LINE_CHART, WidgetType.BAR_CHART):
            data = self._format_chart_data(result, config)
        elif config.widget_type == WidgetType.STAT:
            data = self._format_stat_data(result, config)
        elif config.widget_type == WidgetType.GAUGE:
            data = self._format_gauge_data(result, config)
        elif config.widget_type == WidgetType.COUNTER:
            data = self._format_counter_data(result, config)
        elif config.widget_type == WidgetType.TABLE:
            data = self._format_table_data(result, config)
        else:
            data = result

        return WidgetData(
            widget_id=config.widget_id,
            widget_type=config.widget_type,
            title=config.title,
            data=data,
            unit=config.unit,
            thresholds=config.thresholds,
        )

    def _format_chart_data(
        self,
        result: QueryResult,
        config: WidgetConfig,
    ) -> ChartData:
        """Format data for chart widgets."""
        datasets = []

        for series in result.series:
            label = series.metric_name
            if series.tags:
                label += f" ({', '.join(f'{k}={v}' for k, v in series.tags.items())})"

            data = [p.value for p in series.points]
            labels = [p.timestamp.isoformat() for p in series.points]

            datasets.append({
                "label": label,
                "data": data,
            })

        # Use labels from first series
        all_labels = []
        if result.series and result.series[0].points:
            all_labels = [p.timestamp.isoformat() for p in result.series[0].points]

        return ChartData(labels=all_labels, datasets=datasets)

    def _format_stat_data(
        self,
        result: QueryResult,
        config: WidgetConfig,
    ) -> StatData:
        """Format data for stat widgets."""
        if not result.series or not result.series[0].points:
            return StatData(value=0)

        points = result.series[0].points
        current = points[-1].value if points else 0

        # Calculate previous value and change
        previous = None
        change = None
        trend = None

        if len(points) > 1:
            mid_point = len(points) // 2
            previous = sum(p.value for p in points[:mid_point]) / mid_point
            if previous != 0:
                change = ((current - previous) / previous) * 100
                if change > 1:
                    trend = "up"
                elif change < -1:
                    trend = "down"
                else:
                    trend = "flat"

        return StatData(
            value=round(current, config.decimals),
            previous_value=round(previous, config.decimals) if previous else None,
            change_percent=round(change, 1) if change else None,
            trend=trend,
        )

    def _format_gauge_data(
        self,
        result: QueryResult,
        config: WidgetConfig,
    ) -> Dict[str, Any]:
        """Format data for gauge widgets."""
        if not result.series or not result.series[0].points:
            return {"value": 0, "min": 0, "max": 100}

        value = result.series[0].points[-1].value

        return {
            "value": round(value, config.decimals),
            "min": config.thresholds.get("min", 0) if config.thresholds else 0,
            "max": config.thresholds.get("max", 100) if config.thresholds else 100,
        }

    def _format_counter_data(
        self,
        result: QueryResult,
        config: WidgetConfig,
    ) -> Dict[str, Any]:
        """Format data for counter widgets."""
        if not result.series or not result.series[0].points:
            return {"value": 0}

        # Sum all values for counter
        total = sum(p.value for s in result.series for p in s.points)
        return {"value": round(total, config.decimals)}

    def _format_table_data(
        self,
        result: QueryResult,
        config: WidgetConfig,
    ) -> TableData:
        """Format data for table widgets."""
        columns = [
            {"key": "timestamp", "label": "Time"},
            {"key": "value", "label": "Value"},
        ]

        rows = []
        for series in result.series:
            for point in series.points:
                row = {
                    "timestamp": point.timestamp.isoformat(),
                    "value": round(point.value, config.decimals),
                }
                row.update(series.tags)
                rows.append(row)

        return TableData(columns=columns, rows=rows)

    async def get_dashboard(
        self,
        widget_ids: List[str],
        time_range: TimeRange = TimeRange.LAST_1H,
    ) -> Dict[str, WidgetData]:
        """Get data for multiple widgets.

        Args:
            widget_ids: List of widget IDs.
            time_range: Time range for data.

        Returns:
            Dict mapping widget ID to WidgetData.
        """
        results = {}

        # Fetch all widgets in parallel
        tasks = [
            self.get_widget_data(widget_id, time_range)
            for widget_id in widget_ids
        ]
        data_list = await asyncio.gather(*tasks)

        for widget_id, data in zip(widget_ids, data_list):
            if data:
                results[widget_id] = data

        return results


# Global dashboard provider
_provider: Optional[DashboardProvider] = None


def get_dashboard_provider() -> DashboardProvider:
    """Get the global dashboard provider."""
    global _provider
    if _provider is None:
        _provider = DashboardProvider()
    return _provider
