"""Analytics Module.

Provides complete analytics capabilities:
- Time-series data storage and querying
- Metrics aggregation (counters, gauges, histograms)
- Dashboard data providers
"""

from src.core.analytics.timeseries import (
    AggregationType,
    Resolution,
    DataPoint,
    MetricSeries,
    QueryResult,
    TimeSeriesStore,
    get_timeseries_store,
)
from src.core.analytics.metrics import (
    MetricType,
    MetricValue,
    HistogramBucket,
    HistogramValue,
    Metric,
    Counter,
    Gauge,
    Histogram,
    MetricRegistry,
    get_registry,
)
from src.core.analytics.dashboard import (
    WidgetType,
    TimeRange,
    WidgetConfig,
    WidgetData,
    ChartData,
    StatData,
    TableData,
    DashboardProvider,
    get_dashboard_provider,
)

__all__ = [
    # Time Series
    "AggregationType",
    "Resolution",
    "DataPoint",
    "MetricSeries",
    "QueryResult",
    "TimeSeriesStore",
    "get_timeseries_store",
    # Metrics
    "MetricType",
    "MetricValue",
    "HistogramBucket",
    "HistogramValue",
    "Metric",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricRegistry",
    "get_registry",
    # Dashboard
    "WidgetType",
    "TimeRange",
    "WidgetConfig",
    "WidgetData",
    "ChartData",
    "StatData",
    "TableData",
    "DashboardProvider",
    "get_dashboard_provider",
]
