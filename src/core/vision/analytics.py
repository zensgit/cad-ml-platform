"""Analytics and reporting for vision analysis.

Provides:
- Performance analytics
- Usage statistics
- Trend analysis
- Report generation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from .persistence import ResultPersistence

logger = logging.getLogger(__name__)


class TimeGranularity(Enum):
    """Time granularity for aggregations."""

    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass
class ProviderStats:
    """Statistics for a single provider."""

    provider: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_cost_usd: float
    total_processing_time_ms: float
    avg_confidence: float
    min_confidence: float
    max_confidence: float

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    @property
    def avg_processing_time_ms(self) -> float:
        """Calculate average processing time."""
        if self.total_requests == 0:
            return 0.0
        return self.total_processing_time_ms / self.total_requests

    @property
    def avg_cost_per_request(self) -> float:
        """Calculate average cost per request."""
        if self.total_requests == 0:
            return 0.0
        return self.total_cost_usd / self.total_requests


@dataclass
class TimeBucket:
    """A time bucket for aggregated data."""

    start_time: datetime
    end_time: datetime
    request_count: int
    total_cost_usd: float
    avg_confidence: float
    avg_processing_time_ms: float


@dataclass
class TrendData:
    """Trend data over time."""

    granularity: TimeGranularity
    buckets: List[TimeBucket]
    total_requests: int
    total_cost_usd: float

    @property
    def request_trend(self) -> List[int]:
        """Get request count trend."""
        return [b.request_count for b in self.buckets]

    @property
    def cost_trend(self) -> List[float]:
        """Get cost trend."""
        return [b.total_cost_usd for b in self.buckets]

    @property
    def confidence_trend(self) -> List[float]:
        """Get confidence trend."""
        return [b.avg_confidence for b in self.buckets]


@dataclass
class AnalyticsReport:
    """Comprehensive analytics report."""

    report_id: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    provider_stats: Dict[str, ProviderStats]
    overall_stats: ProviderStats
    trends: Optional[TrendData]
    top_tags: List[tuple[str, int]]
    insights: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
            },
            "overall": {
                "total_requests": self.overall_stats.total_requests,
                "success_rate": self.overall_stats.success_rate,
                "total_cost_usd": self.overall_stats.total_cost_usd,
                "avg_confidence": self.overall_stats.avg_confidence,
                "avg_processing_time_ms": self.overall_stats.avg_processing_time_ms,
            },
            "by_provider": {
                name: {
                    "total_requests": stats.total_requests,
                    "success_rate": stats.success_rate,
                    "total_cost_usd": stats.total_cost_usd,
                    "avg_confidence": stats.avg_confidence,
                }
                for name, stats in self.provider_stats.items()
            },
            "top_tags": self.top_tags,
            "insights": self.insights,
        }


class VisionAnalytics:
    """
    Analytics engine for vision analysis data.

    Features:
    - Performance metrics by provider
    - Usage trends over time
    - Cost analysis
    - Automated insights
    """

    def __init__(self, persistence: ResultPersistence):
        """
        Initialize analytics engine.

        Args:
            persistence: ResultPersistence instance for data access
        """
        self._persistence = persistence

    async def get_provider_stats(
        self,
        provider: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, ProviderStats]:
        """
        Get statistics by provider.

        Args:
            provider: Optional specific provider
            start_date: Period start
            end_date: Period end

        Returns:
            Dictionary of provider name to ProviderStats
        """
        result = await self._persistence.query_results(
            provider=provider,
            start_date=start_date,
            end_date=end_date,
            limit=10000,  # Get all for stats
        )

        # Group by provider
        by_provider: Dict[str, List[Any]] = {}
        for record in result.records:
            if record.provider not in by_provider:
                by_provider[record.provider] = []
            by_provider[record.provider].append(record)

        # Calculate stats
        stats: Dict[str, ProviderStats] = {}
        for prov, records in by_provider.items():
            confidences = [r.result.confidence for r in records]
            stats[prov] = ProviderStats(
                provider=prov,
                total_requests=len(records),
                successful_requests=len(records),  # All saved are successful
                failed_requests=0,
                total_cost_usd=sum(r.cost_usd for r in records),
                total_processing_time_ms=sum(r.processing_time_ms for r in records),
                avg_confidence=sum(confidences) / len(confidences) if confidences else 0,
                min_confidence=min(confidences) if confidences else 0,
                max_confidence=max(confidences) if confidences else 0,
            )

        return stats

    async def get_trends(
        self,
        granularity: TimeGranularity = TimeGranularity.DAY,
        provider: Optional[str] = None,
        days: int = 30,
    ) -> TrendData:
        """
        Get usage trends over time.

        Args:
            granularity: Time bucket granularity
            provider: Optional provider filter
            days: Number of days to analyze

        Returns:
            TrendData with time series
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        result = await self._persistence.query_results(
            provider=provider,
            start_date=start_date,
            end_date=end_date,
            limit=10000,
        )

        # Determine bucket duration
        if granularity == TimeGranularity.HOUR:
            bucket_delta = timedelta(hours=1)
        elif granularity == TimeGranularity.DAY:
            bucket_delta = timedelta(days=1)
        elif granularity == TimeGranularity.WEEK:
            bucket_delta = timedelta(weeks=1)
        else:  # MONTH
            bucket_delta = timedelta(days=30)

        # Create buckets
        buckets: List[TimeBucket] = []
        current = start_date
        while current < end_date:
            bucket_end = min(current + bucket_delta, end_date)

            # Filter records for this bucket
            bucket_records = [
                r for r in result.records
                if current <= r.created_at < bucket_end
            ]

            if bucket_records:
                confidences = [r.result.confidence for r in bucket_records]
                times = [r.processing_time_ms for r in bucket_records]
                bucket = TimeBucket(
                    start_time=current,
                    end_time=bucket_end,
                    request_count=len(bucket_records),
                    total_cost_usd=sum(r.cost_usd for r in bucket_records),
                    avg_confidence=sum(confidences) / len(confidences),
                    avg_processing_time_ms=sum(times) / len(times),
                )
            else:
                bucket = TimeBucket(
                    start_time=current,
                    end_time=bucket_end,
                    request_count=0,
                    total_cost_usd=0.0,
                    avg_confidence=0.0,
                    avg_processing_time_ms=0.0,
                )

            buckets.append(bucket)
            current = bucket_end

        return TrendData(
            granularity=granularity,
            buckets=buckets,
            total_requests=sum(b.request_count for b in buckets),
            total_cost_usd=sum(b.total_cost_usd for b in buckets),
        )

    async def get_top_tags(
        self,
        limit: int = 10,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[tuple[str, int]]:
        """
        Get most used tags.

        Args:
            limit: Number of top tags to return
            start_date: Period start
            end_date: Period end

        Returns:
            List of (tag, count) tuples
        """
        result = await self._persistence.query_results(
            start_date=start_date,
            end_date=end_date,
            limit=10000,
        )

        # Count tags
        tag_counts: Dict[str, int] = {}
        for record in result.records:
            for tag in record.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Sort and return top
        sorted_tags = sorted(
            tag_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_tags[:limit]

    async def generate_insights(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[str]:
        """
        Generate automated insights from the data.

        Args:
            start_date: Period start
            end_date: Period end

        Returns:
            List of insight strings
        """
        insights: List[str] = []

        # Get provider stats
        stats = await self.get_provider_stats(
            start_date=start_date,
            end_date=end_date,
        )

        if not stats:
            return ["No data available for the selected period."]

        # Find best provider by confidence
        if stats:
            best_confidence = max(stats.values(), key=lambda s: s.avg_confidence)
            insights.append(
                f"{best_confidence.provider} has the highest average confidence "
                f"({best_confidence.avg_confidence:.2f})"
            )

        # Find fastest provider
        if stats:
            fastest = min(stats.values(), key=lambda s: s.avg_processing_time_ms)
            insights.append(
                f"{fastest.provider} is the fastest with avg "
                f"{fastest.avg_processing_time_ms:.0f}ms per request"
            )

        # Find most cost-effective
        if stats:
            cheapest = min(stats.values(), key=lambda s: s.avg_cost_per_request)
            if cheapest.avg_cost_per_request > 0:
                insights.append(
                    f"{cheapest.provider} is most cost-effective at "
                    f"${cheapest.avg_cost_per_request:.4f} per request"
                )

        # Usage distribution
        total_requests = sum(s.total_requests for s in stats.values())
        if total_requests > 0:
            for name, s in stats.items():
                pct = (s.total_requests / total_requests) * 100
                if pct > 50:
                    insights.append(
                        f"{name} handles {pct:.0f}% of all requests"
                    )

        return insights

    async def generate_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_trends: bool = True,
    ) -> AnalyticsReport:
        """
        Generate a comprehensive analytics report.

        Args:
            start_date: Report period start
            end_date: Report period end
            include_trends: Whether to include trend data

        Returns:
            AnalyticsReport with all analytics
        """
        import uuid

        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=30))

        # Get provider stats
        provider_stats = await self.get_provider_stats(
            start_date=start_date,
            end_date=end_date,
        )

        # Calculate overall stats
        if provider_stats:
            all_stats = list(provider_stats.values())
            total_requests = sum(s.total_requests for s in all_stats)
            total_cost = sum(s.total_cost_usd for s in all_stats)
            total_time = sum(s.total_processing_time_ms for s in all_stats)

            # Weighted average confidence
            weighted_confidence = sum(
                s.avg_confidence * s.total_requests for s in all_stats
            )
            avg_confidence = (
                weighted_confidence / total_requests if total_requests > 0 else 0
            )

            overall_stats = ProviderStats(
                provider="all",
                total_requests=total_requests,
                successful_requests=total_requests,
                failed_requests=0,
                total_cost_usd=total_cost,
                total_processing_time_ms=total_time,
                avg_confidence=avg_confidence,
                min_confidence=min(s.min_confidence for s in all_stats),
                max_confidence=max(s.max_confidence for s in all_stats),
            )
        else:
            overall_stats = ProviderStats(
                provider="all",
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                total_cost_usd=0,
                total_processing_time_ms=0,
                avg_confidence=0,
                min_confidence=0,
                max_confidence=0,
            )

        # Get trends
        trends = None
        if include_trends:
            days = (end_date - start_date).days
            trends = await self.get_trends(
                granularity=TimeGranularity.DAY,
                days=days,
            )

        # Get top tags
        top_tags = await self.get_top_tags(
            limit=10,
            start_date=start_date,
            end_date=end_date,
        )

        # Generate insights
        insights = await self.generate_insights(
            start_date=start_date,
            end_date=end_date,
        )

        return AnalyticsReport(
            report_id=str(uuid.uuid4()),
            generated_at=datetime.now(),
            period_start=start_date,
            period_end=end_date,
            provider_stats=provider_stats,
            overall_stats=overall_stats,
            trends=trends,
            top_tags=top_tags,
            insights=insights,
        )


def create_analytics(
    persistence: Optional[ResultPersistence] = None,
) -> VisionAnalytics:
    """
    Factory to create analytics instance.

    Args:
        persistence: Optional persistence instance

    Returns:
        VisionAnalytics instance

    Example:
        >>> analytics = create_analytics()
        >>> report = await analytics.generate_report()
        >>> print(f"Total requests: {report.overall_stats.total_requests}")
    """
    from .persistence import get_persistence
    persistence = persistence or get_persistence()
    return VisionAnalytics(persistence)
