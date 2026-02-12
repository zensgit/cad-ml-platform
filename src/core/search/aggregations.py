"""Search Aggregations.

Provides aggregation builders for Elasticsearch analytics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Aggregation:
    """Base aggregation class."""

    name: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Elasticsearch aggregation dict."""
        raise NotImplementedError


@dataclass
class TermsAggregation(Aggregation):
    """Terms bucket aggregation."""

    field: str
    size: int = 10
    min_doc_count: int = 1
    order: Optional[Dict[str, str]] = None
    include: Optional[str] = None
    exclude: Optional[str] = None
    missing: Optional[Any] = None
    sub_aggregations: List[Aggregation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        terms_body: Dict[str, Any] = {
            "field": self.field,
            "size": self.size,
        }

        if self.min_doc_count != 1:
            terms_body["min_doc_count"] = self.min_doc_count

        if self.order:
            terms_body["order"] = self.order

        if self.include:
            terms_body["include"] = self.include

        if self.exclude:
            terms_body["exclude"] = self.exclude

        if self.missing is not None:
            terms_body["missing"] = self.missing

        result: Dict[str, Any] = {"terms": terms_body}

        if self.sub_aggregations:
            result["aggs"] = {
                agg.name: agg.to_dict()[agg.name]
                for agg in self.sub_aggregations
            }

        return {self.name: result}


@dataclass
class DateHistogramAggregation(Aggregation):
    """Date histogram bucket aggregation."""

    field: str
    calendar_interval: Optional[str] = None  # minute, hour, day, week, month, year
    fixed_interval: Optional[str] = None  # 30m, 1h, 1d
    format: Optional[str] = None
    time_zone: Optional[str] = None
    min_doc_count: int = 0
    extended_bounds: Optional[Dict[str, Any]] = None
    sub_aggregations: List[Aggregation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        date_hist_body: Dict[str, Any] = {"field": self.field}

        if self.calendar_interval:
            date_hist_body["calendar_interval"] = self.calendar_interval
        elif self.fixed_interval:
            date_hist_body["fixed_interval"] = self.fixed_interval

        if self.format:
            date_hist_body["format"] = self.format

        if self.time_zone:
            date_hist_body["time_zone"] = self.time_zone

        if self.min_doc_count != 0:
            date_hist_body["min_doc_count"] = self.min_doc_count

        if self.extended_bounds:
            date_hist_body["extended_bounds"] = self.extended_bounds

        result: Dict[str, Any] = {"date_histogram": date_hist_body}

        if self.sub_aggregations:
            result["aggs"] = {
                agg.name: agg.to_dict()[agg.name]
                for agg in self.sub_aggregations
            }

        return {self.name: result}


@dataclass
class HistogramAggregation(Aggregation):
    """Numeric histogram bucket aggregation."""

    field: str
    interval: float
    min_doc_count: int = 0
    offset: float = 0
    extended_bounds: Optional[Dict[str, float]] = None
    sub_aggregations: List[Aggregation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        hist_body: Dict[str, Any] = {
            "field": self.field,
            "interval": self.interval,
        }

        if self.min_doc_count != 0:
            hist_body["min_doc_count"] = self.min_doc_count

        if self.offset != 0:
            hist_body["offset"] = self.offset

        if self.extended_bounds:
            hist_body["extended_bounds"] = self.extended_bounds

        result: Dict[str, Any] = {"histogram": hist_body}

        if self.sub_aggregations:
            result["aggs"] = {
                agg.name: agg.to_dict()[agg.name]
                for agg in self.sub_aggregations
            }

        return {self.name: result}


@dataclass
class RangeAggregation(Aggregation):
    """Range bucket aggregation."""

    field: str
    ranges: List[Dict[str, Any]]  # [{"to": 50}, {"from": 50, "to": 100}, {"from": 100}]
    keyed: bool = False
    sub_aggregations: List[Aggregation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        range_body: Dict[str, Any] = {
            "field": self.field,
            "ranges": self.ranges,
        }

        if self.keyed:
            range_body["keyed"] = True

        result: Dict[str, Any] = {"range": range_body}

        if self.sub_aggregations:
            result["aggs"] = {
                agg.name: agg.to_dict()[agg.name]
                for agg in self.sub_aggregations
            }

        return {self.name: result}


@dataclass
class FilterAggregation(Aggregation):
    """Filter bucket aggregation."""

    filter: Dict[str, Any]
    sub_aggregations: List[Aggregation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"filter": self.filter}

        if self.sub_aggregations:
            result["aggs"] = {
                agg.name: agg.to_dict()[agg.name]
                for agg in self.sub_aggregations
            }

        return {self.name: result}


@dataclass
class NestedAggregation(Aggregation):
    """Nested bucket aggregation."""

    path: str
    sub_aggregations: List[Aggregation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"nested": {"path": self.path}}

        if self.sub_aggregations:
            result["aggs"] = {
                agg.name: agg.to_dict()[agg.name]
                for agg in self.sub_aggregations
            }

        return {self.name: result}


# ============================================================================
# Metrics Aggregations
# ============================================================================

@dataclass
class AvgAggregation(Aggregation):
    """Average metric aggregation."""

    field: str
    missing: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        avg_body: Dict[str, Any] = {"field": self.field}
        if self.missing is not None:
            avg_body["missing"] = self.missing
        return {self.name: {"avg": avg_body}}


@dataclass
class SumAggregation(Aggregation):
    """Sum metric aggregation."""

    field: str
    missing: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        sum_body: Dict[str, Any] = {"field": self.field}
        if self.missing is not None:
            sum_body["missing"] = self.missing
        return {self.name: {"sum": sum_body}}


@dataclass
class MinAggregation(Aggregation):
    """Min metric aggregation."""

    field: str
    missing: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        min_body: Dict[str, Any] = {"field": self.field}
        if self.missing is not None:
            min_body["missing"] = self.missing
        return {self.name: {"min": min_body}}


@dataclass
class MaxAggregation(Aggregation):
    """Max metric aggregation."""

    field: str
    missing: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        max_body: Dict[str, Any] = {"field": self.field}
        if self.missing is not None:
            max_body["missing"] = self.missing
        return {self.name: {"max": max_body}}


@dataclass
class StatsAggregation(Aggregation):
    """Stats metric aggregation (count, min, max, avg, sum)."""

    field: str
    missing: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        stats_body: Dict[str, Any] = {"field": self.field}
        if self.missing is not None:
            stats_body["missing"] = self.missing
        return {self.name: {"stats": stats_body}}


@dataclass
class ExtendedStatsAggregation(Aggregation):
    """Extended stats metric aggregation."""

    field: str
    sigma: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        stats_body: Dict[str, Any] = {"field": self.field}
        if self.sigma is not None:
            stats_body["sigma"] = self.sigma
        return {self.name: {"extended_stats": stats_body}}


@dataclass
class CardinalityAggregation(Aggregation):
    """Cardinality (unique count) metric aggregation."""

    field: str
    precision_threshold: int = 3000

    def to_dict(self) -> Dict[str, Any]:
        return {
            self.name: {
                "cardinality": {
                    "field": self.field,
                    "precision_threshold": self.precision_threshold,
                }
            }
        }


@dataclass
class PercentilesAggregation(Aggregation):
    """Percentiles metric aggregation."""

    field: str
    percents: List[float] = field(default_factory=lambda: [1, 5, 25, 50, 75, 95, 99])

    def to_dict(self) -> Dict[str, Any]:
        return {
            self.name: {
                "percentiles": {
                    "field": self.field,
                    "percents": self.percents,
                }
            }
        }


@dataclass
class ValueCountAggregation(Aggregation):
    """Value count metric aggregation."""

    field: str

    def to_dict(self) -> Dict[str, Any]:
        return {self.name: {"value_count": {"field": self.field}}}


@dataclass
class TopHitsAggregation(Aggregation):
    """Top hits metric aggregation."""

    size: int = 3
    sort: Optional[List[Dict[str, Any]]] = None
    source: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        top_hits_body: Dict[str, Any] = {"size": self.size}

        if self.sort:
            top_hits_body["sort"] = self.sort

        if self.source:
            top_hits_body["_source"] = self.source

        return {self.name: {"top_hits": top_hits_body}}


# ============================================================================
# Aggregation Builder
# ============================================================================

class AggregationBuilder:
    """Fluent aggregation builder."""

    def __init__(self):
        self._aggregations: List[Aggregation] = []

    def terms(
        self,
        name: str,
        field: str,
        size: int = 10,
        **kwargs: Any,
    ) -> "AggregationBuilder":
        """Add terms aggregation."""
        self._aggregations.append(TermsAggregation(
            name=name, field=field, size=size, **kwargs
        ))
        return self

    def date_histogram(
        self,
        name: str,
        field: str,
        calendar_interval: Optional[str] = None,
        fixed_interval: Optional[str] = None,
        **kwargs: Any,
    ) -> "AggregationBuilder":
        """Add date histogram aggregation."""
        self._aggregations.append(DateHistogramAggregation(
            name=name,
            field=field,
            calendar_interval=calendar_interval,
            fixed_interval=fixed_interval,
            **kwargs,
        ))
        return self

    def histogram(
        self,
        name: str,
        field: str,
        interval: float,
        **kwargs: Any,
    ) -> "AggregationBuilder":
        """Add histogram aggregation."""
        self._aggregations.append(HistogramAggregation(
            name=name, field=field, interval=interval, **kwargs
        ))
        return self

    def range(
        self,
        name: str,
        field: str,
        ranges: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> "AggregationBuilder":
        """Add range aggregation."""
        self._aggregations.append(RangeAggregation(
            name=name, field=field, ranges=ranges, **kwargs
        ))
        return self

    def avg(self, name: str, field: str, **kwargs: Any) -> "AggregationBuilder":
        """Add avg aggregation."""
        self._aggregations.append(AvgAggregation(name=name, field=field, **kwargs))
        return self

    def sum(self, name: str, field: str, **kwargs: Any) -> "AggregationBuilder":
        """Add sum aggregation."""
        self._aggregations.append(SumAggregation(name=name, field=field, **kwargs))
        return self

    def min(self, name: str, field: str, **kwargs: Any) -> "AggregationBuilder":
        """Add min aggregation."""
        self._aggregations.append(MinAggregation(name=name, field=field, **kwargs))
        return self

    def max(self, name: str, field: str, **kwargs: Any) -> "AggregationBuilder":
        """Add max aggregation."""
        self._aggregations.append(MaxAggregation(name=name, field=field, **kwargs))
        return self

    def stats(self, name: str, field: str, **kwargs: Any) -> "AggregationBuilder":
        """Add stats aggregation."""
        self._aggregations.append(StatsAggregation(name=name, field=field, **kwargs))
        return self

    def cardinality(
        self,
        name: str,
        field: str,
        **kwargs: Any,
    ) -> "AggregationBuilder":
        """Add cardinality aggregation."""
        self._aggregations.append(CardinalityAggregation(
            name=name, field=field, **kwargs
        ))
        return self

    def percentiles(
        self,
        name: str,
        field: str,
        percents: Optional[List[float]] = None,
    ) -> "AggregationBuilder":
        """Add percentiles aggregation."""
        agg = PercentilesAggregation(name=name, field=field)
        if percents:
            agg.percents = percents
        self._aggregations.append(agg)
        return self

    def top_hits(
        self,
        name: str,
        size: int = 3,
        **kwargs: Any,
    ) -> "AggregationBuilder":
        """Add top_hits aggregation."""
        self._aggregations.append(TopHitsAggregation(name=name, size=size, **kwargs))
        return self

    def build(self) -> Dict[str, Any]:
        """Build aggregations dict."""
        result: Dict[str, Any] = {}
        for agg in self._aggregations:
            agg_dict = agg.to_dict()
            result.update(agg_dict)
        return result
