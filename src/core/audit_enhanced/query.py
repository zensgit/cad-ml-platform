"""Advanced Audit Query Interface.

Provides flexible querying capabilities:
- Complex filter expressions
- Aggregations
- Time-series analysis
- Export functionality
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Union

from src.core.audit_enhanced.record import (
    AuditCategory,
    AuditOutcome,
    AuditRecord,
    AuditSeverity,
)
from src.core.audit_enhanced.storage import AuditStorage

logger = logging.getLogger(__name__)


class ComparisonOperator(Enum):
    """Comparison operators for filters."""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUALS = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUALS = "lte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"


class LogicalOperator(Enum):
    """Logical operators for combining filters."""
    AND = "and"
    OR = "or"
    NOT = "not"


@dataclass
class FilterCondition:
    """A single filter condition."""
    field: str
    operator: ComparisonOperator
    value: Any

    def evaluate(self, record: AuditRecord) -> bool:
        """Evaluate condition against a record."""
        actual_value = self._get_field_value(record)

        if self.operator == ComparisonOperator.EQUALS:
            return actual_value == self.value
        elif self.operator == ComparisonOperator.NOT_EQUALS:
            return actual_value != self.value
        elif self.operator == ComparisonOperator.GREATER_THAN:
            return actual_value > self.value
        elif self.operator == ComparisonOperator.GREATER_THAN_OR_EQUALS:
            return actual_value >= self.value
        elif self.operator == ComparisonOperator.LESS_THAN:
            return actual_value < self.value
        elif self.operator == ComparisonOperator.LESS_THAN_OR_EQUALS:
            return actual_value <= self.value
        elif self.operator == ComparisonOperator.IN:
            return actual_value in self.value
        elif self.operator == ComparisonOperator.NOT_IN:
            return actual_value not in self.value
        elif self.operator == ComparisonOperator.CONTAINS:
            return self.value in str(actual_value) if actual_value else False
        elif self.operator == ComparisonOperator.STARTS_WITH:
            return str(actual_value).startswith(self.value) if actual_value else False
        elif self.operator == ComparisonOperator.ENDS_WITH:
            return str(actual_value).endswith(self.value) if actual_value else False
        elif self.operator == ComparisonOperator.REGEX:
            import re
            return bool(re.search(self.value, str(actual_value))) if actual_value else False

        return False

    def _get_field_value(self, record: AuditRecord) -> Any:
        """Extract field value from record."""
        parts = self.field.split(".")

        if parts[0] == "context" and record.context:
            if len(parts) > 1:
                return getattr(record.context, parts[1], None)
            return record.context
        elif parts[0] == "details" and len(parts) > 1:
            return record.details.get(parts[1])

        return getattr(record, parts[0], None)


@dataclass
class FilterGroup:
    """A group of filter conditions."""
    operator: LogicalOperator = LogicalOperator.AND
    conditions: List[Union[FilterCondition, "FilterGroup"]] = field(default_factory=list)

    def evaluate(self, record: AuditRecord) -> bool:
        """Evaluate filter group against a record."""
        if not self.conditions:
            return True

        results = [c.evaluate(record) for c in self.conditions]

        if self.operator == LogicalOperator.AND:
            return all(results)
        elif self.operator == LogicalOperator.OR:
            return any(results)
        elif self.operator == LogicalOperator.NOT:
            return not results[0] if results else True

        return False

    def add_condition(
        self,
        field: str,
        operator: ComparisonOperator,
        value: Any,
    ) -> "FilterGroup":
        """Add a condition to the group."""
        self.conditions.append(FilterCondition(field, operator, value))
        return self

    def add_group(self, group: "FilterGroup") -> "FilterGroup":
        """Add a nested group."""
        self.conditions.append(group)
        return self


class FilterBuilder:
    """Fluent builder for creating filters."""

    def __init__(self):
        self._root = FilterGroup()
        self._current = self._root

    def where(self, field: str) -> "ConditionBuilder":
        """Start a new condition."""
        return ConditionBuilder(self, field)

    def and_group(self) -> "FilterBuilder":
        """Start an AND group."""
        group = FilterGroup(LogicalOperator.AND)
        self._current.add_group(group)
        self._current = group
        return self

    def or_group(self) -> "FilterBuilder":
        """Start an OR group."""
        group = FilterGroup(LogicalOperator.OR)
        self._current.add_group(group)
        self._current = group
        return self

    def end_group(self) -> "FilterBuilder":
        """End current group."""
        self._current = self._root
        return self

    def build(self) -> FilterGroup:
        """Build the filter."""
        return self._root


class ConditionBuilder:
    """Builder for individual conditions."""

    def __init__(self, filter_builder: FilterBuilder, field: str):
        self._filter_builder = filter_builder
        self._field = field

    def equals(self, value: Any) -> FilterBuilder:
        self._filter_builder._current.add_condition(
            self._field, ComparisonOperator.EQUALS, value
        )
        return self._filter_builder

    def not_equals(self, value: Any) -> FilterBuilder:
        self._filter_builder._current.add_condition(
            self._field, ComparisonOperator.NOT_EQUALS, value
        )
        return self._filter_builder

    def greater_than(self, value: Any) -> FilterBuilder:
        self._filter_builder._current.add_condition(
            self._field, ComparisonOperator.GREATER_THAN, value
        )
        return self._filter_builder

    def less_than(self, value: Any) -> FilterBuilder:
        self._filter_builder._current.add_condition(
            self._field, ComparisonOperator.LESS_THAN, value
        )
        return self._filter_builder

    def is_in(self, values: List[Any]) -> FilterBuilder:
        self._filter_builder._current.add_condition(
            self._field, ComparisonOperator.IN, values
        )
        return self._filter_builder

    def contains(self, value: str) -> FilterBuilder:
        self._filter_builder._current.add_condition(
            self._field, ComparisonOperator.CONTAINS, value
        )
        return self._filter_builder

    def starts_with(self, value: str) -> FilterBuilder:
        self._filter_builder._current.add_condition(
            self._field, ComparisonOperator.STARTS_WITH, value
        )
        return self._filter_builder

    def matches(self, pattern: str) -> FilterBuilder:
        self._filter_builder._current.add_condition(
            self._field, ComparisonOperator.REGEX, pattern
        )
        return self._filter_builder


class AggregationType(Enum):
    """Types of aggregations."""
    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"
    MIN = "min"
    MAX = "max"
    AVG = "avg"
    SUM = "sum"


@dataclass
class AggregationResult:
    """Result of an aggregation."""
    name: str
    value: Any
    group_by: Optional[Dict[str, Any]] = None


@dataclass
class TimeSeriesPoint:
    """A point in a time series."""
    timestamp: datetime
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


class QueryResult:
    """Container for query results with metadata."""

    def __init__(
        self,
        records: List[AuditRecord],
        total_count: int,
        query_time_ms: float,
        aggregations: Optional[List[AggregationResult]] = None,
    ):
        self.records = records
        self.total_count = total_count
        self.query_time_ms = query_time_ms
        self.aggregations = aggregations or []

    @property
    def count(self) -> int:
        return len(self.records)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "records": [r.to_dict() for r in self.records],
            "total_count": self.total_count,
            "returned_count": self.count,
            "query_time_ms": self.query_time_ms,
            "aggregations": [
                {"name": a.name, "value": a.value, "group_by": a.group_by}
                for a in self.aggregations
            ],
        }


class AuditQuery:
    """Advanced query interface for audit records."""

    def __init__(self, storage: AuditStorage):
        self._storage = storage

    async def execute(
        self,
        filter_group: Optional[FilterGroup] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        sort_by: str = "timestamp",
        sort_desc: bool = True,
        limit: int = 100,
        offset: int = 0,
        aggregations: Optional[List[tuple[str, AggregationType]]] = None,
        group_by: Optional[List[str]] = None,
    ) -> QueryResult:
        """Execute a query with filters and aggregations."""
        import time
        start_ms = time.time() * 1000

        # Get base records
        records = await self._storage.query(
            start_time=start_time,
            end_time=end_time,
            limit=10000,  # Get more for filtering
        )

        # Apply custom filters
        if filter_group:
            records = [r for r in records if filter_group.evaluate(r)]

        total_count = len(records)

        # Sort
        records = self._sort_records(records, sort_by, sort_desc)

        # Calculate aggregations before pagination
        agg_results = []
        if aggregations:
            agg_results = self._calculate_aggregations(records, aggregations, group_by)

        # Apply pagination
        paginated = records[offset:offset + limit]

        query_time_ms = time.time() * 1000 - start_ms

        return QueryResult(
            records=paginated,
            total_count=total_count,
            query_time_ms=query_time_ms,
            aggregations=agg_results,
        )

    def _sort_records(
        self,
        records: List[AuditRecord],
        sort_by: str,
        desc: bool,
    ) -> List[AuditRecord]:
        """Sort records by field."""
        def get_sort_key(record: AuditRecord) -> Any:
            value = getattr(record, sort_by, None)
            if value is None:
                return datetime.min if sort_by == "timestamp" else ""
            return value

        return sorted(records, key=get_sort_key, reverse=desc)

    def _calculate_aggregations(
        self,
        records: List[AuditRecord],
        aggregations: List[tuple[str, AggregationType]],
        group_by: Optional[List[str]],
    ) -> List[AggregationResult]:
        """Calculate aggregations on records."""
        results = []

        if group_by:
            # Group records
            groups: Dict[tuple, List[AuditRecord]] = {}
            for record in records:
                key = tuple(str(getattr(record, f, "")) for f in group_by)
                if key not in groups:
                    groups[key] = []
                groups[key].append(record)

            # Calculate per group
            for key, group_records in groups.items():
                group_dict = dict(zip(group_by, key))
                for field, agg_type in aggregations:
                    value = self._aggregate(group_records, field, agg_type)
                    results.append(AggregationResult(
                        name=f"{agg_type.value}_{field}",
                        value=value,
                        group_by=group_dict,
                    ))
        else:
            # Global aggregations
            for field, agg_type in aggregations:
                value = self._aggregate(records, field, agg_type)
                results.append(AggregationResult(
                    name=f"{agg_type.value}_{field}",
                    value=value,
                ))

        return results

    def _aggregate(
        self,
        records: List[AuditRecord],
        field: str,
        agg_type: AggregationType,
    ) -> Any:
        """Perform single aggregation."""
        if agg_type == AggregationType.COUNT:
            return len(records)

        values = [getattr(r, field, None) for r in records if getattr(r, field, None) is not None]

        if agg_type == AggregationType.COUNT_DISTINCT:
            return len(set(values))
        elif agg_type == AggregationType.MIN:
            return min(values) if values else None
        elif agg_type == AggregationType.MAX:
            return max(values) if values else None
        elif agg_type == AggregationType.AVG:
            return sum(values) / len(values) if values else None
        elif agg_type == AggregationType.SUM:
            return sum(values) if values else None

        return None

    async def time_series(
        self,
        start_time: datetime,
        end_time: datetime,
        interval: timedelta,
        filter_group: Optional[FilterGroup] = None,
        aggregation: AggregationType = AggregationType.COUNT,
        field: Optional[str] = None,
    ) -> List[TimeSeriesPoint]:
        """Generate time series data."""
        records = await self._storage.query(
            start_time=start_time,
            end_time=end_time,
            limit=100000,
        )

        if filter_group:
            records = [r for r in records if filter_group.evaluate(r)]

        # Build time buckets
        points = []
        current = start_time
        while current < end_time:
            bucket_end = current + interval
            bucket_records = [
                r for r in records
                if current <= r.timestamp < bucket_end
            ]

            if aggregation == AggregationType.COUNT:
                value = len(bucket_records)
            elif field:
                value = self._aggregate(bucket_records, field, aggregation)
            else:
                value = len(bucket_records)

            points.append(TimeSeriesPoint(
                timestamp=current,
                value=value,
                metadata={"bucket_end": bucket_end.isoformat()},
            ))

            current = bucket_end

        return points


class AuditExporter:
    """Export audit records to various formats."""

    def __init__(self, storage: AuditStorage):
        self._storage = storage

    async def to_json(
        self,
        records: List[AuditRecord],
        pretty: bool = True,
    ) -> str:
        """Export records to JSON."""
        data = [r.to_dict() for r in records]
        if pretty:
            return json.dumps(data, indent=2, default=str)
        return json.dumps(data, default=str)

    async def to_csv(
        self,
        records: List[AuditRecord],
        fields: Optional[List[str]] = None,
    ) -> str:
        """Export records to CSV."""
        if not records:
            return ""

        default_fields = [
            "record_id", "timestamp", "category", "action",
            "severity", "outcome", "user_id", "resource_type",
            "resource_id", "description",
        ]
        fields = fields or default_fields

        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()

        for record in records:
            row = {}
            for field in fields:
                if field == "user_id" and record.context:
                    row[field] = record.context.user_id
                elif field == "category":
                    row[field] = record.category.value
                elif field == "severity":
                    row[field] = record.severity.value
                elif field == "outcome":
                    row[field] = record.outcome.value
                elif field == "timestamp":
                    row[field] = record.timestamp.isoformat()
                else:
                    row[field] = getattr(record, field, "")
            writer.writerow(row)

        return output.getvalue()

    async def to_jsonl(
        self,
        records: List[AuditRecord],
    ) -> str:
        """Export records to JSON Lines format."""
        lines = [json.dumps(r.to_dict(), default=str) for r in records]
        return "\n".join(lines)

    async def stream_export(
        self,
        start_time: datetime,
        end_time: datetime,
        format: str = "jsonl",
        batch_size: int = 1000,
    ):
        """Stream export for large datasets."""
        offset = 0
        while True:
            records = await self._storage.query(
                start_time=start_time,
                end_time=end_time,
                limit=batch_size,
                offset=offset,
            )

            if not records:
                break

            if format == "jsonl":
                yield await self.to_jsonl(records)
            elif format == "json":
                yield await self.to_json(records, pretty=False)
            elif format == "csv":
                yield await self.to_csv(records)

            offset += batch_size
            if len(records) < batch_size:
                break
