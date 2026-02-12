"""Data Transformations.

Provides data transformation primitives:
- Field mappings
- Type conversions
- Filtering and validation
- Aggregations
"""

from __future__ import annotations

import copy
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class TransformError(Exception):
    """Transformation error."""
    pass


class Transformer(ABC, Generic[T, R]):
    """Abstract base class for data transformers."""

    @abstractmethod
    def transform(self, data: T) -> R:
        """Transform input data."""
        pass

    def __call__(self, data: T) -> R:
        return self.transform(data)


class MapTransformer(Transformer[Dict[str, Any], Dict[str, Any]]):
    """Transform dictionary fields."""

    def __init__(
        self,
        mappings: Dict[str, str],
        include_unmapped: bool = True,
        drop_none: bool = False,
    ):
        """
        Args:
            mappings: Dict of source_field -> target_field
            include_unmapped: Include fields not in mappings
            drop_none: Drop fields with None values
        """
        self.mappings = mappings
        self.include_unmapped = include_unmapped
        self.drop_none = drop_none

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result = {}

        # Apply mappings
        for source, target in self.mappings.items():
            if source in data:
                value = data[source]
                if not self.drop_none or value is not None:
                    result[target] = value

        # Include unmapped fields
        if self.include_unmapped:
            mapped_sources = set(self.mappings.keys())
            for key, value in data.items():
                if key not in mapped_sources:
                    if not self.drop_none or value is not None:
                        result[key] = value

        return result


class SelectTransformer(Transformer[Dict[str, Any], Dict[str, Any]]):
    """Select specific fields."""

    def __init__(self, fields: List[str]):
        self.fields = fields

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {k: data[k] for k in self.fields if k in data}


class DropTransformer(Transformer[Dict[str, Any], Dict[str, Any]]):
    """Drop specific fields."""

    def __init__(self, fields: List[str]):
        self.fields = set(fields)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in data.items() if k not in self.fields}


class TypeConvertTransformer(Transformer[Dict[str, Any], Dict[str, Any]]):
    """Convert field types."""

    TYPE_CONVERTERS = {
        "str": str,
        "int": int,
        "float": float,
        "bool": lambda x: x if isinstance(x, bool) else str(x).lower() in ("true", "1", "yes"),
        "datetime": lambda x: datetime.fromisoformat(x) if isinstance(x, str) else x,
        "json": lambda x: json.loads(x) if isinstance(x, str) else x,
    }

    def __init__(
        self,
        conversions: Dict[str, str],
        on_error: str = "skip",  # skip, raise, null
    ):
        self.conversions = conversions
        self.on_error = on_error

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result = copy.copy(data)

        for field_name, target_type in self.conversions.items():
            if field_name not in result:
                continue

            value = result[field_name]
            if value is None:
                continue

            converter = self.TYPE_CONVERTERS.get(target_type)
            if not converter:
                continue

            try:
                result[field_name] = converter(value)
            except Exception as e:
                if self.on_error == "raise":
                    raise TransformError(f"Failed to convert {field_name} to {target_type}: {e}")
                elif self.on_error == "null":
                    result[field_name] = None
                # skip: leave original value

        return result


class ComputeTransformer(Transformer[Dict[str, Any], Dict[str, Any]]):
    """Compute new fields from existing data."""

    def __init__(
        self,
        computations: Dict[str, Callable[[Dict[str, Any]], Any]],
        on_error: str = "skip",
    ):
        self.computations = computations
        self.on_error = on_error

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result = copy.copy(data)

        for field_name, compute_fn in self.computations.items():
            try:
                result[field_name] = compute_fn(data)
            except Exception as e:
                if self.on_error == "raise":
                    raise TransformError(f"Failed to compute {field_name}: {e}")
                elif self.on_error == "null":
                    result[field_name] = None

        return result


class FilterTransformer(Transformer[Dict[str, Any], Optional[Dict[str, Any]]]):
    """Filter records based on conditions."""

    def __init__(self, predicate: Callable[[Dict[str, Any]], bool]):
        self.predicate = predicate

    def transform(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self.predicate(data):
            return data
        return None


class ValidateTransformer(Transformer[Dict[str, Any], Dict[str, Any]]):
    """Validate records against schema."""

    def __init__(
        self,
        validators: Dict[str, Callable[[Any], bool]],
        required_fields: Optional[List[str]] = None,
        on_error: str = "raise",
    ):
        self.validators = validators
        self.required_fields = required_fields or []
        self.on_error = on_error

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        errors = []

        # Check required fields
        for field_name in self.required_fields:
            if field_name not in data or data[field_name] is None:
                errors.append(f"Missing required field: {field_name}")

        # Run validators
        for field_name, validator in self.validators.items():
            if field_name in data:
                try:
                    if not validator(data[field_name]):
                        errors.append(f"Validation failed for field: {field_name}")
                except Exception as e:
                    errors.append(f"Validator error for {field_name}: {e}")

        if errors:
            if self.on_error == "raise":
                raise TransformError("; ".join(errors))
            elif self.on_error == "tag":
                data["_validation_errors"] = errors

        return data


class FlattenTransformer(Transformer[Dict[str, Any], Dict[str, Any]]):
    """Flatten nested dictionaries."""

    def __init__(self, separator: str = ".", max_depth: int = 10):
        self.separator = separator
        self.max_depth = max_depth

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._flatten(data, "", 0)

    def _flatten(
        self,
        obj: Any,
        prefix: str,
        depth: int,
    ) -> Dict[str, Any]:
        result = {}

        if depth >= self.max_depth:
            result[prefix.rstrip(self.separator)] = obj
            return result

        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{prefix}{key}"
                if isinstance(value, dict):
                    result.update(self._flatten(value, f"{new_key}{self.separator}", depth + 1))
                else:
                    result[new_key] = value
        else:
            result[prefix.rstrip(self.separator)] = obj

        return result


class NestTransformer(Transformer[Dict[str, Any], Dict[str, Any]]):
    """Nest flat fields into hierarchical structure."""

    def __init__(self, separator: str = "."):
        self.separator = separator

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        for key, value in data.items():
            parts = key.split(self.separator)
            current = result

            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            current[parts[-1]] = value

        return result


class ChainTransformer(Transformer[T, R]):
    """Chain multiple transformers."""

    def __init__(self, transformers: List[Transformer]):
        self.transformers = transformers

    def transform(self, data: T) -> R:
        result: Any = data
        for transformer in self.transformers:
            if result is None:
                return None  # type: ignore
            result = transformer.transform(result)
        return result

    def add(self, transformer: Transformer) -> "ChainTransformer":
        self.transformers.append(transformer)
        return self


class ConditionalTransformer(Transformer[T, R]):
    """Apply different transformers based on conditions."""

    def __init__(
        self,
        condition: Callable[[T], bool],
        if_true: Transformer[T, R],
        if_false: Optional[Transformer[T, R]] = None,
    ):
        self.condition = condition
        self.if_true = if_true
        self.if_false = if_false

    def transform(self, data: T) -> R:
        if self.condition(data):
            return self.if_true.transform(data)
        elif self.if_false:
            return self.if_false.transform(data)
        return data  # type: ignore


# Aggregators

class AggregatorType(Enum):
    """Types of aggregations."""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"
    FIRST = "first"
    LAST = "last"
    COLLECT = "collect"


@dataclass
class Aggregator:
    """Aggregation configuration."""
    field: str
    type: AggregatorType
    alias: Optional[str] = None

    @property
    def output_name(self) -> str:
        return self.alias or f"{self.type.value}_{self.field}"


class GroupByAggregator:
    """Group records and apply aggregations."""

    def __init__(
        self,
        group_by: List[str],
        aggregators: List[Aggregator],
    ):
        self.group_by = group_by
        self.aggregators = aggregators

    def aggregate(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregate records by group."""
        if not records:
            return []

        # Group records
        groups: Dict[tuple, List[Dict[str, Any]]] = {}
        for record in records:
            key = tuple(record.get(f) for f in self.group_by)
            if key not in groups:
                groups[key] = []
            groups[key].append(record)

        # Apply aggregations
        results = []
        for key, group in groups.items():
            result = dict(zip(self.group_by, key))

            for agg in self.aggregators:
                values = [r.get(agg.field) for r in group if r.get(agg.field) is not None]
                result[agg.output_name] = self._compute(values, agg.type)

            results.append(result)

        return results

    def _compute(self, values: List[Any], agg_type: AggregatorType) -> Any:
        if not values:
            return None

        if agg_type == AggregatorType.SUM:
            return sum(values)
        elif agg_type == AggregatorType.AVG:
            return sum(values) / len(values)
        elif agg_type == AggregatorType.MIN:
            return min(values)
        elif agg_type == AggregatorType.MAX:
            return max(values)
        elif agg_type == AggregatorType.COUNT:
            return len(values)
        elif agg_type == AggregatorType.COUNT_DISTINCT:
            return len(set(values))
        elif agg_type == AggregatorType.FIRST:
            return values[0]
        elif agg_type == AggregatorType.LAST:
            return values[-1]
        elif agg_type == AggregatorType.COLLECT:
            return values

        return None


# Window functions

@dataclass
class WindowSpec:
    """Window specification for window functions."""
    partition_by: List[str] = field(default_factory=list)
    order_by: Optional[str] = None
    order_desc: bool = False
    window_size: Optional[int] = None


class WindowFunction:
    """Apply window functions."""

    def __init__(self, spec: WindowSpec):
        self.spec = spec

    def row_number(
        self,
        records: List[Dict[str, Any]],
        output_field: str = "row_number",
    ) -> List[Dict[str, Any]]:
        """Add row number within partitions."""
        partitioned = self._partition(records)
        results = []

        for partition in partitioned.values():
            sorted_partition = self._sort(partition)
            for i, record in enumerate(sorted_partition, 1):
                record[output_field] = i
                results.append(record)

        return results

    def rank(
        self,
        records: List[Dict[str, Any]],
        output_field: str = "rank",
    ) -> List[Dict[str, Any]]:
        """Add rank within partitions."""
        partitioned = self._partition(records)
        results = []

        for partition in partitioned.values():
            sorted_partition = self._sort(partition)
            rank = 1
            prev_value = None

            for i, record in enumerate(sorted_partition):
                if self.spec.order_by:
                    current_value = record.get(self.spec.order_by)
                    if current_value != prev_value:
                        rank = i + 1
                    prev_value = current_value
                else:
                    rank = i + 1

                record[output_field] = rank
                results.append(record)

        return results

    def running_sum(
        self,
        records: List[Dict[str, Any]],
        value_field: str,
        output_field: str = "running_sum",
    ) -> List[Dict[str, Any]]:
        """Calculate running sum within partitions."""
        partitioned = self._partition(records)
        results = []

        for partition in partitioned.values():
            sorted_partition = self._sort(partition)
            running = 0

            for record in sorted_partition:
                value = record.get(value_field, 0) or 0
                running += value
                record[output_field] = running
                results.append(record)

        return results

    def moving_avg(
        self,
        records: List[Dict[str, Any]],
        value_field: str,
        output_field: str = "moving_avg",
    ) -> List[Dict[str, Any]]:
        """Calculate moving average within partitions."""
        window_size = self.spec.window_size or 3
        partitioned = self._partition(records)
        results = []

        for partition in partitioned.values():
            sorted_partition = self._sort(partition)
            values: List[float] = []

            for record in sorted_partition:
                value = record.get(value_field, 0) or 0
                values.append(value)

                if len(values) > window_size:
                    values = values[-window_size:]

                record[output_field] = sum(values) / len(values)
                results.append(record)

        return results

    def _partition(
        self,
        records: List[Dict[str, Any]],
    ) -> Dict[tuple, List[Dict[str, Any]]]:
        """Partition records by group fields."""
        if not self.spec.partition_by:
            return {(): [copy.copy(r) for r in records]}

        partitions: Dict[tuple, List[Dict[str, Any]]] = {}
        for record in records:
            key = tuple(record.get(f) for f in self.spec.partition_by)
            if key not in partitions:
                partitions[key] = []
            partitions[key].append(copy.copy(record))

        return partitions

    def _sort(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort records within partition."""
        if not self.spec.order_by:
            return records

        return sorted(
            records,
            key=lambda r: r.get(self.spec.order_by, ""),
            reverse=self.spec.order_desc,
        )
