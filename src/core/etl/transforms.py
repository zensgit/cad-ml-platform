"""ETL Transformations.

Provides data transformation operations:
- Field mapping and renaming
- Data type conversions
- Filtering and validation
- Aggregation and enrichment
"""

from __future__ import annotations

import asyncio
import copy
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

from src.core.etl.sources import Record

logger = logging.getLogger(__name__)


class TransformType(str, Enum):
    """Transformation types."""
    MAP = "map"
    FILTER = "filter"
    VALIDATE = "validate"
    AGGREGATE = "aggregate"
    ENRICH = "enrich"
    CUSTOM = "custom"


@dataclass
class TransformResult:
    """Result of a transformation."""
    records: List[Record]
    dropped: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)


class Transform(ABC):
    """Abstract base class for transformations."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def apply(self, records: List[Record]) -> TransformResult:
        """Apply transformation to records."""
        pass


class MapTransform(Transform):
    """Map/transform fields in records."""

    def __init__(
        self,
        name: str,
        mapping: Dict[str, Union[str, Callable[[Any], Any]]],
        drop_unmapped: bool = False,
    ):
        """Initialize map transform.

        Args:
            name: Transform name.
            mapping: Field mapping (old_name -> new_name or old_name -> transform_func).
            drop_unmapped: Whether to drop fields not in mapping.
        """
        super().__init__(name)
        self.mapping = mapping
        self.drop_unmapped = drop_unmapped

    async def apply(self, records: List[Record]) -> TransformResult:
        result_records = []

        for record in records:
            new_data = {}

            if not self.drop_unmapped:
                new_data = copy.copy(record.data)

            for old_key, transform in self.mapping.items():
                if old_key in record.data:
                    value = record.data[old_key]

                    if callable(transform):
                        # Apply function
                        try:
                            value = transform(value)
                        except Exception as e:
                            logger.warning(f"Transform error for {old_key}: {e}")
                            continue
                        new_key = old_key
                    else:
                        # Rename
                        new_key = transform

                    new_data[new_key] = value

                    # Remove old key if renamed
                    if not self.drop_unmapped and callable(transform) is False:
                        new_data.pop(old_key, None)

            result_records.append(Record(
                data=new_data,
                source=record.source,
                offset=record.offset,
                timestamp=record.timestamp,
                metadata=record.metadata,
            ))

        return TransformResult(records=result_records)


class FilterTransform(Transform):
    """Filter records based on conditions."""

    def __init__(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
    ):
        """Initialize filter transform.

        Args:
            name: Transform name.
            condition: Function that returns True to keep record.
        """
        super().__init__(name)
        self.condition = condition

    async def apply(self, records: List[Record]) -> TransformResult:
        result_records = []
        dropped = 0

        for record in records:
            try:
                if self.condition(record.data):
                    result_records.append(record)
                else:
                    dropped += 1
            except Exception as e:
                logger.warning(f"Filter condition error: {e}")
                dropped += 1

        return TransformResult(records=result_records, dropped=dropped)


class ValidateTransform(Transform):
    """Validate records against schema/rules."""

    def __init__(
        self,
        name: str,
        required_fields: Optional[List[str]] = None,
        field_types: Optional[Dict[str, type]] = None,
        validators: Optional[Dict[str, Callable[[Any], bool]]] = None,
        on_error: str = "drop",  # drop, raise, mark
    ):
        """Initialize validate transform.

        Args:
            name: Transform name.
            required_fields: Fields that must be present.
            field_types: Expected field types.
            validators: Custom validation functions.
            on_error: Action on validation error.
        """
        super().__init__(name)
        self.required_fields = required_fields or []
        self.field_types = field_types or {}
        self.validators = validators or {}
        self.on_error = on_error

    async def apply(self, records: List[Record]) -> TransformResult:
        result_records = []
        dropped = 0
        errors = []

        for record in records:
            validation_errors = self._validate_record(record.data)

            if validation_errors:
                if self.on_error == "drop":
                    dropped += 1
                    errors.append({
                        "offset": record.offset,
                        "errors": validation_errors,
                    })
                elif self.on_error == "raise":
                    raise ValueError(f"Validation failed: {validation_errors}")
                elif self.on_error == "mark":
                    record.metadata["validation_errors"] = validation_errors
                    result_records.append(record)
            else:
                result_records.append(record)

        return TransformResult(records=result_records, dropped=dropped, errors=errors)

    def _validate_record(self, data: Dict[str, Any]) -> List[str]:
        """Validate a single record."""
        errors = []

        # Check required fields
        for field in self.required_fields:
            if field not in data or data[field] is None:
                errors.append(f"Missing required field: {field}")

        # Check field types
        for field, expected_type in self.field_types.items():
            if field in data and data[field] is not None:
                if not isinstance(data[field], expected_type):
                    errors.append(
                        f"Invalid type for {field}: expected {expected_type.__name__}, "
                        f"got {type(data[field]).__name__}"
                    )

        # Custom validators
        for field, validator in self.validators.items():
            if field in data:
                try:
                    if not validator(data[field]):
                        errors.append(f"Validation failed for {field}")
                except Exception as e:
                    errors.append(f"Validator error for {field}: {e}")

        return errors


class TypeConvertTransform(Transform):
    """Convert field types."""

    def __init__(
        self,
        name: str,
        conversions: Dict[str, type],
        on_error: str = "null",  # null, keep, raise
    ):
        """Initialize type convert transform.

        Args:
            name: Transform name.
            conversions: Field to type mappings.
            on_error: Action on conversion error.
        """
        super().__init__(name)
        self.conversions = conversions
        self.on_error = on_error

    async def apply(self, records: List[Record]) -> TransformResult:
        result_records = []
        errors = []

        for record in records:
            new_data = copy.copy(record.data)

            for field, target_type in self.conversions.items():
                if field in new_data:
                    try:
                        new_data[field] = self._convert(new_data[field], target_type)
                    except Exception as e:
                        if self.on_error == "null":
                            new_data[field] = None
                        elif self.on_error == "raise":
                            raise
                        # "keep" does nothing
                        errors.append({
                            "offset": record.offset,
                            "field": field,
                            "error": str(e),
                        })

            result_records.append(Record(
                data=new_data,
                source=record.source,
                offset=record.offset,
                timestamp=record.timestamp,
                metadata=record.metadata,
            ))

        return TransformResult(records=result_records, errors=errors)

    def _convert(self, value: Any, target_type: type) -> Any:
        """Convert a value to target type."""
        if value is None:
            return None

        if target_type == bool:
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes", "y")
            return bool(value)

        if target_type == datetime:
            if isinstance(value, datetime):
                return value
            if isinstance(value, str):
                # Try common formats
                for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
                    try:
                        return datetime.strptime(value, fmt)
                    except ValueError:
                        continue
                raise ValueError(f"Cannot parse datetime: {value}")

        return target_type(value)


class EnrichTransform(Transform):
    """Enrich records with additional data."""

    def __init__(
        self,
        name: str,
        enricher: Callable[[Dict[str, Any]], Dict[str, Any]],
        merge: bool = True,
    ):
        """Initialize enrich transform.

        Args:
            name: Transform name.
            enricher: Function that returns additional data.
            merge: Whether to merge or replace data.
        """
        super().__init__(name)
        self.enricher = enricher
        self.merge = merge

    async def apply(self, records: List[Record]) -> TransformResult:
        result_records = []

        for record in records:
            try:
                enrichment = self.enricher(record.data)

                if self.merge:
                    new_data = {**record.data, **enrichment}
                else:
                    new_data = enrichment

                result_records.append(Record(
                    data=new_data,
                    source=record.source,
                    offset=record.offset,
                    timestamp=record.timestamp,
                    metadata=record.metadata,
                ))
            except Exception as e:
                logger.warning(f"Enrichment error: {e}")
                result_records.append(record)

        return TransformResult(records=result_records)


class SelectFieldsTransform(Transform):
    """Select specific fields from records."""

    def __init__(self, name: str, fields: List[str]):
        """Initialize select fields transform.

        Args:
            name: Transform name.
            fields: Fields to keep.
        """
        super().__init__(name)
        self.fields = set(fields)

    async def apply(self, records: List[Record]) -> TransformResult:
        result_records = []

        for record in records:
            new_data = {
                k: v for k, v in record.data.items()
                if k in self.fields
            }
            result_records.append(Record(
                data=new_data,
                source=record.source,
                offset=record.offset,
                timestamp=record.timestamp,
                metadata=record.metadata,
            ))

        return TransformResult(records=result_records)


class DropFieldsTransform(Transform):
    """Drop specific fields from records."""

    def __init__(self, name: str, fields: List[str]):
        """Initialize drop fields transform.

        Args:
            name: Transform name.
            fields: Fields to drop.
        """
        super().__init__(name)
        self.fields = set(fields)

    async def apply(self, records: List[Record]) -> TransformResult:
        result_records = []

        for record in records:
            new_data = {
                k: v for k, v in record.data.items()
                if k not in self.fields
            }
            result_records.append(Record(
                data=new_data,
                source=record.source,
                offset=record.offset,
                timestamp=record.timestamp,
                metadata=record.metadata,
            ))

        return TransformResult(records=result_records)


class FunctionTransform(Transform):
    """Apply a custom function to each record."""

    def __init__(
        self,
        name: str,
        func: Callable[[Dict[str, Any]], Dict[str, Any]],
    ):
        """Initialize function transform.

        Args:
            name: Transform name.
            func: Function to apply to each record's data.
        """
        super().__init__(name)
        self.func = func

    async def apply(self, records: List[Record]) -> TransformResult:
        result_records = []
        errors = []

        for record in records:
            try:
                new_data = self.func(record.data)
                result_records.append(Record(
                    data=new_data,
                    source=record.source,
                    offset=record.offset,
                    timestamp=record.timestamp,
                    metadata=record.metadata,
                ))
            except Exception as e:
                errors.append({
                    "offset": record.offset,
                    "error": str(e),
                })

        return TransformResult(records=result_records, errors=errors)


class ChainTransform(Transform):
    """Chain multiple transforms together."""

    def __init__(self, name: str, transforms: List[Transform]):
        """Initialize chain transform.

        Args:
            name: Transform name.
            transforms: List of transforms to apply in order.
        """
        super().__init__(name)
        self.transforms = transforms

    async def apply(self, records: List[Record]) -> TransformResult:
        current_records = records
        total_dropped = 0
        all_errors = []

        for transform in self.transforms:
            result = await transform.apply(current_records)
            current_records = result.records
            total_dropped += result.dropped
            all_errors.extend(result.errors)

        return TransformResult(
            records=current_records,
            dropped=total_dropped,
            errors=all_errors,
        )
