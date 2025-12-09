"""Data validation and quality control for Vision Provider system.

This module provides data validation features including:
- Schema validation
- Data quality checks
- Constraint validation
- Custom validators
- Validation reporting
"""

import json
import re
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Pattern, Set, Tuple, Type, TypeVar, Union

from .base import VisionDescription, VisionProvider


class ValidationSeverity(Enum):
    """Validation severity levels."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationStatus(Enum):
    """Validation status."""

    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    SKIPPED = "skipped"


class DataType(Enum):
    """Data types for validation."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    BYTES = "bytes"
    DATETIME = "datetime"
    ANY = "any"


@dataclass
class ValidationError:
    """Validation error."""

    error_id: str
    field: str
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    code: Optional[str] = None
    value: Optional[Any] = None
    constraint: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error_id": self.error_id,
            "field": self.field,
            "message": self.message,
            "severity": self.severity.value,
            "code": self.code,
            "value": str(self.value)[:100] if self.value is not None else None,
            "constraint": self.constraint,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ValidationResult:
    """Validation result."""

    valid: bool
    status: ValidationStatus
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    validated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        """Get error count."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Get warning count."""
        return len(self.warnings)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "status": self.status.value,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "validated_at": self.validated_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge with another result."""
        return ValidationResult(
            valid=self.valid and other.valid,
            status=ValidationStatus.INVALID if not (self.valid and other.valid) else ValidationStatus.VALID,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            metadata={**self.metadata, **other.metadata},
        )


class Validator(ABC):
    """Abstract validator."""

    @abstractmethod
    def validate(self, value: Any, field_name: str = "") -> ValidationResult:
        """Validate value."""
        pass


class TypeValidator(Validator):
    """Type validator."""

    def __init__(self, expected_type: DataType) -> None:
        """Initialize validator."""
        self._expected_type = expected_type

    def validate(self, value: Any, field_name: str = "") -> ValidationResult:
        """Validate type."""
        type_map = {
            DataType.STRING: str,
            DataType.INTEGER: int,
            DataType.FLOAT: (int, float),
            DataType.BOOLEAN: bool,
            DataType.LIST: list,
            DataType.DICT: dict,
            DataType.BYTES: bytes,
            DataType.DATETIME: datetime,
        }

        if self._expected_type == DataType.ANY:
            return ValidationResult(valid=True, status=ValidationStatus.VALID)

        expected = type_map.get(self._expected_type)
        if expected is None:
            return ValidationResult(valid=True, status=ValidationStatus.VALID)

        if not isinstance(value, expected):
            return ValidationResult(
                valid=False,
                status=ValidationStatus.INVALID,
                errors=[ValidationError(
                    error_id=str(uuid.uuid4()),
                    field=field_name,
                    message=f"Expected {self._expected_type.value}, got {type(value).__name__}",
                    code="TYPE_MISMATCH",
                    value=value,
                    constraint=f"type={self._expected_type.value}",
                )],
            )

        return ValidationResult(valid=True, status=ValidationStatus.VALID)


class RequiredValidator(Validator):
    """Required field validator."""

    def validate(self, value: Any, field_name: str = "") -> ValidationResult:
        """Validate required field."""
        if value is None:
            return ValidationResult(
                valid=False,
                status=ValidationStatus.INVALID,
                errors=[ValidationError(
                    error_id=str(uuid.uuid4()),
                    field=field_name,
                    message=f"Field '{field_name}' is required",
                    code="REQUIRED",
                    constraint="required=true",
                )],
            )
        return ValidationResult(valid=True, status=ValidationStatus.VALID)


class RangeValidator(Validator):
    """Range validator for numeric values."""

    def __init__(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        exclusive_min: bool = False,
        exclusive_max: bool = False,
    ) -> None:
        """Initialize validator."""
        self._min_value = min_value
        self._max_value = max_value
        self._exclusive_min = exclusive_min
        self._exclusive_max = exclusive_max

    def validate(self, value: Any, field_name: str = "") -> ValidationResult:
        """Validate range."""
        if not isinstance(value, (int, float)):
            return ValidationResult(
                valid=False,
                status=ValidationStatus.INVALID,
                errors=[ValidationError(
                    error_id=str(uuid.uuid4()),
                    field=field_name,
                    message=f"Value must be numeric for range validation",
                    code="TYPE_ERROR",
                    value=value,
                )],
            )

        errors = []

        if self._min_value is not None:
            if self._exclusive_min:
                if value <= self._min_value:
                    errors.append(ValidationError(
                        error_id=str(uuid.uuid4()),
                        field=field_name,
                        message=f"Value must be greater than {self._min_value}",
                        code="MIN_EXCLUSIVE",
                        value=value,
                        constraint=f"min>{self._min_value}",
                    ))
            else:
                if value < self._min_value:
                    errors.append(ValidationError(
                        error_id=str(uuid.uuid4()),
                        field=field_name,
                        message=f"Value must be at least {self._min_value}",
                        code="MIN_VALUE",
                        value=value,
                        constraint=f"min>={self._min_value}",
                    ))

        if self._max_value is not None:
            if self._exclusive_max:
                if value >= self._max_value:
                    errors.append(ValidationError(
                        error_id=str(uuid.uuid4()),
                        field=field_name,
                        message=f"Value must be less than {self._max_value}",
                        code="MAX_EXCLUSIVE",
                        value=value,
                        constraint=f"max<{self._max_value}",
                    ))
            else:
                if value > self._max_value:
                    errors.append(ValidationError(
                        error_id=str(uuid.uuid4()),
                        field=field_name,
                        message=f"Value must be at most {self._max_value}",
                        code="MAX_VALUE",
                        value=value,
                        constraint=f"max<={self._max_value}",
                    ))

        if errors:
            return ValidationResult(valid=False, status=ValidationStatus.INVALID, errors=errors)

        return ValidationResult(valid=True, status=ValidationStatus.VALID)


class LengthValidator(Validator):
    """Length validator for strings and lists."""

    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> None:
        """Initialize validator."""
        self._min_length = min_length
        self._max_length = max_length

    def validate(self, value: Any, field_name: str = "") -> ValidationResult:
        """Validate length."""
        if not hasattr(value, "__len__"):
            return ValidationResult(
                valid=False,
                status=ValidationStatus.INVALID,
                errors=[ValidationError(
                    error_id=str(uuid.uuid4()),
                    field=field_name,
                    message="Value must have length for length validation",
                    code="TYPE_ERROR",
                    value=value,
                )],
            )

        length = len(value)
        errors = []

        if self._min_length is not None and length < self._min_length:
            errors.append(ValidationError(
                error_id=str(uuid.uuid4()),
                field=field_name,
                message=f"Length must be at least {self._min_length}, got {length}",
                code="MIN_LENGTH",
                value=value,
                constraint=f"min_length={self._min_length}",
            ))

        if self._max_length is not None and length > self._max_length:
            errors.append(ValidationError(
                error_id=str(uuid.uuid4()),
                field=field_name,
                message=f"Length must be at most {self._max_length}, got {length}",
                code="MAX_LENGTH",
                value=value,
                constraint=f"max_length={self._max_length}",
            ))

        if errors:
            return ValidationResult(valid=False, status=ValidationStatus.INVALID, errors=errors)

        return ValidationResult(valid=True, status=ValidationStatus.VALID)


class PatternValidator(Validator):
    """Pattern validator for strings."""

    def __init__(self, pattern: str, flags: int = 0) -> None:
        """Initialize validator."""
        self._pattern = re.compile(pattern, flags)
        self._pattern_str = pattern

    def validate(self, value: Any, field_name: str = "") -> ValidationResult:
        """Validate pattern."""
        if not isinstance(value, str):
            return ValidationResult(
                valid=False,
                status=ValidationStatus.INVALID,
                errors=[ValidationError(
                    error_id=str(uuid.uuid4()),
                    field=field_name,
                    message="Value must be string for pattern validation",
                    code="TYPE_ERROR",
                    value=value,
                )],
            )

        if not self._pattern.match(value):
            return ValidationResult(
                valid=False,
                status=ValidationStatus.INVALID,
                errors=[ValidationError(
                    error_id=str(uuid.uuid4()),
                    field=field_name,
                    message=f"Value does not match pattern: {self._pattern_str}",
                    code="PATTERN_MISMATCH",
                    value=value,
                    constraint=f"pattern={self._pattern_str}",
                )],
            )

        return ValidationResult(valid=True, status=ValidationStatus.VALID)


class EnumValidator(Validator):
    """Enum validator."""

    def __init__(self, allowed_values: List[Any]) -> None:
        """Initialize validator."""
        self._allowed_values = set(allowed_values)

    def validate(self, value: Any, field_name: str = "") -> ValidationResult:
        """Validate enum."""
        if value not in self._allowed_values:
            return ValidationResult(
                valid=False,
                status=ValidationStatus.INVALID,
                errors=[ValidationError(
                    error_id=str(uuid.uuid4()),
                    field=field_name,
                    message=f"Value must be one of: {list(self._allowed_values)}",
                    code="ENUM_MISMATCH",
                    value=value,
                    constraint=f"enum={list(self._allowed_values)}",
                )],
            )

        return ValidationResult(valid=True, status=ValidationStatus.VALID)


class CustomValidator(Validator):
    """Custom validator using a function."""

    def __init__(
        self,
        func: Callable[[Any], bool],
        error_message: str = "Validation failed",
        error_code: str = "CUSTOM",
    ) -> None:
        """Initialize validator."""
        self._func = func
        self._error_message = error_message
        self._error_code = error_code

    def validate(self, value: Any, field_name: str = "") -> ValidationResult:
        """Validate using custom function."""
        try:
            if self._func(value):
                return ValidationResult(valid=True, status=ValidationStatus.VALID)
            else:
                return ValidationResult(
                    valid=False,
                    status=ValidationStatus.INVALID,
                    errors=[ValidationError(
                        error_id=str(uuid.uuid4()),
                        field=field_name,
                        message=self._error_message,
                        code=self._error_code,
                        value=value,
                    )],
                )
        except Exception as e:
            return ValidationResult(
                valid=False,
                status=ValidationStatus.INVALID,
                errors=[ValidationError(
                    error_id=str(uuid.uuid4()),
                    field=field_name,
                    message=f"Validation error: {str(e)}",
                    code="VALIDATION_ERROR",
                    value=value,
                )],
            )


class CompositeValidator(Validator):
    """Composite validator combining multiple validators."""

    def __init__(self, validators: List[Validator], mode: str = "all") -> None:
        """Initialize validator."""
        self._validators = validators
        self._mode = mode  # "all" or "any"

    def validate(self, value: Any, field_name: str = "") -> ValidationResult:
        """Validate using all validators."""
        results = [v.validate(value, field_name) for v in self._validators]

        if self._mode == "all":
            all_errors = []
            all_warnings = []
            valid = True

            for result in results:
                all_errors.extend(result.errors)
                all_warnings.extend(result.warnings)
                if not result.valid:
                    valid = False

            return ValidationResult(
                valid=valid,
                status=ValidationStatus.VALID if valid else ValidationStatus.INVALID,
                errors=all_errors,
                warnings=all_warnings,
            )

        else:  # "any"
            for result in results:
                if result.valid:
                    return result

            # All failed, return combined errors
            all_errors = []
            for result in results:
                all_errors.extend(result.errors)

            return ValidationResult(
                valid=False,
                status=ValidationStatus.INVALID,
                errors=all_errors,
            )


@dataclass
class FieldSchema:
    """Field schema definition."""

    name: str
    data_type: DataType = DataType.ANY
    required: bool = False
    validators: List[Validator] = field(default_factory=list)
    default: Any = None
    description: str = ""

    def validate(self, value: Any) -> ValidationResult:
        """Validate field value."""
        results: List[ValidationResult] = []

        # Check required
        if self.required:
            results.append(RequiredValidator().validate(value, self.name))
            if not results[-1].valid:
                return results[-1]

        # If value is None and not required, skip other validations
        if value is None:
            return ValidationResult(valid=True, status=ValidationStatus.VALID)

        # Check type
        results.append(TypeValidator(self.data_type).validate(value, self.name))

        # Run custom validators
        for validator in self.validators:
            results.append(validator.validate(value, self.name))

        # Merge results
        final = ValidationResult(valid=True, status=ValidationStatus.VALID)
        for result in results:
            final = final.merge(result)

        return final


@dataclass
class Schema:
    """Data schema."""

    name: str
    fields: List[FieldSchema] = field(default_factory=list)
    allow_extra_fields: bool = False
    strict: bool = True

    def add_field(self, field_schema: FieldSchema) -> None:
        """Add field schema."""
        self.fields.append(field_schema)

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate data against schema."""
        if not isinstance(data, dict):
            return ValidationResult(
                valid=False,
                status=ValidationStatus.INVALID,
                errors=[ValidationError(
                    error_id=str(uuid.uuid4()),
                    field="",
                    message="Data must be a dictionary",
                    code="TYPE_ERROR",
                )],
            )

        results: List[ValidationResult] = []
        field_names = {f.name for f in self.fields}

        # Validate each field
        for field_schema in self.fields:
            value = data.get(field_schema.name, field_schema.default)
            results.append(field_schema.validate(value))

        # Check for extra fields
        if not self.allow_extra_fields:
            extra_fields = set(data.keys()) - field_names
            if extra_fields:
                results.append(ValidationResult(
                    valid=not self.strict,
                    status=ValidationStatus.WARNING if not self.strict else ValidationStatus.INVALID,
                    errors=[] if not self.strict else [ValidationError(
                        error_id=str(uuid.uuid4()),
                        field=", ".join(extra_fields),
                        message=f"Unexpected fields: {extra_fields}",
                        code="EXTRA_FIELDS",
                    )],
                    warnings=[ValidationError(
                        error_id=str(uuid.uuid4()),
                        field=", ".join(extra_fields),
                        message=f"Unexpected fields: {extra_fields}",
                        code="EXTRA_FIELDS",
                        severity=ValidationSeverity.WARNING,
                    )] if not self.strict else [],
                ))

        # Merge results
        final = ValidationResult(valid=True, status=ValidationStatus.VALID)
        for result in results:
            final = final.merge(result)

        return final


class SchemaBuilder:
    """Schema builder."""

    def __init__(self, name: str) -> None:
        """Initialize builder."""
        self._name = name
        self._fields: List[FieldSchema] = []
        self._allow_extra_fields = False
        self._strict = True

    def field(
        self,
        name: str,
        data_type: DataType = DataType.ANY,
        required: bool = False,
        validators: Optional[List[Validator]] = None,
        default: Any = None,
        description: str = "",
    ) -> "SchemaBuilder":
        """Add field."""
        self._fields.append(FieldSchema(
            name=name,
            data_type=data_type,
            required=required,
            validators=validators or [],
            default=default,
            description=description,
        ))
        return self

    def string_field(
        self,
        name: str,
        required: bool = False,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
    ) -> "SchemaBuilder":
        """Add string field."""
        validators: List[Validator] = []
        if min_length is not None or max_length is not None:
            validators.append(LengthValidator(min_length, max_length))
        if pattern:
            validators.append(PatternValidator(pattern))

        return self.field(name, DataType.STRING, required, validators)

    def integer_field(
        self,
        name: str,
        required: bool = False,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
    ) -> "SchemaBuilder":
        """Add integer field."""
        validators: List[Validator] = []
        if min_value is not None or max_value is not None:
            validators.append(RangeValidator(min_value, max_value))

        return self.field(name, DataType.INTEGER, required, validators)

    def float_field(
        self,
        name: str,
        required: bool = False,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> "SchemaBuilder":
        """Add float field."""
        validators: List[Validator] = []
        if min_value is not None or max_value is not None:
            validators.append(RangeValidator(min_value, max_value))

        return self.field(name, DataType.FLOAT, required, validators)

    def enum_field(
        self,
        name: str,
        allowed_values: List[Any],
        required: bool = False,
    ) -> "SchemaBuilder":
        """Add enum field."""
        return self.field(name, DataType.ANY, required, [EnumValidator(allowed_values)])

    def allow_extra(self, allow: bool = True) -> "SchemaBuilder":
        """Set allow extra fields."""
        self._allow_extra_fields = allow
        return self

    def strict_mode(self, strict: bool = True) -> "SchemaBuilder":
        """Set strict mode."""
        self._strict = strict
        return self

    def build(self) -> Schema:
        """Build schema."""
        return Schema(
            name=self._name,
            fields=self._fields,
            allow_extra_fields=self._allow_extra_fields,
            strict=self._strict,
        )


class DataQualityChecker:
    """Data quality checker."""

    def __init__(self) -> None:
        """Initialize checker."""
        self._checks: List[Tuple[str, Callable[[Any], bool], str]] = []
        self._results: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def add_check(
        self,
        name: str,
        check_func: Callable[[Any], bool],
        description: str = "",
    ) -> None:
        """Add quality check."""
        self._checks.append((name, check_func, description))

    def check(self, data: Any) -> Dict[str, Any]:
        """Run quality checks."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_checks": len(self._checks),
            "passed": 0,
            "failed": 0,
            "checks": [],
        }

        for name, check_func, description in self._checks:
            try:
                passed = check_func(data)
                if passed:
                    results["passed"] += 1
                else:
                    results["failed"] += 1

                results["checks"].append({
                    "name": name,
                    "passed": passed,
                    "description": description,
                })
            except Exception as e:
                results["failed"] += 1
                results["checks"].append({
                    "name": name,
                    "passed": False,
                    "error": str(e),
                    "description": description,
                })

        results["quality_score"] = (
            results["passed"] / results["total_checks"]
            if results["total_checks"] > 0
            else 0.0
        )

        with self._lock:
            self._results.append(results)

        return results

    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get check history."""
        with self._lock:
            return list(self._results[-limit:])


class ValidatedVisionProvider(VisionProvider):
    """Vision provider with data validation."""

    def __init__(
        self,
        provider: VisionProvider,
        input_schema: Optional[Schema] = None,
        output_schema: Optional[Schema] = None,
    ) -> None:
        """Initialize provider."""
        self._provider = provider
        self._input_schema = input_schema
        self._output_schema = output_schema
        self._validation_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"validated_{self._provider.provider_name}"

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image with validation."""
        request_id = str(uuid.uuid4())

        # Validate input
        if self._input_schema:
            input_data = {
                "image_data_size": len(image_data),
                "include_description": include_description,
            }
            input_result = self._input_schema.validate(input_data)
            if not input_result.valid:
                with self._lock:
                    self._validation_history.append({
                        "request_id": request_id,
                        "type": "input",
                        "result": input_result.to_dict(),
                    })
                raise ValueError(f"Input validation failed: {input_result.errors}")

        # Analyze
        result = await self._provider.analyze_image(image_data, include_description)

        # Validate output
        if self._output_schema:
            output_data = {
                "summary": result.summary,
                "details": result.details,
                "confidence": result.confidence,
            }
            output_result = self._output_schema.validate(output_data)

            with self._lock:
                self._validation_history.append({
                    "request_id": request_id,
                    "type": "output",
                    "result": output_result.to_dict(),
                })

            if not output_result.valid:
                raise ValueError(f"Output validation failed: {output_result.errors}")

        return result

    def get_validation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get validation history."""
        with self._lock:
            return list(self._validation_history[-limit:])


def create_schema(name: str) -> SchemaBuilder:
    """Create schema builder.

    Args:
        name: Schema name

    Returns:
        Schema builder
    """
    return SchemaBuilder(name)


def create_validator(
    func: Callable[[Any], bool],
    error_message: str = "Validation failed",
) -> CustomValidator:
    """Create custom validator.

    Args:
        func: Validation function
        error_message: Error message

    Returns:
        Custom validator
    """
    return CustomValidator(func, error_message)


def create_quality_checker() -> DataQualityChecker:
    """Create data quality checker.

    Returns:
        Data quality checker
    """
    return DataQualityChecker()


def create_validated_provider(
    provider: VisionProvider,
    input_schema: Optional[Schema] = None,
    output_schema: Optional[Schema] = None,
) -> ValidatedVisionProvider:
    """Create validated vision provider.

    Args:
        provider: Provider to wrap
        input_schema: Optional input schema
        output_schema: Optional output schema

    Returns:
        Validated provider
    """
    return ValidatedVisionProvider(provider, input_schema, output_schema)
