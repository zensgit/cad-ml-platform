"""Response validation module for Vision Provider system.

This module provides schema validation and quality assurance for provider
responses including:
- Response schema validation
- Field completeness checks
- Confidence thresholds
- Custom validation rules
- Validation reporting
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Union

from .base import VisionDescription, VisionProvider


class ValidationSeverity(Enum):
    """Severity level of validation issues."""

    ERROR = "error"  # Critical issue, response invalid
    WARNING = "warning"  # Issue but response usable
    INFO = "info"  # Informational note


class ValidationRuleType(Enum):
    """Types of validation rules."""

    REQUIRED_FIELD = "required_field"
    MIN_LENGTH = "min_length"
    MAX_LENGTH = "max_length"
    PATTERN = "pattern"
    RANGE = "range"
    CUSTOM = "custom"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    LIST_MIN_ITEMS = "list_min_items"
    LIST_MAX_ITEMS = "list_max_items"


@dataclass
class ValidationIssue:
    """A single validation issue."""

    rule_name: str
    field: str
    message: str
    severity: ValidationSeverity
    actual_value: Any = None
    expected_value: Any = None


@dataclass
class ValidationResult:
    """Result of validation checks."""

    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    validated_at: datetime = field(default_factory=datetime.now)
    validation_time_ms: float = 0.0

    @property
    def errors(self) -> List[ValidationIssue]:
        """Get only error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get only warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0


@dataclass
class ValidationRule:
    """A validation rule definition."""

    name: str
    rule_type: ValidationRuleType
    field: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    params: Dict[str, Any] = field(default_factory=dict)
    message: Optional[str] = None
    enabled: bool = True


@dataclass
class ValidationSchema:
    """Schema for validating responses."""

    name: str
    description: str = ""
    rules: List[ValidationRule] = field(default_factory=list)
    fail_fast: bool = False  # Stop on first error
    strict_mode: bool = False  # Treat warnings as errors


class ResponseValidator:
    """Validates vision provider responses."""

    def __init__(self, schema: Optional[ValidationSchema] = None) -> None:
        """Initialize the validator.

        Args:
            schema: Optional validation schema
        """
        self._schema = schema
        self._custom_validators: Dict[str, Callable[[Any, Dict[str, Any]], Optional[str]]] = {}

    def set_schema(self, schema: ValidationSchema) -> None:
        """Set the validation schema."""
        self._schema = schema

    def register_custom_validator(
        self,
        name: str,
        validator: Callable[[Any, Dict[str, Any]], Optional[str]],
    ) -> None:
        """Register a custom validation function.

        Args:
            name: Validator name
            validator: Function that returns error message or None if valid
        """
        self._custom_validators[name] = validator

    def validate(self, response: VisionDescription) -> ValidationResult:
        """Validate a vision response.

        Args:
            response: The response to validate

        Returns:
            Validation result with any issues found
        """
        import time

        start_time = time.time()
        issues: List[ValidationIssue] = []

        if not self._schema:
            return ValidationResult(
                is_valid=True,
                validation_time_ms=(time.time() - start_time) * 1000,
            )

        # Convert response to dict for validation
        response_dict = self._response_to_dict(response)

        for rule in self._schema.rules:
            if not rule.enabled:
                continue

            issue = self._apply_rule(rule, response_dict)
            if issue:
                issues.append(issue)

                if self._schema.fail_fast and issue.severity == ValidationSeverity.ERROR:
                    break

        # Determine validity
        is_valid = not any(i.severity == ValidationSeverity.ERROR for i in issues)

        if self._schema.strict_mode:
            is_valid = len(issues) == 0

        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            validation_time_ms=(time.time() - start_time) * 1000,
        )

    def _response_to_dict(self, response: VisionDescription) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "summary": response.summary,
            "confidence": response.confidence,
            "details": response.details,
        }

    def _apply_rule(
        self, rule: ValidationRule, response: Dict[str, Any]
    ) -> Optional[ValidationIssue]:
        """Apply a single validation rule."""
        value = self._get_field_value(response, rule.field)

        if rule.rule_type == ValidationRuleType.REQUIRED_FIELD:
            return self._validate_required(rule, value)
        elif rule.rule_type == ValidationRuleType.MIN_LENGTH:
            return self._validate_min_length(rule, value)
        elif rule.rule_type == ValidationRuleType.MAX_LENGTH:
            return self._validate_max_length(rule, value)
        elif rule.rule_type == ValidationRuleType.PATTERN:
            return self._validate_pattern(rule, value)
        elif rule.rule_type == ValidationRuleType.RANGE:
            return self._validate_range(rule, value)
        elif rule.rule_type == ValidationRuleType.CONFIDENCE_THRESHOLD:
            return self._validate_confidence(rule, value)
        elif rule.rule_type == ValidationRuleType.LIST_MIN_ITEMS:
            return self._validate_list_min(rule, value)
        elif rule.rule_type == ValidationRuleType.LIST_MAX_ITEMS:
            return self._validate_list_max(rule, value)
        elif rule.rule_type == ValidationRuleType.CUSTOM:
            return self._validate_custom(rule, value, response)

        return None

    def _get_field_value(self, data: Dict[str, Any], field: str) -> Any:
        """Get a field value using dot notation."""
        parts = field.split(".")
        value = data

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif isinstance(value, list) and part.isdigit():
                idx = int(part)
                value = value[idx] if idx < len(value) else None
            else:
                return None

        return value

    def _validate_required(self, rule: ValidationRule, value: Any) -> Optional[ValidationIssue]:
        """Validate required field."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return ValidationIssue(
                rule_name=rule.name,
                field=rule.field,
                message=rule.message or f"Field '{rule.field}' is required",
                severity=rule.severity,
                actual_value=value,
                expected_value="non-empty value",
            )
        return None

    def _validate_min_length(self, rule: ValidationRule, value: Any) -> Optional[ValidationIssue]:
        """Validate minimum length."""
        min_len = rule.params.get("min_length", 0)

        if value is None:
            return None

        actual_len = len(value) if hasattr(value, "__len__") else 0

        if actual_len < min_len:
            return ValidationIssue(
                rule_name=rule.name,
                field=rule.field,
                message=rule.message
                or f"Field '{rule.field}' must be at least {min_len} characters",
                severity=rule.severity,
                actual_value=actual_len,
                expected_value=f">= {min_len}",
            )
        return None

    def _validate_max_length(self, rule: ValidationRule, value: Any) -> Optional[ValidationIssue]:
        """Validate maximum length."""
        max_len = rule.params.get("max_length", float("inf"))

        if value is None:
            return None

        actual_len = len(value) if hasattr(value, "__len__") else 0

        if actual_len > max_len:
            return ValidationIssue(
                rule_name=rule.name,
                field=rule.field,
                message=rule.message
                or f"Field '{rule.field}' must be at most {max_len} characters",
                severity=rule.severity,
                actual_value=actual_len,
                expected_value=f"<= {max_len}",
            )
        return None

    def _validate_pattern(self, rule: ValidationRule, value: Any) -> Optional[ValidationIssue]:
        """Validate against regex pattern."""
        pattern = rule.params.get("pattern", "")

        if value is None or not isinstance(value, str):
            return None

        if not re.match(pattern, value):
            return ValidationIssue(
                rule_name=rule.name,
                field=rule.field,
                message=rule.message or f"Field '{rule.field}' does not match required pattern",
                severity=rule.severity,
                actual_value=value,
                expected_value=f"pattern: {pattern}",
            )
        return None

    def _validate_range(self, rule: ValidationRule, value: Any) -> Optional[ValidationIssue]:
        """Validate numeric range."""
        min_val = rule.params.get("min")
        max_val = rule.params.get("max")

        if value is None:
            return None

        try:
            num_value = float(value)
        except (TypeError, ValueError):
            return ValidationIssue(
                rule_name=rule.name,
                field=rule.field,
                message=f"Field '{rule.field}' must be numeric",
                severity=rule.severity,
                actual_value=value,
                expected_value="numeric value",
            )

        if min_val is not None and num_value < min_val:
            return ValidationIssue(
                rule_name=rule.name,
                field=rule.field,
                message=rule.message or f"Field '{rule.field}' must be >= {min_val}",
                severity=rule.severity,
                actual_value=num_value,
                expected_value=f">= {min_val}",
            )

        if max_val is not None and num_value > max_val:
            return ValidationIssue(
                rule_name=rule.name,
                field=rule.field,
                message=rule.message or f"Field '{rule.field}' must be <= {max_val}",
                severity=rule.severity,
                actual_value=num_value,
                expected_value=f"<= {max_val}",
            )

        return None

    def _validate_confidence(self, rule: ValidationRule, value: Any) -> Optional[ValidationIssue]:
        """Validate confidence threshold."""
        threshold = rule.params.get("threshold", 0.5)

        if value is None:
            return None

        try:
            conf_value = float(value)
        except (TypeError, ValueError):
            return ValidationIssue(
                rule_name=rule.name,
                field=rule.field,
                message="Confidence must be numeric",
                severity=rule.severity,
                actual_value=value,
                expected_value="numeric value",
            )

        if conf_value < threshold:
            return ValidationIssue(
                rule_name=rule.name,
                field=rule.field,
                message=rule.message or f"Confidence {conf_value:.2f} below threshold {threshold}",
                severity=rule.severity,
                actual_value=conf_value,
                expected_value=f">= {threshold}",
            )

        return None

    def _validate_list_min(self, rule: ValidationRule, value: Any) -> Optional[ValidationIssue]:
        """Validate minimum list items."""
        min_items = rule.params.get("min_items", 0)

        if value is None:
            value = []

        if not isinstance(value, list):
            return ValidationIssue(
                rule_name=rule.name,
                field=rule.field,
                message=f"Field '{rule.field}' must be a list",
                severity=rule.severity,
                actual_value=type(value).__name__,
                expected_value="list",
            )

        if len(value) < min_items:
            return ValidationIssue(
                rule_name=rule.name,
                field=rule.field,
                message=rule.message
                or f"Field '{rule.field}' must have at least {min_items} items",
                severity=rule.severity,
                actual_value=len(value),
                expected_value=f">= {min_items}",
            )

        return None

    def _validate_list_max(self, rule: ValidationRule, value: Any) -> Optional[ValidationIssue]:
        """Validate maximum list items."""
        max_items = rule.params.get("max_items", float("inf"))

        if value is None or not isinstance(value, list):
            return None

        if len(value) > max_items:
            return ValidationIssue(
                rule_name=rule.name,
                field=rule.field,
                message=rule.message or f"Field '{rule.field}' must have at most {max_items} items",
                severity=rule.severity,
                actual_value=len(value),
                expected_value=f"<= {max_items}",
            )

        return None

    def _validate_custom(
        self, rule: ValidationRule, value: Any, response: Dict[str, Any]
    ) -> Optional[ValidationIssue]:
        """Apply custom validation."""
        validator_name = rule.params.get("validator")

        if not validator_name or validator_name not in self._custom_validators:
            return None

        validator = self._custom_validators[validator_name]
        error_message = validator(value, rule.params)

        if error_message:
            return ValidationIssue(
                rule_name=rule.name,
                field=rule.field,
                message=error_message,
                severity=rule.severity,
                actual_value=value,
            )

        return None


class ValidatingVisionProvider(VisionProvider):
    """Vision provider wrapper that validates responses."""

    def __init__(
        self,
        provider: VisionProvider,
        validator: ResponseValidator,
        reject_invalid: bool = True,
        on_validation_failure: Optional[
            Callable[[VisionDescription, ValidationResult], VisionDescription]
        ] = None,
    ) -> None:
        """Initialize the validating provider.

        Args:
            provider: The underlying provider
            validator: The response validator
            reject_invalid: Whether to raise on invalid responses
            on_validation_failure: Optional handler for invalid responses
        """
        self._provider = provider
        self._validator = validator
        self._reject_invalid = reject_invalid
        self._on_failure = on_validation_failure
        self._validation_stats = ValidationStats()

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return f"validating_{self._provider.provider_name}"

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image with response validation.

        Args:
            image_data: Raw image bytes
            include_description: Whether to include description

        Returns:
            Validated vision analysis description

        Raises:
            ValidationError: If response is invalid and reject_invalid is True
        """
        response = await self._provider.analyze_image(image_data, include_description)

        result = self._validator.validate(response)
        self._validation_stats.record(result)

        if not result.is_valid:
            if self._on_failure:
                return self._on_failure(response, result)

            if self._reject_invalid:
                raise ValidationError(
                    f"Response validation failed: {len(result.errors)} errors",
                    result,
                )

        return response

    def get_validation_stats(self) -> "ValidationStats":
        """Get validation statistics."""
        return self._validation_stats


@dataclass
class ValidationStats:
    """Statistics about validation results."""

    total_validations: int = 0
    passed_validations: int = 0
    failed_validations: int = 0
    total_errors: int = 0
    total_warnings: int = 0
    errors_by_rule: Dict[str, int] = field(default_factory=dict)
    warnings_by_rule: Dict[str, int] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        """Calculate validation pass rate."""
        if self.total_validations == 0:
            return 0.0
        return self.passed_validations / self.total_validations

    def record(self, result: ValidationResult) -> None:
        """Record a validation result."""
        self.total_validations += 1

        if result.is_valid:
            self.passed_validations += 1
        else:
            self.failed_validations += 1

        for issue in result.issues:
            if issue.severity == ValidationSeverity.ERROR:
                self.total_errors += 1
                self.errors_by_rule[issue.rule_name] = (
                    self.errors_by_rule.get(issue.rule_name, 0) + 1
                )
            elif issue.severity == ValidationSeverity.WARNING:
                self.total_warnings += 1
                self.warnings_by_rule[issue.rule_name] = (
                    self.warnings_by_rule.get(issue.rule_name, 0) + 1
                )


class ValidationError(Exception):
    """Error raised when validation fails."""

    def __init__(self, message: str, result: ValidationResult) -> None:
        """Initialize the error."""
        super().__init__(message)
        self.result = result


# Pre-built validation schemas
STANDARD_SCHEMA = ValidationSchema(
    name="standard",
    description="Standard validation schema for vision responses",
    rules=[
        ValidationRule(
            name="summary_required",
            rule_type=ValidationRuleType.REQUIRED_FIELD,
            field="summary",
            severity=ValidationSeverity.ERROR,
            message="Summary is required",
        ),
        ValidationRule(
            name="summary_min_length",
            rule_type=ValidationRuleType.MIN_LENGTH,
            field="summary",
            params={"min_length": 10},
            severity=ValidationSeverity.WARNING,
            message="Summary should be at least 10 characters",
        ),
        ValidationRule(
            name="confidence_range",
            rule_type=ValidationRuleType.RANGE,
            field="confidence",
            params={"min": 0.0, "max": 1.0},
            severity=ValidationSeverity.ERROR,
            message="Confidence must be between 0 and 1",
        ),
        ValidationRule(
            name="confidence_threshold",
            rule_type=ValidationRuleType.CONFIDENCE_THRESHOLD,
            field="confidence",
            params={"threshold": 0.5},
            severity=ValidationSeverity.WARNING,
            message="Low confidence response",
        ),
    ],
)

STRICT_SCHEMA = ValidationSchema(
    name="strict",
    description="Strict validation schema with high confidence requirements",
    rules=[
        ValidationRule(
            name="summary_required",
            rule_type=ValidationRuleType.REQUIRED_FIELD,
            field="summary",
            severity=ValidationSeverity.ERROR,
        ),
        ValidationRule(
            name="summary_min_length",
            rule_type=ValidationRuleType.MIN_LENGTH,
            field="summary",
            params={"min_length": 50},
            severity=ValidationSeverity.ERROR,
            message="Summary must be at least 50 characters",
        ),
        ValidationRule(
            name="confidence_high",
            rule_type=ValidationRuleType.CONFIDENCE_THRESHOLD,
            field="confidence",
            params={"threshold": 0.8},
            severity=ValidationSeverity.ERROR,
            message="Confidence must be at least 0.8",
        ),
        ValidationRule(
            name="details_required",
            rule_type=ValidationRuleType.LIST_MIN_ITEMS,
            field="details",
            params={"min_items": 1},
            severity=ValidationSeverity.ERROR,
            message="At least one detail is required",
        ),
    ],
    strict_mode=True,
)

OCR_SCHEMA = ValidationSchema(
    name="ocr",
    description="Validation schema for OCR-focused responses",
    rules=[
        ValidationRule(
            name="summary_required",
            rule_type=ValidationRuleType.REQUIRED_FIELD,
            field="summary",
            severity=ValidationSeverity.ERROR,
            message="Summary is required for OCR response",
        ),
        ValidationRule(
            name="details_required",
            rule_type=ValidationRuleType.LIST_MIN_ITEMS,
            field="details",
            params={"min_items": 1},
            severity=ValidationSeverity.WARNING,
            message="Details should contain OCR extracted text",
        ),
    ],
)


def create_validating_provider(
    provider: VisionProvider,
    schema: Optional[ValidationSchema] = None,
    reject_invalid: bool = True,
) -> ValidatingVisionProvider:
    """Create a validating provider wrapper.

    Args:
        provider: The underlying provider
        schema: Validation schema (defaults to STANDARD_SCHEMA)
        reject_invalid: Whether to raise on invalid responses

    Returns:
        Validating provider wrapper
    """
    validator = ResponseValidator(schema or STANDARD_SCHEMA)
    return ValidatingVisionProvider(
        provider=provider,
        validator=validator,
        reject_invalid=reject_invalid,
    )
