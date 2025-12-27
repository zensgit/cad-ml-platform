"""Model interface validation for security and compatibility.

This module validates loaded ML model objects to ensure they:
1. Have required interface methods (predict)
2. Don't have excessively large attribute graphs (DoS prevention)
3. Don't have suspicious magic methods (__reduce__, etc.)
4. Have valid method signatures

Usage:
    from src.core.model_interface_validation import validate_model_interface

    result = validate_model_interface(model_obj)
    if not result["valid"]:
        raise SecurityError(f"Invalid model: {result['reason']}")
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# Configuration from environment
MAX_ATTRIBUTES = int(os.getenv("MODEL_MAX_ATTRIBUTES", "1000"))
MAX_ATTRIBUTE_DEPTH = int(os.getenv("MODEL_MAX_ATTRIBUTE_DEPTH", "10"))
MAX_SIGNATURE_PARAMS = int(os.getenv("MODEL_MAX_SIGNATURE_PARAMS", "20"))

# Suspicious magic methods that may indicate pickle exploitation
SUSPICIOUS_METHODS: Set[str] = {
    "__reduce__",
    "__reduce_ex__",
    "__getstate__",
    "__setstate__",
    "__getnewargs__",
    "__getnewargs_ex__",
}

# Required interface methods for ML models
REQUIRED_METHODS: Set[str] = {
    "predict",
}

# Optional but expected methods
OPTIONAL_METHODS: Set[str] = {
    "predict_proba",
    "fit",
    "transform",
    "score",
}


class ValidationResult:
    """Result of model validation."""

    def __init__(
        self,
        valid: bool,
        reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.valid = valid
        self.reason = reason
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "valid": self.valid,
            "reason": self.reason,
            "details": self.details,
        }


def count_attributes(obj: Any, max_depth: int = MAX_ATTRIBUTE_DEPTH) -> int:
    """Count attributes in object graph (with depth limit).

    Prevents infinite recursion and DoS from circular references.

    Args:
        obj: Object to count attributes for
        max_depth: Maximum recursion depth

    Returns:
        Number of attributes counted
    """
    if max_depth <= 0:
        return 0

    seen: Set[int] = set()

    def _count(o: Any, depth: int) -> int:
        if depth <= 0:
            return 0

        obj_id = id(o)
        if obj_id in seen:
            return 0  # Prevent circular reference counting
        seen.add(obj_id)

        count = 0
        try:
            # Count direct attributes
            if hasattr(o, "__dict__"):
                attrs = getattr(o, "__dict__", {})
                count += len(attrs)

                # Recursively count nested objects (limited depth)
                for v in attrs.values():
                    if hasattr(v, "__dict__") and not isinstance(v, type):
                        count += _count(v, depth - 1)
        except Exception:
            pass  # Skip on access errors

        return count

    return _count(obj, max_depth)


def check_suspicious_methods(obj: Any) -> List[str]:
    """Check for suspicious magic methods on model object.

    Args:
        obj: Object to check

    Returns:
        List of suspicious method names found
    """
    found = []

    for method_name in SUSPICIOUS_METHODS:
        try:
            # Check if method exists and is custom (not from base object)
            if hasattr(obj, method_name):
                method = getattr(obj, method_name)
                # Check if it's defined on the class, not inherited from object
                obj_class = type(obj)
                if hasattr(obj_class, method_name):
                    class_method = getattr(obj_class, method_name)
                    base_method = getattr(object, method_name, None)

                    # If different from base object's method, it's custom
                    if class_method != base_method:
                        found.append(method_name)
        except Exception:
            pass

    return found


def validate_predict_signature(obj: Any) -> Dict[str, Any]:
    """Validate the predict method signature.

    Expected: predict(X) where X is array-like (list of feature vectors)

    Returns:
        Validation result with signature details
    """
    if not hasattr(obj, "predict"):
        return {"valid": False, "reason": "no_predict_method"}

    predict_method = getattr(obj, "predict")

    try:
        sig = inspect.signature(predict_method)
        params = list(sig.parameters.keys())

        # Remove 'self' if present (bound method)
        if params and params[0] == "self":
            params = params[1:]

        result = {
            "valid": True,
            "params": params,
            "param_count": len(params),
        }

        # Check parameter count (should have at least 1 for input X)
        if len(params) < 1:
            result["valid"] = False
            result["reason"] = "predict_needs_input_param"

        # Check for excessive parameters (potential exploit vector)
        if len(params) > MAX_SIGNATURE_PARAMS:
            result["valid"] = False
            result["reason"] = "too_many_params"

        # Check first parameter name (common conventions)
        if params:
            first_param = params[0]
            common_names = {"X", "x", "data", "input", "inputs", "features"}
            result["first_param_name"] = first_param
            result["conventional_name"] = first_param in common_names

        return result

    except (ValueError, TypeError) as e:
        return {"valid": False, "reason": f"signature_error: {str(e)}"}


def validate_model_interface(
    model: Any,
    check_attributes: bool = True,
    check_methods: bool = True,
    check_signature: bool = True,
) -> ValidationResult:
    """Validate a model object's interface for security and compatibility.

    Performs multiple validation checks:
    1. Required methods exist (predict)
    2. Attribute graph is not excessively large
    3. No suspicious magic methods defined
    4. Predict method has valid signature

    Args:
        model: The model object to validate
        check_attributes: Whether to check attribute count
        check_methods: Whether to check for suspicious methods
        check_signature: Whether to validate predict signature

    Returns:
        ValidationResult with validation status and details
    """
    details: Dict[str, Any] = {
        "model_type": type(model).__name__,
        "model_module": type(model).__module__,
    }

    # Check required methods
    missing_methods = []
    for method in REQUIRED_METHODS:
        if not hasattr(model, method) or not callable(getattr(model, method)):
            missing_methods.append(method)

    if missing_methods:
        details["missing_methods"] = missing_methods
        return ValidationResult(
            valid=False,
            reason="missing_required_methods",
            details=details,
        )

    # Record which optional methods are available
    available_optional = [m for m in OPTIONAL_METHODS if hasattr(model, m)]
    details["available_methods"] = list(REQUIRED_METHODS) + available_optional

    # Check attribute count (DoS prevention)
    if check_attributes:
        attr_count = count_attributes(model)
        details["attribute_count"] = attr_count

        if attr_count > MAX_ATTRIBUTES:
            details["max_attributes"] = MAX_ATTRIBUTES
            return ValidationResult(
                valid=False,
                reason="large_attribute_graph",
                details=details,
            )

    # Check suspicious methods
    if check_methods:
        suspicious = check_suspicious_methods(model)
        details["suspicious_methods"] = suspicious

        # Note: Having these methods is not necessarily bad (sklearn uses them)
        # but we flag them for awareness. Critical decision is made by mode config.
        if suspicious:
            strict_mode = os.getenv("MODEL_INTERFACE_STRICT", "0") == "1"
            if strict_mode:
                return ValidationResult(
                    valid=False,
                    reason="suspicious_methods_found",
                    details=details,
                )
            else:
                # Log warning but allow
                logger.warning(
                    "Model has suspicious methods (allowed in non-strict mode)",
                    extra={"methods": suspicious, "model_type": details["model_type"]},
                )

    # Validate predict signature
    if check_signature:
        sig_result = validate_predict_signature(model)
        details["predict_signature"] = sig_result

        if not sig_result.get("valid", True):
            return ValidationResult(
                valid=False,
                reason=f"invalid_signature: {sig_result.get('reason', 'unknown')}",
                details=details,
            )

    # All validations passed
    return ValidationResult(valid=True, details=details)


def get_validation_config() -> Dict[str, Any]:
    """Get current validation configuration."""
    return {
        "max_attributes": MAX_ATTRIBUTES,
        "max_attribute_depth": MAX_ATTRIBUTE_DEPTH,
        "max_signature_params": MAX_SIGNATURE_PARAMS,
        "suspicious_methods": sorted(SUSPICIOUS_METHODS),
        "required_methods": sorted(REQUIRED_METHODS),
        "optional_methods": sorted(OPTIONAL_METHODS),
        "strict_mode": os.getenv("MODEL_INTERFACE_STRICT", "0") == "1",
    }


__all__ = [
    "validate_model_interface",
    "ValidationResult",
    "count_attributes",
    "check_suspicious_methods",
    "validate_predict_signature",
    "get_validation_config",
    "SUSPICIOUS_METHODS",
    "REQUIRED_METHODS",
    "MAX_ATTRIBUTES",
]
