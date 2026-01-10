"""Tests for model interface validation module.

Verifies that:
1. Valid models pass validation
2. Large attribute graphs are rejected
3. Suspicious magic methods are detected
4. Invalid signatures are rejected
5. Configuration is correctly exposed
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from src.core.model_interface_validation import (
    MAX_ATTRIBUTES,
    REQUIRED_METHODS,
    SUSPICIOUS_METHODS,
    ValidationResult,
    check_suspicious_methods,
    count_attributes,
    get_validation_config,
    validate_model_interface,
    validate_predict_signature,
)


class MockValidModel:
    """A valid sklearn-like model for testing."""

    def __init__(self):
        self.coef_ = [1.0, 2.0, 3.0]
        self.intercept_ = 0.5
        self.classes_ = ["A", "B"]

    def predict(self, X):
        """Predict labels."""
        return ["A"] * len(X)

    def predict_proba(self, X):
        """Predict probabilities."""
        return [[0.6, 0.4]] * len(X)


class MockModelMissingPredict:
    """Model without predict method."""

    def __init__(self):
        self.data = [1, 2, 3]

    def transform(self, X):
        return X


class MockModelBadSignature:
    """Model with unusual predict signature."""

    def predict(self):
        """Predict with no input parameter."""
        return ["A"]


class MockModelSuspicious:
    """Model with suspicious __reduce__ method."""

    def __init__(self):
        self.data = "test"

    def predict(self, X):
        return ["A"] * len(X)

    def __reduce__(self):
        """Custom reduce - potentially dangerous."""
        return (str, ("exploit",))


class MockLargeModel:
    """Model with many attributes for DoS testing."""

    def __init__(self, num_attrs: int = 2000):
        for i in range(num_attrs):
            setattr(self, f"attr_{i}", i)

    def predict(self, X):
        return ["A"] * len(X)


class TestValidModelValidation:
    """Test validation of valid models."""

    def test_valid_model_passes(self):
        """Test valid sklearn-like model passes validation."""
        model = MockValidModel()
        result = validate_model_interface(model)

        assert result.valid is True
        assert result.reason is None
        assert "model_type" in result.details
        assert result.details["model_type"] == "MockValidModel"

    def test_valid_model_has_required_methods(self):
        """Test valid model has required methods recorded."""
        model = MockValidModel()
        result = validate_model_interface(model)

        assert "available_methods" in result.details
        assert "predict" in result.details["available_methods"]

    def test_valid_model_attribute_count_recorded(self):
        """Test attribute count is recorded."""
        model = MockValidModel()
        result = validate_model_interface(model)

        assert "attribute_count" in result.details
        assert result.details["attribute_count"] > 0

    def test_valid_model_predict_signature_recorded(self):
        """Test predict signature is recorded."""
        model = MockValidModel()
        result = validate_model_interface(model)

        assert "predict_signature" in result.details
        sig = result.details["predict_signature"]
        assert sig["valid"] is True
        assert "params" in sig


class TestMissingMethodsValidation:
    """Test validation fails for missing methods."""

    def test_missing_predict_fails(self):
        """Test model without predict method fails."""
        model = MockModelMissingPredict()
        result = validate_model_interface(model)

        assert result.valid is False
        assert result.reason == "missing_required_methods"
        assert "missing_methods" in result.details
        assert "predict" in result.details["missing_methods"]

    def test_none_model_fails(self):
        """Test None model fails validation."""
        result = validate_model_interface(None)

        assert result.valid is False


class TestLargeAttributeGraph:
    """Test rejection of large attribute graphs."""

    def test_large_model_rejected(self):
        """Test model with too many attributes is rejected."""
        model = MockLargeModel(num_attrs=MAX_ATTRIBUTES + 100)
        result = validate_model_interface(model)

        assert result.valid is False
        assert result.reason == "large_attribute_graph"
        assert result.details["attribute_count"] > MAX_ATTRIBUTES

    def test_normal_model_attribute_count_ok(self):
        """Test normal model passes attribute check."""
        model = MockValidModel()
        result = validate_model_interface(model)

        assert result.valid is True
        assert result.details["attribute_count"] < MAX_ATTRIBUTES

    def test_skip_attribute_check(self):
        """Test attribute check can be skipped."""
        model = MockLargeModel(num_attrs=MAX_ATTRIBUTES + 100)
        result = validate_model_interface(model, check_attributes=False)

        # Should pass because attribute check is skipped
        assert result.valid is True


class TestSuspiciousMethods:
    """Test detection of suspicious methods."""

    def test_detects_reduce_method(self):
        """Test __reduce__ is detected."""
        model = MockModelSuspicious()
        result = validate_model_interface(model)

        assert "suspicious_methods" in result.details
        assert "__reduce__" in result.details["suspicious_methods"]

    def test_strict_mode_rejects_suspicious(self):
        """Test strict mode rejects models with suspicious methods."""
        model = MockModelSuspicious()

        with patch.dict(os.environ, {"MODEL_INTERFACE_STRICT": "1"}):
            result = validate_model_interface(model)

            assert result.valid is False
            assert result.reason == "suspicious_methods_found"

    def test_non_strict_mode_allows_suspicious(self):
        """Test non-strict mode allows suspicious methods with warning."""
        model = MockModelSuspicious()

        with patch.dict(os.environ, {"MODEL_INTERFACE_STRICT": "0"}):
            result = validate_model_interface(model)

            # Should pass but suspicious methods are recorded
            assert result.valid is True
            assert "__reduce__" in result.details["suspicious_methods"]

    def test_valid_model_no_suspicious_methods(self):
        """Test valid model has no custom suspicious methods."""
        model = MockValidModel()
        suspicious = check_suspicious_methods(model)

        # MockValidModel doesn't define __reduce__, etc.
        assert len(suspicious) == 0


class TestPredictSignature:
    """Test predict signature validation."""

    def test_valid_signature(self):
        """Test valid predict(X) signature passes."""
        model = MockValidModel()
        result = validate_predict_signature(model)

        assert result["valid"] is True
        assert "X" in result["params"] or "x" in result.get("params", [])

    def test_no_param_signature_fails(self):
        """Test predict() with no params fails."""
        model = MockModelBadSignature()
        result = validate_predict_signature(model)

        assert result["valid"] is False
        assert "needs_input_param" in result["reason"]

    def test_missing_predict_method(self):
        """Test missing predict method is detected."""
        model = MockModelMissingPredict()
        result = validate_predict_signature(model)

        assert result["valid"] is False
        assert result["reason"] == "no_predict_method"


class TestCountAttributes:
    """Test attribute counting function."""

    def test_count_simple_object(self):
        """Test counting attributes on simple object."""
        model = MockValidModel()
        count = count_attributes(model)

        # Should count coef_, intercept_, classes_
        assert count >= 3

    def test_count_respects_depth_limit(self):
        """Test depth limit is respected."""

        class NestedModel:
            def __init__(self, depth: int):
                self.data = "test"
                if depth > 0:
                    self.nested = NestedModel(depth - 1)

        model = NestedModel(depth=20)

        # With default depth limit, shouldn't count all nested attributes
        count = count_attributes(model, max_depth=3)
        assert count < 100  # Much less than full depth

    def test_count_handles_circular_refs(self):
        """Test circular references don't cause infinite loop."""

        class CircularModel:
            def __init__(self):
                self.data = "test"
                self.self_ref = self

        model = CircularModel()
        count = count_attributes(model)

        # Should not hang and return reasonable count
        assert count >= 1
        assert count < 100


class TestValidationResult:
    """Test ValidationResult class."""

    def test_to_dict_valid(self):
        """Test to_dict for valid result."""
        result = ValidationResult(valid=True, details={"test": "data"})
        d = result.to_dict()

        assert d["valid"] is True
        assert d["reason"] is None
        assert d["details"]["test"] == "data"

    def test_to_dict_invalid(self):
        """Test to_dict for invalid result."""
        result = ValidationResult(
            valid=False,
            reason="test_failure",
            details={"error": "details"},
        )
        d = result.to_dict()

        assert d["valid"] is False
        assert d["reason"] == "test_failure"
        assert d["details"]["error"] == "details"


class TestGetValidationConfig:
    """Test configuration retrieval."""

    def test_config_structure(self):
        """Test config has expected fields."""
        config = get_validation_config()

        assert "max_attributes" in config
        assert "max_attribute_depth" in config
        assert "max_signature_params" in config
        assert "suspicious_methods" in config
        assert "required_methods" in config
        assert "strict_mode" in config

    def test_suspicious_methods_list(self):
        """Test suspicious methods list is complete."""
        config = get_validation_config()

        assert "__reduce__" in config["suspicious_methods"]
        assert "__reduce_ex__" in config["suspicious_methods"]

    def test_required_methods_list(self):
        """Test required methods list."""
        config = get_validation_config()

        assert "predict" in config["required_methods"]


class TestEnvironmentConfiguration:
    """Test environment variable configuration."""

    def test_custom_max_attributes(self):
        """Test custom MAX_ATTRIBUTES from env."""
        # This tests the module-level constant which is set at import time
        # So we test the config function returns the current value
        config = get_validation_config()
        assert isinstance(config["max_attributes"], int)

    def test_strict_mode_config(self):
        """Test strict mode configuration."""
        with patch.dict(os.environ, {"MODEL_INTERFACE_STRICT": "1"}):
            config = get_validation_config()
            assert config["strict_mode"] is True

        with patch.dict(os.environ, {"MODEL_INTERFACE_STRICT": "0"}):
            config = get_validation_config()
            assert config["strict_mode"] is False


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_builtin_type(self):
        """Test validation of builtin type fails gracefully."""
        result = validate_model_interface([1, 2, 3])

        assert result.valid is False

    def test_callable_without_predict(self):
        """Test callable object without predict fails."""

        class CallableModel:
            def __call__(self, X):
                return X

        model = CallableModel()
        result = validate_model_interface(model)

        assert result.valid is False
        assert "predict" in result.details.get("missing_methods", [])

    def test_model_with_non_callable_predict(self):
        """Test model where predict is not callable."""

        class BadPredictModel:
            predict = "not a method"

        model = BadPredictModel()
        result = validate_model_interface(model)

        assert result.valid is False

    def test_sklearn_like_model(self):
        """Test sklearn-like model structure."""

        class SklearnLikeModel:
            def __init__(self):
                self.coef_ = [[0.1, 0.2], [0.3, 0.4]]
                self.intercept_ = [0.5, 0.6]
                self.classes_ = [0, 1]
                self.n_features_in_ = 2

            def predict(self, X):
                return [0] * len(X)

            def predict_proba(self, X):
                return [[0.5, 0.5]] * len(X)

            def fit(self, X, y):
                return self

        model = SklearnLikeModel()
        result = validate_model_interface(model)

        assert result.valid is True
        assert "predict" in result.details["available_methods"]
        assert "predict_proba" in result.details["available_methods"]


class TestSelectiveValidation:
    """Test selective validation options."""

    def test_skip_all_checks(self):
        """Test skipping all checks."""
        model = MockLargeModel(num_attrs=5000)
        result = validate_model_interface(
            model,
            check_attributes=False,
            check_methods=False,
            check_signature=False,
        )

        # Only checks required methods exist
        assert result.valid is True

    def test_only_signature_check(self):
        """Test only signature validation."""
        model = MockLargeModel(num_attrs=5000)
        result = validate_model_interface(
            model,
            check_attributes=False,
            check_methods=False,
            check_signature=True,
        )

        assert result.valid is True
        assert "predict_signature" in result.details

    def test_only_attribute_check(self):
        """Test only attribute validation."""
        model = MockLargeModel(num_attrs=5000)
        result = validate_model_interface(
            model,
            check_attributes=True,
            check_methods=False,
            check_signature=False,
        )

        assert result.valid is False
        assert result.reason == "large_attribute_graph"
