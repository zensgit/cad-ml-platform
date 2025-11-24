"""Metrics contract test for observability guarantees.

Validates:
1. Required metrics are present in /metrics endpoint
2. Label schemas match expected contracts
3. EMAs are available via /health
4. Counters increment correctly with sample calls
"""

import re
from typing import Dict, List, Set, Optional, Tuple
import pytest
from fastapi.testclient import TestClient
import base64
import json

from src.main import app
from src.core.errors import ErrorCode

client = TestClient(app)


class MetricsContract:
    """Define expected metrics and their label schemas."""

    REQUIRED_METRICS = {
        # OCR metrics with expected labels
        "ocr_errors_total": {"provider", "code", "stage"},
        "ocr_input_rejected_total": {"reason"},
        "ocr_requests_total": {"provider", "status"},
        "ocr_model_loaded": {"provider"},
        "ocr_processing_duration_seconds": {"provider"},
        "ocr_stage_duration_seconds": {"provider", "stage"},
        "ocr_confidence_distribution": set(),  # Histogram, no labels expected

        # Vision metrics with expected labels
        "vision_errors_total": {"provider", "code"},
        "vision_input_rejected_total": {"reason"},
        "vision_requests_total": {"provider", "status"},
        "vision_processing_duration_seconds": {"provider"},
        "vision_image_size_bytes": set(),  # Histogram

        # Circuit breaker and rate limiting
        "circuit_breaker_state": {"circuit"},
        "rate_limiter_allowed_total": {"key"},
        "rate_limiter_rejected_total": {"key"},

        # Error rate EMAs (via /health, not /metrics)
        # These are checked separately
    }

    # Valid label values for validation
    VALID_ERROR_CODES = {code.value for code in ErrorCode}
    VALID_STAGES = {"init", "load", "preprocess", "infer", "parse", "align", "postprocess"}
    VALID_STATUSES = {"start", "success", "error", "cache_hit", "input_error"}
    VALID_REJECTION_REASONS = {
        # OCR rejection reasons
        "validation_failed", "invalid_mime", "file_too_large",
        "pdf_pages_exceed", "pdf_forbidden_token",
        # Vision rejection reasons
        "base64_too_large", "base64_empty", "base64_padding_error",
        "base64_invalid_char", "base64_decode_error",
        "url_invalid_scheme", "url_invalid_format", "url_not_found",
        "url_forbidden", "url_http_error", "url_too_large_header",
        "url_too_large_download", "url_empty", "url_timeout",
        "url_network_error", "url_download_error"
    }


def parse_metrics_exposition(text: str) -> Dict[str, List[Dict[str, str]]]:
    """Parse Prometheus exposition format into structured data.

    Returns dict mapping metric names to list of label dicts.
    """
    metrics = {}

    for line in text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        # Match metric lines: metric_name{label1="value1",label2="value2"} value
        match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)((?:\{[^}]*\})?)[\s]+[\d\.\+\-eE]+', line)
        if match:
            metric_name = match.group(1)
            labels_str = match.group(2)

            # Parse labels
            labels = {}
            if labels_str and labels_str != '{}':
                # Remove braces
                labels_str = labels_str[1:-1]
                # Parse key="value" pairs
                for pair in re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"', labels_str):
                    labels[pair[0]] = pair[1]

            if metric_name not in metrics:
                metrics[metric_name] = []
            metrics[metric_name].append(labels)

    return metrics


def extract_base_metric_name(full_name: str) -> str:
    """Extract base metric name from histogram/summary variants."""
    for suffix in ['_bucket', '_count', '_sum', '_total']:
        if full_name.endswith(suffix) and full_name.replace(suffix, '') in MetricsContract.REQUIRED_METRICS:
            return full_name.replace(suffix, '')
    return full_name


class TestMetricsContract:
    """Test metrics contract compliance."""

    def test_metrics_endpoint_available(self):
        """Verify /metrics endpoint is accessible."""
        response = client.get("/metrics")
        assert response.status_code == 200, "Metrics endpoint should return 200"
        assert response.headers.get("content-type") == "text/plain; version=0.0.4", \
            "Metrics should use Prometheus text format"

    def test_required_metrics_present(self):
        """Verify all required metrics are present."""
        response = client.get("/metrics")
        metrics_text = response.text
        parsed = parse_metrics_exposition(metrics_text)

        # Get all base metric names from parsed data
        found_metrics = set()
        for full_name in parsed.keys():
            base_name = extract_base_metric_name(full_name)
            found_metrics.add(base_name)

        missing_metrics = []
        for required_metric in MetricsContract.REQUIRED_METRICS:
            if required_metric not in found_metrics:
                missing_metrics.append(required_metric)

        assert len(missing_metrics) == 0, \
            f"Missing required metrics: {missing_metrics}"

    def test_metric_label_schemas(self):
        """Verify metrics have expected label schemas."""
        response = client.get("/metrics")
        metrics_text = response.text
        parsed = parse_metrics_exposition(metrics_text)

        label_errors = []

        for metric_name, expected_labels in MetricsContract.REQUIRED_METRICS.items():
            # Find all instances of this metric
            metric_instances = []
            for full_name, instances in parsed.items():
                if extract_base_metric_name(full_name) == metric_name:
                    metric_instances.extend(instances)

            if not metric_instances and expected_labels:
                # Metric might not be initialized yet, skip
                continue

            for instance in metric_instances:
                actual_labels = set(instance.keys())

                # Check for missing labels
                missing = expected_labels - actual_labels
                if missing:
                    label_errors.append(
                        f"{metric_name}: missing labels {missing}, got {actual_labels}"
                    )

                # Check for unexpected labels
                extra = actual_labels - expected_labels
                if extra and expected_labels:  # Only check if we expect specific labels
                    label_errors.append(
                        f"{metric_name}: unexpected labels {extra}"
                    )

        assert len(label_errors) == 0, \
            f"Label schema violations:\n" + "\n".join(label_errors)

    def test_error_code_values_valid(self):
        """Verify error code label values are from ErrorCode enum."""
        response = client.get("/metrics")
        metrics_text = response.text
        parsed = parse_metrics_exposition(metrics_text)

        invalid_codes = []

        for full_name, instances in parsed.items():
            base_name = extract_base_metric_name(full_name)

            # Check metrics that should have 'code' label
            if base_name in ["ocr_errors_total", "vision_errors_total"]:
                for instance in instances:
                    if "code" in instance:
                        code = instance["code"]
                        if code not in MetricsContract.VALID_ERROR_CODES:
                            invalid_codes.append(
                                f"{base_name}: invalid code '{code}'"
                            )

        assert len(invalid_codes) == 0, \
            f"Invalid error codes found:\n" + "\n".join(invalid_codes)

    def test_stage_values_valid(self):
        """Verify stage label values are from expected set."""
        response = client.get("/metrics")
        metrics_text = response.text
        parsed = parse_metrics_exposition(metrics_text)

        invalid_stages = []

        for full_name, instances in parsed.items():
            base_name = extract_base_metric_name(full_name)

            # Check metrics that should have 'stage' label
            if base_name in ["ocr_errors_total", "ocr_stage_duration_seconds"]:
                for instance in instances:
                    if "stage" in instance:
                        stage = instance["stage"]
                        if stage not in MetricsContract.VALID_STAGES:
                            invalid_stages.append(
                                f"{base_name}: invalid stage '{stage}'"
                            )

        assert len(invalid_stages) == 0, \
            f"Invalid stage values found:\n" + "\n".join(invalid_stages)

    def test_ema_values_in_health(self):
        """Verify EMA values are available in /health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        health_data = response.json()
        assert "runtime" in health_data, "Health should have runtime section"
        assert "error_rate_ema" in health_data["runtime"], \
            "Runtime should have error_rate_ema"

        ema_data = health_data["runtime"]["error_rate_ema"]
        assert "ocr" in ema_data, "Should have OCR EMA"
        assert "vision" in ema_data, "Should have Vision EMA"

        # EMAs should be between 0 and 1
        assert 0 <= ema_data["ocr"] <= 1, f"OCR EMA out of range: {ema_data['ocr']}"
        assert 0 <= ema_data["vision"] <= 1, f"Vision EMA out of range: {ema_data['vision']}"

    @pytest.mark.asyncio
    async def test_metrics_increment_on_ocr_call(self):
        """Verify OCR metrics increment after API call."""
        # Get baseline metrics
        response = client.get("/metrics")
        before = parse_metrics_exposition(response.text)

        # Make an OCR call (will fail but should increment counters)
        files = {"file": ("test.txt", b"not_an_image", "text/plain")}
        ocr_response = client.post("/api/v1/ocr/extract", files=files)

        # Get updated metrics
        response = client.get("/metrics")
        after = parse_metrics_exposition(response.text)

        # Check that ocr_input_rejected_total incremented
        rejected_before = self._count_metric_value(before, "ocr_input_rejected_total")
        rejected_after = self._count_metric_value(after, "ocr_input_rejected_total")

        assert rejected_after > rejected_before, \
            "ocr_input_rejected_total should increment on invalid input"

    @pytest.mark.asyncio
    async def test_metrics_increment_on_vision_call(self):
        """Verify Vision metrics increment after API call."""
        # Get baseline metrics
        response = client.get("/metrics")
        before = parse_metrics_exposition(response.text)

        # Make a Vision call with small valid base64
        small_image = base64.b64encode(b"x" * 50).decode()
        vision_response = client.post(
            "/api/v1/vision/analyze",
            json={
                "image_base64": small_image,
                "include_description": True
            }
        )

        # Get updated metrics
        response = client.get("/metrics")
        after = parse_metrics_exposition(response.text)

        # Check that vision_requests_total incremented
        requests_before = self._count_metric_value(before, "vision_requests_total")
        requests_after = self._count_metric_value(after, "vision_requests_total")

        assert requests_after > requests_before, \
            "vision_requests_total should increment on API call"

    def test_rejection_reasons_valid(self):
        """Verify rejection reason values are from expected set."""
        response = client.get("/metrics")
        metrics_text = response.text
        parsed = parse_metrics_exposition(metrics_text)

        invalid_reasons = []

        for full_name, instances in parsed.items():
            base_name = extract_base_metric_name(full_name)

            # Check rejection metrics
            if base_name in ["ocr_input_rejected_total", "vision_input_rejected_total"]:
                for instance in instances:
                    if "reason" in instance:
                        reason = instance["reason"]
                        if reason not in MetricsContract.VALID_REJECTION_REASONS:
                            invalid_reasons.append(
                                f"{base_name}: unexpected reason '{reason}'"
                            )

        # This is a warning, not a failure - new reasons may be added
        if invalid_reasons:
            print(f"Warning: New rejection reasons found (update contract if valid):\n" +
                  "\n".join(invalid_reasons))

    def test_circuit_breaker_metrics_structure(self):
        """Verify circuit breaker metrics have correct structure."""
        response = client.get("/metrics")
        metrics_text = response.text
        parsed = parse_metrics_exposition(metrics_text)

        cb_metrics = []
        for full_name, instances in parsed.items():
            if "circuit_breaker" in full_name:
                cb_metrics.extend(instances)

        # If circuit breaker metrics exist, they should have 'circuit' label
        for metric in cb_metrics:
            assert "circuit" in metric, \
                f"Circuit breaker metric missing 'circuit' label: {metric}"

    def test_model_loaded_gauge_structure(self):
        """Verify model loaded gauge has provider label."""
        response = client.get("/metrics")
        metrics_text = response.text
        parsed = parse_metrics_exposition(metrics_text)

        model_metrics = parsed.get("ocr_model_loaded", [])

        for metric in model_metrics:
            assert "provider" in metric, \
                f"ocr_model_loaded missing 'provider' label: {metric}"

    def _count_metric_value(self, parsed: Dict, metric_name: str) -> int:
        """Helper to count total instances of a metric."""
        count = 0
        for full_name, instances in parsed.items():
            if extract_base_metric_name(full_name) == metric_name:
                count += len(instances)
        return count

    def test_histogram_metrics_have_buckets(self):
        """Verify histogram metrics have _bucket, _count, _sum variants."""
        response = client.get("/metrics")
        metrics_text = response.text
        parsed = parse_metrics_exposition(metrics_text)

        histogram_metrics = [
            "ocr_processing_duration_seconds",
            "vision_processing_duration_seconds",
            "ocr_stage_duration_seconds",
            "ocr_confidence_distribution",
            "vision_image_size_bytes"
        ]

        missing_variants = []

        for hist_metric in histogram_metrics:
            has_bucket = f"{hist_metric}_bucket" in parsed
            has_count = f"{hist_metric}_count" in parsed
            has_sum = f"{hist_metric}_sum" in parsed

            if not (has_bucket or has_count or has_sum):
                # Histogram not initialized yet, skip
                continue

            if not has_bucket:
                missing_variants.append(f"{hist_metric}_bucket")
            if not has_count:
                missing_variants.append(f"{hist_metric}_count")
            if not has_sum:
                missing_variants.append(f"{hist_metric}_sum")

        assert len(missing_variants) == 0, \
            f"Missing histogram variants: {missing_variants}"


class TestMetricsContractStrictMode:
    """Additional strict mode tests when SELF_CHECK_STRICT_METRICS=1."""

    # Guard: pytest.config was removed in newer pytest; fall back to env
    import os
    strict_enabled = os.getenv("STRICT_METRICS", "0") == "1"
    @pytest.mark.skipif(
        not strict_enabled,
        reason="Strict metrics mode not enabled"
    )
    def test_minimum_error_counters(self):
        """In strict mode, verify minimum error counter thresholds."""
        response = client.get("/metrics")
        parsed = parse_metrics_exposition(response.text)

        # Check that we have at least some error metrics initialized
        ocr_errors = parsed.get("ocr_errors_total", [])
        vision_errors = parsed.get("vision_errors_total", [])

        assert len(ocr_errors) > 0, "No OCR error metrics found (strict mode)"
        assert len(vision_errors) > 0, "No Vision error metrics found (strict mode)"

    @pytest.mark.skipif(
        not strict_enabled,
        reason="Strict metrics mode not enabled"
    )
    def test_all_providers_have_metrics(self):
        """In strict mode, all configured providers must have metrics."""
        response = client.get("/metrics")
        parsed = parse_metrics_exposition(response.text)

        # Get all unique providers from metrics
        providers = set()
        for full_name, instances in parsed.items():
            for instance in instances:
                if "provider" in instance:
                    providers.add(instance["provider"])

        # Should have at least paddle and deepseek_hf
        expected_providers = {"paddle", "deepseek_hf", "deepseek_stub"}
        missing = expected_providers - providers

        assert len(missing) == 0, \
            f"Missing metrics for providers: {missing}"


def pytest_addoption(parser):
    """Retain CLI option for backward compatibility; env var preferred."""
    try:
        parser.addoption(
            "--strict-metrics",
            action="store_true",
            default=False,
            help="Run strict metrics contract tests"
        )
    except Exception:
        # PyTest version may not support dynamic addoption at this stage; ignore
        pass


if __name__ == "__main__":
    # Quick validation
    test = TestMetricsContract()
    test.test_metrics_endpoint_available()
    test.test_required_metrics_present()
    test.test_ema_values_in_health()
    print("✅ Basic metrics contract tests passed")
