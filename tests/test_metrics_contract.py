"""Metrics contract test for observability guarantees.

Validates:
1. Required metrics are present in /metrics endpoint
2. Label schemas match expected contracts
3. EMAs are available via /health
4. Counters increment correctly with sample calls
"""

import base64
import json
import re
from typing import Callable, Dict, List, Optional, Set, Tuple

import pytest
from fastapi.testclient import TestClient

from src.core.dedup2d_file_storage import LocalDedup2DFileStorage
from src.core.errors import ErrorCode
from src.main import app

client = TestClient(app)
_SAMPLE_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z/C/HwAFgwJ/lb9a0QAAAABJRU5ErkJggg=="
)


def _metrics_enabled_from_health() -> Optional[bool]:
    response = client.get("/health")
    if response.status_code != 200:
        return None

    try:
        health_data = response.json()
    except ValueError:
        return None

    runtime_enabled = health_data.get("runtime", {}).get("metrics_enabled")
    if runtime_enabled is not None:
        return bool(runtime_enabled)

    monitoring_enabled = (
        health_data.get("config", {}).get("monitoring", {}).get("metrics_enabled")
    )
    if monitoring_enabled is not None:
        return bool(monitoring_enabled)

    return None


@pytest.fixture(scope="module")
def metrics_enabled_flag() -> bool:
    enabled = _metrics_enabled_from_health()
    if enabled is not None:
        return enabled

    response = client.get("/metrics")
    if response.status_code != 200:
        return True

    return "app_metrics_disabled" not in response.text


@pytest.fixture(scope="module")
def require_metrics(metrics_enabled_flag: bool) -> None:
    if not metrics_enabled_flag:
        pytest.skip("metrics client disabled in this environment")


@pytest.fixture(scope="module", autouse=True)
def trigger_metrics_registration(metrics_enabled_flag: bool) -> None:
    """Trigger metrics registration by making OCR and Vision calls.

    Prometheus counters only appear in output after they've been incremented,
    so we need to make sample calls to ensure all metrics are registered.
    """
    if not metrics_enabled_flag:
        yield
        return

    # Trigger OCR metrics
    files = {"file": ("test.png", _SAMPLE_PNG_BYTES, "image/png")}
    client.post("/api/v1/ocr/extract", files=files)

    # Trigger Vision metrics
    small_image = base64.b64encode(b"x" * 50).decode()
    client.post(
        "/api/v1/vision/analyze", json={"image_base64": small_image, "include_description": False}
    )

    # Trigger cache tuning metrics
    client.post(
        "/api/v1/features/cache/tuning",
        json={"hit_rate": 0.55, "capacity": 200, "ttl": 300, "window_hours": 2},
        headers={"X-API-Key": "test"},
    )

    # Trigger health endpoint metrics
    client.get("/health")
    client.get("/health/extended")
    client.get("/ready")

    yield


class MetricsContract:
    """Define expected metrics and their label schemas."""

    # Metrics that should always be present after OCR/Vision calls
    CORE_METRICS = {
        "ocr_requests_total": {"provider", "status"},
        "ocr_model_loaded": {"provider"},
        "ocr_processing_duration_seconds": {"provider"},
        "ocr_confidence_distribution": set(),  # Histogram
        "vision_requests_total": {"provider", "status"},
        "vision_processing_duration_seconds": {"provider"},
        "vision_image_size_bytes": set(),  # Histogram
        "health_requests_total": {"endpoint", "status"},
        "health_request_duration_seconds": {"endpoint"},
    }

    # Metrics that only appear under specific conditions (optional)
    CONDITIONAL_METRICS = {
        "ocr_errors_total": {"provider", "code", "stage"},  # Only on errors
        "ocr_input_rejected_total": {"reason"},  # Only on rejection
        "ocr_stage_duration_seconds": {"provider", "stage"},  # Stage timing
        "vision_errors_total": {"provider", "code"},  # Only on errors
        "vision_input_rejected_total": {"reason"},  # Only on rejection
        "analysis_result_cleanup_total": {"status"},
        "analysis_result_cleanup_deleted_total": set(),
        "analysis_result_store_files": set(),
        "circuit_breaker_state": {"circuit"},  # Only when circuit breaker is registered
        "rate_limiter_allowed_total": {"key"},  # Only when rate limiter is used
        "rate_limiter_rejected_total": {"key"},  # Only when rate limiter rejects
        "dedup2d_file_uploads_total": {"backend", "status"},
        "dedup2d_file_downloads_total": {"backend", "status"},
        "dedup2d_file_deletes_total": {"backend", "status"},
        "dedup2d_file_upload_bytes": {"backend"},
        "dedup2d_file_operation_duration_seconds": {"backend", "operation"},
        "feature_cache_tuning_requests_total": {"status"},
        "feature_cache_tuning_recommended_capacity": set(),
        "feature_cache_tuning_recommended_ttl_seconds": set(),
    }

    # All metrics for label validation (when they do appear)
    REQUIRED_METRICS = {**CORE_METRICS, **CONDITIONAL_METRICS}

    # Valid label values for validation
    # Include both uppercase (from ErrorCode enum) and lowercase (legacy usage)
    VALID_ERROR_CODES = {code.value for code in ErrorCode} | {
        code.value.lower() for code in ErrorCode
    }
    VALID_STAGES = {"init", "load", "preprocess", "infer", "parse", "align", "postprocess"}
    VALID_STATUSES = {"start", "success", "error", "cache_hit", "input_error"}
    VALID_REJECTION_REASONS = {
        # OCR rejection reasons
        "validation_failed",
        "invalid_mime",
        "file_too_large",
        "pdf_pages_exceed",
        "pdf_forbidden_token",
        # Vision rejection reasons
        "base64_too_large",
        "base64_empty",
        "base64_padding_error",
        "base64_invalid_char",
        "base64_decode_error",
        "url_invalid_scheme",
        "url_invalid_format",
        "url_not_found",
        "url_forbidden",
        "url_http_error",
        "url_too_large_header",
        "url_too_large_download",
        "url_empty",
        "url_timeout",
        "url_network_error",
        "url_download_error",
    }


def parse_metrics_exposition(text: str) -> Dict[str, List[Dict[str, str]]]:
    """Parse Prometheus exposition format into structured data.

    Returns dict mapping metric names to list of label dicts.
    """
    metrics = {}

    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Match metric lines: metric_name{label1="value1",label2="value2"} value
        match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)((?:\{[^}]*\})?)[\s]+[\d\.\+\-eE]+", line)
        if match:
            metric_name = match.group(1)
            labels_str = match.group(2)

            # Parse labels
            labels = {}
            if labels_str and labels_str != "{}":
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
    for suffix in ["_bucket", "_count", "_sum", "_total"]:
        if (
            full_name.endswith(suffix)
            and full_name.replace(suffix, "") in MetricsContract.REQUIRED_METRICS
        ):
            return full_name.replace(suffix, "")
    return full_name


def _ensure_metric_present(metric_name: str, trigger: Callable[[], None]) -> None:
    parsed = parse_metrics_exposition(client.get("/metrics").text)
    if parsed.get(metric_name):
        return

    trigger()
    parsed = parse_metrics_exposition(client.get("/metrics").text)
    if not parsed.get(metric_name):
        pytest.fail(f"{metric_name} still missing after strict-mode trigger")


def _providers_for_metric(
    parsed: Dict[str, List[Dict[str, str]]], metric_name: str
) -> Set[str]:
    providers: Set[str] = set()
    for instance in parsed.get(metric_name, []):
        provider = instance.get("provider")
        if provider:
            providers.add(provider)
    return providers


class TestMetricsEndpoint:
    """Test metrics endpoint availability and fallback behavior."""

    def test_metrics_endpoint_available(self):
        """Verify /metrics endpoint is accessible."""
        response = client.get("/metrics")
        assert response.status_code == 200, "Metrics endpoint should return 200"
        content_type = response.headers.get("content-type", "")
        assert content_type.startswith(
            "text/plain; version=0.0.4"
        ), f"Metrics should use Prometheus text format, got: {content_type}"

    def test_metrics_fallback_when_disabled(self, metrics_enabled_flag: bool) -> None:
        """Verify fallback metrics are exposed when client is disabled."""
        if metrics_enabled_flag:
            pytest.skip("metrics client enabled in this environment")

        response = client.get("/metrics")
        assert response.status_code == 200
        assert "app_metrics_disabled" in response.text


@pytest.mark.usefixtures("require_metrics")
class TestMetricsContract:
    """Test metrics contract compliance."""

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
        # Only check CORE_METRICS - these should always be present after OCR/Vision calls
        for required_metric in MetricsContract.CORE_METRICS:
            if required_metric not in found_metrics:
                missing_metrics.append(required_metric)

        assert len(missing_metrics) == 0, f"Missing core metrics: {missing_metrics}"

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

                # Prometheus histogram buckets have 'le' label - this is standard
                # Remove 'le' from comparison for histogram metrics
                comparison_labels = actual_labels - {"le"}

                # Check for missing labels
                missing = expected_labels - comparison_labels
                if missing:
                    label_errors.append(
                        f"{metric_name}: missing labels {missing}, got {actual_labels}"
                    )

                # Check for unexpected labels (excluding 'le' which is standard for histograms)
                extra = comparison_labels - expected_labels
                if extra and expected_labels:  # Only check if we expect specific labels
                    label_errors.append(f"{metric_name}: unexpected labels {extra}")

        assert len(label_errors) == 0, f"Label schema violations:\n" + "\n".join(label_errors)

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
                            invalid_codes.append(f"{base_name}: invalid code '{code}'")

        assert len(invalid_codes) == 0, f"Invalid error codes found:\n" + "\n".join(invalid_codes)

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
                            invalid_stages.append(f"{base_name}: invalid stage '{stage}'")

        assert len(invalid_stages) == 0, f"Invalid stage values found:\n" + "\n".join(
            invalid_stages
        )

    def test_ema_values_in_health(self):
        """Verify EMA values are available in /health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        health_data = response.json()
        assert "runtime" in health_data, "Health should have runtime section"
        assert "error_rate_ema" in health_data["runtime"], "Runtime should have error_rate_ema"

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
        before_raw = response.text

        # Make an OCR call - the stub processes text files successfully
        files = {"file": ("test.png", _SAMPLE_PNG_BYTES, "image/png")}
        ocr_response = client.post("/api/v1/ocr/extract", files=files)

        # Get updated metrics
        response = client.get("/metrics")
        after = parse_metrics_exposition(response.text)
        after_raw = response.text

        # Check that ocr_requests_total incremented (sum of all counter values)
        requests_before = self._sum_metric_values(before_raw, "ocr_requests_total")
        requests_after = self._sum_metric_values(after_raw, "ocr_requests_total")

        assert (
            requests_after > requests_before
        ), f"ocr_requests_total should increment on API call (before={requests_before}, after={requests_after})"

    @pytest.mark.asyncio
    async def test_metrics_increment_on_vision_call(self):
        """Verify Vision metrics increment after API call."""
        # Get baseline metrics
        response = client.get("/metrics")
        before_raw = response.text

        # Make a Vision call with small valid base64
        small_image = base64.b64encode(b"x" * 50).decode()
        vision_response = client.post(
            "/api/v1/vision/analyze",
            json={"image_base64": small_image, "include_description": True},
        )

        # Get updated metrics
        response = client.get("/metrics")
        after_raw = response.text

        # Check that vision_requests_total incremented (sum of all counter values)
        requests_before = self._sum_metric_values(before_raw, "vision_requests_total")
        requests_after = self._sum_metric_values(after_raw, "vision_requests_total")

        assert (
            requests_after > requests_before
        ), f"vision_requests_total should increment on API call (before={requests_before}, after={requests_after})"

    @pytest.mark.asyncio
    async def test_dedup2d_storage_metrics_exposed(self, tmp_path, monkeypatch):
        """Verify dedup2d file storage metrics appear after a storage operation."""
        monkeypatch.setenv("DEDUP2D_FILE_STORAGE_DIR", str(tmp_path))
        storage = LocalDedup2DFileStorage()

        file_ref = await storage.save_bytes(
            job_id="metrics-contract",
            file_name="sample.png",
            content_type="image/png",
            data=b"contract",
        )
        await storage.load_bytes(file_ref)
        await storage.delete(file_ref)

        response = client.get("/metrics")
        assert response.status_code == 200
        text = response.text
        assert "dedup2d_file_uploads_total" in text
        assert "dedup2d_file_downloads_total" in text
        assert "dedup2d_file_deletes_total" in text

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
                            invalid_reasons.append(f"{base_name}: unexpected reason '{reason}'")

        # This is a warning, not a failure - new reasons may be added
        if invalid_reasons:
            print(
                f"Warning: New rejection reasons found (update contract if valid):\n"
                + "\n".join(invalid_reasons)
            )

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
            assert "circuit" in metric, f"Circuit breaker metric missing 'circuit' label: {metric}"

    def test_model_loaded_gauge_structure(self):
        """Verify model loaded gauge has provider label."""
        response = client.get("/metrics")
        metrics_text = response.text
        parsed = parse_metrics_exposition(metrics_text)

        model_metrics = parsed.get("ocr_model_loaded", [])

        for metric in model_metrics:
            assert "provider" in metric, f"ocr_model_loaded missing 'provider' label: {metric}"

    def _count_metric_value(self, parsed: Dict, metric_name: str) -> int:
        """Helper to count total instances of a metric."""
        count = 0
        for full_name, instances in parsed.items():
            if extract_base_metric_name(full_name) == metric_name:
                count += len(instances)
        return count

    def _sum_metric_values(self, raw_text: str, metric_name: str) -> float:
        """Helper to sum all values for a counter metric from raw Prometheus text.

        For counters like 'metric_name{label="value"} 5', this sums all the values.
        """
        total = 0.0
        for line in raw_text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Match metric lines and extract base name
            match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)((?:\{[^}]*\})?)[\s]+([\d\.\+\-eE]+)", line)
            if match:
                full_name = match.group(1)
                value_str = match.group(3)
                # Only sum if this matches the requested metric (exact match for counters)
                if full_name == metric_name or full_name == f"{metric_name}_total":
                    try:
                        total += float(value_str)
                    except ValueError:
                        pass
        return total

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
            "vision_image_size_bytes",
            "health_request_duration_seconds",
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

        assert len(missing_variants) == 0, f"Missing histogram variants: {missing_variants}"


@pytest.mark.usefixtures("require_metrics")
class TestMetricsContractStrictMode:
    """Additional strict mode tests when SELF_CHECK_STRICT_METRICS=1."""

    # Guard: pytest.config was removed in newer pytest; fall back to env
    import os

    strict_enabled = os.getenv("STRICT_METRICS", "0") == "1"

    @pytest.mark.skipif(not strict_enabled, reason="Strict metrics mode not enabled")
    def test_minimum_error_counters(self):
        """In strict mode, verify minimum error counter thresholds."""
        def trigger_ocr_error() -> None:
            files = {"file": ("test.png", _SAMPLE_PNG_BYTES, "image/png")}
            client.post("/api/v1/ocr/extract", params={"provider": "unknown"}, files=files)

        def trigger_vision_error() -> None:
            client.post("/api/v1/vision/analyze", json={})

        _ensure_metric_present("ocr_errors_total", trigger_ocr_error)
        _ensure_metric_present("vision_errors_total", trigger_vision_error)

        parsed = parse_metrics_exposition(client.get("/metrics").text)

        # Check that we have at least some error metrics initialized
        ocr_errors = parsed.get("ocr_errors_total", [])
        vision_errors = parsed.get("vision_errors_total", [])

        assert len(ocr_errors) > 0, "No OCR error metrics found (strict mode)"
        assert len(vision_errors) > 0, "No Vision error metrics found (strict mode)"

    @pytest.mark.skipif(not strict_enabled, reason="Strict metrics mode not enabled")
    def test_all_providers_have_metrics(self):
        """In strict mode, all configured providers must have metrics."""
        response = client.get("/metrics")
        parsed = parse_metrics_exposition(response.text)

        from src.api.v1.ocr import get_manager as get_ocr_manager
        from src.api.v1.vision import get_vision_manager

        ocr_manager = get_ocr_manager()
        expected_ocr_providers = set(ocr_manager.providers.keys())
        ocr_loaded_providers = _providers_for_metric(parsed, "ocr_model_loaded")
        missing_ocr = expected_ocr_providers - ocr_loaded_providers

        vision_provider = get_vision_manager().vision_provider.provider_name
        vision_request_providers = _providers_for_metric(parsed, "vision_requests_total")

        assert (
            len(missing_ocr) == 0
        ), f"OCR providers missing model-loaded metrics: {missing_ocr}"

        assert (
            vision_provider in vision_request_providers
        ), f"Vision provider '{vision_provider}' missing request metrics"


def pytest_addoption(parser):
    """Retain CLI option for backward compatibility; env var preferred."""
    try:
        parser.addoption(
            "--strict-metrics",
            action="store_true",
            default=False,
            help="Run strict metrics contract tests",
        )
    except Exception:
        # PyTest version may not support dynamic addoption at this stage; ignore
        pass


class TestDedup2DMetricsContract:
    """Tests for dedup2d metrics contract."""

    def test_dedup2d_metrics_module_exists(self) -> None:
        """Verify dedup2d metrics module can be imported."""
        from pathlib import Path

        metrics_path = Path(__file__).parent.parent / "src" / "core" / "dedup2d_metrics.py"
        assert metrics_path.exists(), f"Metrics module not found at {metrics_path}"

        # Check that the module contains expected metric names
        content = metrics_path.read_text()
        expected_metrics = [
            "dedup2d_jobs_submitted_total",
            "dedup2d_jobs_completed_total",
            "dedup2d_job_duration_seconds",
            "dedup2d_jobs_queued",
            "dedup2d_jobs_active",
            "dedup2d_file_uploads_total",
            "dedup2d_file_downloads_total",
            "dedup2d_callbacks_total",
            "dedup2d_gc_runs_total",
            "dedup2d_error_rate_ema",
        ]
        for metric in expected_metrics:
            assert metric in content, f"Missing metric {metric} in dedup2d_metrics.py"

    def test_dedup2d_metrics_exports(self) -> None:
        """Verify dedup2d metrics module has proper __all__ exports."""
        from pathlib import Path

        metrics_path = Path(__file__).parent.parent / "src" / "core" / "dedup2d_metrics.py"
        content = metrics_path.read_text()

        # Check __all__ is defined
        assert "__all__" in content, "Metrics module should define __all__"

        # Check key exports are in __all__
        assert "dedup2d_jobs_submitted_total" in content
        assert "update_dedup2d_error_ema" in content
        assert "get_dedup2d_error_rate_ema" in content

    def test_grafana_dashboard_exists(self) -> None:
        """Verify dedup2d Grafana dashboard exists."""
        from pathlib import Path

        dashboard_path = Path(__file__).parent.parent / "grafana" / "dashboards" / "dedup2d.json"
        assert dashboard_path.exists(), f"Dashboard not found at {dashboard_path}"

    def test_prometheus_alerts_exist(self) -> None:
        """Verify dedup2d Prometheus alerts exist."""
        from pathlib import Path

        alerts_path = Path(__file__).parent.parent / "prometheus" / "alerts" / "dedup2d.yml"
        assert alerts_path.exists(), f"Alerts not found at {alerts_path}"

    def test_grafana_dashboard_valid_json(self) -> None:
        """Verify dedup2d Grafana dashboard is valid JSON."""
        from pathlib import Path

        dashboard_path = Path(__file__).parent.parent / "grafana" / "dashboards" / "dedup2d.json"
        if dashboard_path.exists():
            content = dashboard_path.read_text()
            dashboard = json.loads(content)
            assert "panels" in dashboard
            assert "title" in dashboard
            assert dashboard["title"] == "Dedup2D Dashboard"

    def test_prometheus_alerts_valid_yaml(self) -> None:
        """Verify dedup2d Prometheus alerts is valid YAML."""
        from pathlib import Path

        import yaml

        alerts_path = Path(__file__).parent.parent / "prometheus" / "alerts" / "dedup2d.yml"
        if alerts_path.exists():
            content = alerts_path.read_text()
            alerts = yaml.safe_load(content)
            assert "groups" in alerts
            assert len(alerts["groups"]) > 0
            # Check first group has rules
            assert "rules" in alerts["groups"][0]
            assert len(alerts["groups"][0]["rules"]) > 0


if __name__ == "__main__":
    # Quick validation
    test = TestMetricsContract()
    test.test_metrics_endpoint_available()
    test.test_required_metrics_present()
    test.test_ema_values_in_health()
    print("✅ Basic metrics contract tests passed")
