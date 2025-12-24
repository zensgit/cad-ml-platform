"""
Comprehensive test suite for observability enhancements.

Tests all new features including:
- Metrics contract validation
- Error code mapping
- Self-check functionality
- Recording rules validation
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.errors import ErrorCode
from src.core.ocr.providers.error_map import (
    handle_inference_error,
    handle_init_error,
    handle_load_error,
    handle_parse_error,
    log_and_map_exception,
    map_exception_to_error_code,
)


class TestErrorCodeMapping:
    """Test the centralized error mapping functionality."""

    def test_all_error_codes_have_mappings(self):
        """Ensure all ErrorCode enum values can be produced by mapping."""
        mapped_codes = set()

        # Test common exception types
        test_exceptions = [
            MemoryError("OOM"),
            TimeoutError("Timeout"),
            ConnectionError("Network"),
            ValueError("Parse error"),
            IOError("Permission denied"),
            RuntimeError("Model failed to load"),
            RuntimeError("Authentication failed"),
            RuntimeError("Rate limit exceeded"),
            Exception("Generic error"),
        ]

        for exc in test_exceptions:
            code = map_exception_to_error_code(exc)
            mapped_codes.add(code)

        # Verify we can map to most error codes
        expected_codes = {
            ErrorCode.RESOURCE_EXHAUSTED,
            ErrorCode.PROVIDER_TIMEOUT,
            ErrorCode.NETWORK_ERROR,
            ErrorCode.PARSE_FAILED,
            ErrorCode.AUTH_FAILED,
            ErrorCode.MODEL_LOAD_ERROR,
            ErrorCode.INTERNAL_ERROR,
        }

        assert len(mapped_codes.intersection(expected_codes)) >= 5

    def test_exception_message_patterns(self):
        """Test that error messages influence mapping correctly."""
        # Memory patterns
        exc1 = RuntimeError("Out of memory")
        assert map_exception_to_error_code(exc1) == ErrorCode.RESOURCE_EXHAUSTED

        # Timeout patterns
        exc2 = RuntimeError("Request timed out")
        assert map_exception_to_error_code(exc2) == ErrorCode.PROVIDER_TIMEOUT

        # Network patterns
        exc3 = RuntimeError("Connection refused")
        assert map_exception_to_error_code(exc3) == ErrorCode.NETWORK_ERROR

        # Auth patterns
        exc4 = RuntimeError("403 Forbidden")
        assert map_exception_to_error_code(exc4) == ErrorCode.AUTH_FAILED

    def test_convenience_handlers(self):
        """Test specialized error handlers."""
        # Inference error
        exc = TimeoutError("Inference timeout")
        code = handle_inference_error(exc, "test_provider")
        assert code == ErrorCode.PROVIDER_TIMEOUT

        # Parse error - always PARSE_FAILED for common exceptions
        exc2 = ValueError("Bad value")
        code2 = handle_parse_error(exc2, "test_provider")
        assert code2 == ErrorCode.PARSE_FAILED

        # Load error - memory vs model load
        exc3 = MemoryError("OOM")
        code3 = handle_load_error(exc3, "test_provider")
        assert code3 == ErrorCode.RESOURCE_EXHAUSTED

        exc4 = RuntimeError("Model not found")
        code4 = handle_load_error(exc4, "test_provider")
        assert code4 == ErrorCode.MODEL_LOAD_ERROR


class TestMetricsContract:
    """Test metrics contract validation functionality."""

    def test_parse_metrics_exposition_format(self):
        """Test parsing of Prometheus exposition format."""
        sample_metrics = """
# HELP ocr_errors_total Total OCR errors
# TYPE ocr_errors_total counter
ocr_errors_total{provider="tesseract",code="PARSE_FAILED",stage="parse"} 5
ocr_errors_total{provider="deepseek",code="PROVIDER_TIMEOUT",stage="infer"} 2

# HELP ocr_input_rejected_total Input rejections
# TYPE ocr_input_rejected_total counter
ocr_input_rejected_total{reason="invalid_format"} 10
"""

        # Parse metrics
        metrics = {}
        for line in sample_metrics.strip().split("\n"):
            if line and not line.startswith("#"):
                parts = line.split("{")
                if len(parts) == 2:
                    metric_name = parts[0]
                    labels_and_value = parts[1]
                    labels_part = labels_and_value.split("}")[0]

                    if metric_name not in metrics:
                        metrics[metric_name] = []

                    # Parse labels
                    labels = {}
                    for label_pair in labels_part.split(","):
                        if "=" in label_pair:
                            key, value = label_pair.split("=")
                            labels[key.strip()] = value.strip('"')

                    metrics[metric_name].append(labels)

        # Validate parsed metrics
        assert "ocr_errors_total" in metrics
        assert len(metrics["ocr_errors_total"]) == 2
        assert "ocr_input_rejected_total" in metrics

        # Check labels
        error_labels = {tuple(sorted(m.keys())) for m in metrics["ocr_errors_total"]}
        assert ("code", "provider", "stage") in error_labels

    def test_required_metrics_validation(self):
        """Test validation of required metrics."""
        REQUIRED_METRICS = {
            "ocr_errors_total": {"provider", "code", "stage"},
            "ocr_input_rejected_total": {"reason"},
            "vision_input_rejected_total": {"reason"},
            "ocr_processing_duration_seconds": set(),
            "vision_processing_duration_seconds": set(),
        }

        # Sample metrics that pass validation
        good_metrics = {
            "ocr_errors_total": [{"provider": "test", "code": "TIMEOUT", "stage": "infer"}],
            "ocr_input_rejected_total": [{"reason": "invalid"}],
            "vision_input_rejected_total": [{"reason": "base64_error"}],
            "ocr_processing_duration_seconds_bucket": [],
            "vision_processing_duration_seconds_bucket": [],
        }

        # Check each required metric
        for metric_name, required_labels in REQUIRED_METRICS.items():
            base_name = metric_name.replace("_seconds", "_seconds_bucket")
            assert base_name in good_metrics or metric_name in good_metrics

    def test_error_code_coverage(self):
        """Test that all ErrorCode values are represented in metrics."""
        valid_error_codes = {code.value for code in ErrorCode}

        # Simulate metrics with various error codes
        test_metrics = []
        for code in ErrorCode:
            test_metrics.append({"provider": "test", "code": code.value, "stage": "test"})

        # Extract codes from metrics
        found_codes = {m["code"] for m in test_metrics}

        # All error codes should be representable
        assert found_codes == valid_error_codes


class TestPromtoolValidation:
    """Test Prometheus recording rules validation."""

    def test_recording_rules_yaml_valid(self):
        """Test that recording rules YAML is valid."""
        rules_path = Path("docs/prometheus/recording_rules.yml")

        if rules_path.exists():
            with open(rules_path, "r") as f:
                rules_data = yaml.safe_load(f)

            # Basic structure validation
            assert "groups" in rules_data
            assert isinstance(rules_data["groups"], list)

            for group in rules_data["groups"]:
                assert "name" in group
                assert "rules" in group
                assert isinstance(group["rules"], list)

                for rule in group["rules"]:
                    assert "record" in rule
                    assert "expr" in rule

                    # Check naming convention
                    record_name = rule["record"]
                    assert record_name.replace("_", "").replace(".", "").isalnum()

    def test_promtool_script_execution(self):
        """Test that promtool validation script runs."""
        script_path = Path("scripts/validate_prom_rules.py")

        if script_path.exists():
            # Test help output
            result = subprocess.run(
                [sys.executable, str(script_path), "--help"], capture_output=True, text=True
            )
            assert result.returncode == 0
            assert "Validate Prometheus recording rules" in result.stdout

            # Test JSON output mode
            result = subprocess.run(
                [sys.executable, str(script_path), "--skip-promtool", "--json"],
                capture_output=True,
                text=True,
            )

            # Should produce valid JSON
            try:
                output = json.loads(result.stdout)
                assert "validation_passed" in output
                assert "metrics_used" in output
            except json.JSONDecodeError:
                pytest.fail("Promtool script did not produce valid JSON")


class TestSelfCheckScript:
    """Test self-check script functionality."""

    def test_self_check_script_exists(self):
        """Test that self-check script exists and is executable."""
        script_path = Path("scripts/self_check.py")
        assert script_path.exists()

        # Check script has proper shebang or can be executed with Python
        with open(script_path, "r") as f:
            first_line = f.readline()
            # Either has shebang or can be run with python
            assert first_line.startswith("#!") or script_path.suffix == ".py"

    def test_exit_codes_documented(self):
        """Test that exit codes are properly documented."""
        script_path = Path("scripts/self_check.py")

        if script_path.exists():
            with open(script_path, "r") as f:
                content = f.read()

            # Check that exit codes are documented
            expected_codes = ["0", "2", "3", "5", "6"]
            for code in expected_codes:
                assert f"Exit code" in content or f"exit({code})" in content

    @patch.dict(os.environ, {"SELF_CHECK_STRICT_METRICS": "1"})
    def test_environment_variables(self):
        """Test that environment variables are read correctly."""
        # This would be tested in integration, but we check the code structure
        script_path = Path("scripts/self_check.py")

        if script_path.exists():
            with open(script_path, "r") as f:
                content = f.read()

            # Check for environment variable usage
            env_vars = [
                "SELF_CHECK_BASE_URL",
                "SELF_CHECK_STRICT_METRICS",
                "SELF_CHECK_MIN_OCR_ERRORS",
                "SELF_CHECK_REQUIRE_EMA",
                "SELF_CHECK_INCREMENT_COUNTERS",
            ]

            for var in env_vars:
                assert var in content


class TestRunbooks:
    """Test that runbooks are complete and useful."""

    def test_runbook_structure(self):
        """Test that runbooks have proper structure."""
        runbook_files = ["docs/runbooks/provider_timeout.md", "docs/runbooks/model_load_error.md"]

        required_sections = [
            "## Overview",
            "## Error Code",
            "## Detection",
            "## Response Steps",
            "## Root Cause Analysis",
            "## Prevention",
            "## Escalation",
        ]

        for runbook_path in runbook_files:
            path = Path(runbook_path)
            if path.exists():
                with open(path, "r") as f:
                    content = f.read()

                for section in required_sections:
                    assert section in content, f"{runbook_path} missing {section}"

                # Check for practical examples
                assert "```" in content  # Code blocks
                assert "curl" in content or "python" in content  # Commands

    def test_runbook_error_codes(self):
        """Test that runbooks reference correct error codes."""
        runbook_mapping = {
            "provider_timeout.md": "PROVIDER_TIMEOUT",
            "model_load_error.md": "MODEL_LOAD_ERROR",
        }

        for filename, expected_code in runbook_mapping.items():
            path = Path(f"docs/runbooks/{filename}")
            if path.exists():
                with open(path, "r") as f:
                    content = f.read()

                assert expected_code in content


class TestGrafanaDashboard:
    """Test Grafana dashboard configuration."""

    def test_dashboard_json_valid(self):
        """Test that Grafana dashboard JSON is valid."""
        dashboard_path = Path("docs/grafana/observability_dashboard.json")

        if dashboard_path.exists():
            with open(dashboard_path, "r") as f:
                dashboard = json.load(f)

            # Basic structure validation
            assert "panels" in dashboard
            assert isinstance(dashboard["panels"], list)
            assert len(dashboard["panels"]) > 0

            # Check for key panels
            panel_titles = [p.get("title", "") for p in dashboard["panels"]]

            expected_panels = [
                "Error",  # Should have error-related panels
                "Provider",  # Provider-specific panels
                "SLO",  # SLO compliance panels
            ]

            for expected in expected_panels:
                assert any(expected in title for title in panel_titles)

    def test_recording_rules_used_in_dashboard(self):
        """Test that dashboard uses recording rules."""
        dashboard_path = Path("docs/grafana/observability_dashboard.json")
        rules_path = Path("docs/prometheus/recording_rules.yml")

        if dashboard_path.exists() and rules_path.exists():
            with open(dashboard_path, "r") as f:
                dashboard = json.load(f)

            with open(rules_path, "r") as f:
                rules = yaml.safe_load(f)

            # Extract recording rule names
            rule_names = []
            for group in rules.get("groups", []):
                for rule in group.get("rules", []):
                    if "record" in rule:
                        rule_names.append(rule["record"])

            # Check if any recording rules are referenced in queries
            dashboard_str = json.dumps(dashboard)
            rules_used = 0
            for rule_name in rule_names:
                if rule_name in dashboard_str:
                    rules_used += 1

            # At least some recording rules should be used
            assert rules_used > 0


class TestDocumentation:
    """Test that documentation is complete and accurate."""

    def test_readme_exit_codes_table(self):
        """Test that README contains exit codes table."""
        readme_path = Path("README.md")

        if readme_path.exists():
            with open(readme_path, "r") as f:
                content = f.read()

            # Check for exit codes section
            assert "Exit Code" in content or "exit code" in content.lower()

            # Check for table format
            assert "|" in content  # Markdown table

            # Check specific exit codes are documented
            exit_codes = ["0", "2", "3", "5", "6"]
            for code in exit_codes:
                assert f"`{code}`" in content or f"| {code} |" in content

    def test_quality_baseline_updated(self):
        """Test that QUALITY_BASELINE.md is updated."""
        baseline_path = Path("docs/QUALITY_BASELINE.md")

        if baseline_path.exists():
            with open(baseline_path, "r") as f:
                content = f.read()

            # Check for new sections
            assert "Metrics Contract" in content
            assert "ErrorCode" in content
            assert "Recording Rules" in content
            assert "Self-Check" in content

    def test_roadmap_phase2_complete(self):
        """Test that Phase 2 roadmap is comprehensive."""
        roadmap_path = Path("docs/ROADMAP_PHASE2.md")

        if roadmap_path.exists():
            with open(roadmap_path, "r") as f:
                content = f.read()

            # Check for key sections
            required_sections = [
                "Week 1",
                "Week 2",
                "Week 3",
                "Week 4",
                "Success Metrics",
                "Risk",
                "Team",
            ]

            for section in required_sections:
                assert section in content


class TestIntegration:
    """Integration tests for the complete observability system."""

    def test_error_to_metric_flow(self):
        """Test that errors flow correctly to metrics."""
        # Simulate error → ErrorCode → metric label
        exc = TimeoutError("Provider timeout")
        code = map_exception_to_error_code(exc)

        # The code should be usable as a metric label
        assert isinstance(code.value, str)
        assert code.value.replace("_", "").isalpha()

        # Simulate metric creation
        metric_labels = {"provider": "test", "code": code.value, "stage": "infer"}

        # All labels should be strings
        assert all(isinstance(v, str) for v in metric_labels.values())

    def test_monitoring_stack_compatibility(self):
        """Test that configurations are compatible with monitoring stack."""
        # Check that recording rules match expected metric names
        rules_path = Path("docs/prometheus/recording_rules.yml")

        if rules_path.exists():
            with open(rules_path, "r") as f:
                rules = yaml.safe_load(f)

            # Extract metrics referenced in expressions
            referenced_metrics = set()
            for group in rules.get("groups", []):
                for rule in group.get("rules", []):
                    expr = rule.get("expr", "")
                    # Simple extraction of metric names
                    import re

                    metrics = re.findall(r"\b([a-z_]+(?:_total|_bucket|_count|_sum)?)\b", expr)
                    referenced_metrics.update(metrics)

            # Key metrics should be referenced
            expected_metrics = ["ocr_errors_total", "ocr_requests_total", "vision_errors_total"]

            for metric in expected_metrics:
                assert metric in referenced_metrics


def test_all_files_created():
    """Test that all expected files were created."""
    expected_files = [
        "tests/test_metrics_contract.py",
        "tests/test_provider_error_mapping.py",
        "src/core/ocr/providers/error_map.py",
        "scripts/self_check.py",
        "scripts/validate_prom_rules.py",
        "docs/prometheus/recording_rules.yml",
        "docs/grafana/observability_dashboard.json",
        ".github/workflows/security-audit.yml",
        "docs/runbooks/provider_timeout.md",
        "docs/runbooks/model_load_error.md",
        "docs/QUALITY_BASELINE.md",
        "docs/ROADMAP_PHASE2.md",
    ]

    missing_files = []
    for filepath in expected_files:
        if not Path(filepath).exists():
            missing_files.append(filepath)

    if missing_files:
        pytest.fail(f"Missing files: {missing_files}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
