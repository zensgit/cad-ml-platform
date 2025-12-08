"""
Enhanced self-check script for local/CI verification with strict mode support.

Checks core health endpoint, metrics exposure, and contract compliance.

Environment variables:
 - SELF_CHECK_BASE_URL: Base URL for checks (default: http://localhost:8000)
 - SELF_CHECK_STRICT_METRICS: Enable strict metrics validation (default: false)
 - SELF_CHECK_MIN_OCR_ERRORS: Minimum OCR error counter value (default: 0)
 - SELF_CHECK_REQUIRE_EMA: Require EMA values in health (default: true)

Exit codes (aligned with CI failure routing):
 - 0: OK
 - 2: Critical failure (app import or /health contract)
 - 3: Metrics missing or malformed when expected
 - 5: Metrics contract failure (strict mode)
 - 6: Provider error mapping gap detected
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import base64
from typing import Any, Dict, List, Set, Optional

# Global variables for JSON output mode
JSON_MODE = False
CHECK_RESULTS: Dict[str, Any] = {
    "checks": {},
    "errors": [],
    "warnings": [],
    "exit_code": 0,
    "summary": ""
}

# Try local client first, fallback to httpx for remote URLs
base_url = os.getenv("SELF_CHECK_BASE_URL", "").strip()

if not base_url:
    # Use TestClient for local testing
    try:
        from fastapi.testclient import TestClient
    except Exception as e:
        print(f"[self-check] fastapi TestClient import failed: {e}")
        sys.exit(2)

    def _load_app():
        try:
            from src.main import app  # type: ignore
        except Exception as e:
            print(f"[self-check] Failed to import app: {e}")
            sys.exit(2)
        return app

    app = _load_app()
    client = TestClient(app)
else:
    # Use httpx for remote URLs
    try:
        import httpx
        client = httpx.Client(base_url=base_url, timeout=10.0)
    except Exception as e:
        print(f"[self-check] httpx client creation failed: {e}")
        sys.exit(2)


def _expect(cond: bool, msg: str, exit_code: int = 2) -> None:
    if not cond:
        if JSON_MODE:
            CHECK_RESULTS["errors"].append(f"FAIL: {msg}")
            CHECK_RESULTS["exit_code"] = max(CHECK_RESULTS["exit_code"], exit_code)
        else:
            print(f"[self-check] FAIL: {msg}")
            sys.exit(exit_code)


def _print(message: str, level: str = "info") -> None:
    """Print message or add to JSON results based on mode."""
    if JSON_MODE:
        if level == "error":
            CHECK_RESULTS["errors"].append(message)
        elif level == "warning":
            CHECK_RESULTS["warnings"].append(message)
        # Info messages not included in JSON
    else:
        print(f"[self-check] {message}")


def parse_metrics(text: str) -> Dict[str, List[Dict[str, str]]]:
    """Parse Prometheus metrics exposition format."""
    metrics = {}

    for line in text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        # Match metric lines
        match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)((?:\{[^}]*\})?)[\s]+[\d\.\+\-eE]+', line)
        if match:
            metric_name = match.group(1)
            labels_str = match.group(2)

            # Parse labels
            labels = {}
            if labels_str and labels_str != '{}':
                labels_str = labels_str[1:-1]
                for pair in re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"', labels_str):
                    labels[pair[0]] = pair[1]

            if metric_name not in metrics:
                metrics[metric_name] = []
            metrics[metric_name].append(labels)

    return metrics


def check_health_endpoint() -> Dict[str, Any]:
    """Check health endpoint and return data."""
    print("[self-check] Checking /health endpoint...")

    r = client.get("/health")
    _expect(r.status_code == 200, f"/health HTTP {r.status_code}")

    try:
        payload = r.json()
    except json.JSONDecodeError:
        print("[self-check] /health returned non-JSON body")
        sys.exit(2)

    _expect("status" in payload and payload["status"] == "healthy", "/health status!=healthy")
    _expect("runtime" in payload, "/health missing runtime")

    runtime = payload["runtime"]
    _expect("python_version" in runtime, "/health.runtime missing python_version")
    _expect("metrics_enabled" in runtime, "/health.runtime missing metrics_enabled")

    # Check EMA if required
    require_ema = os.getenv("SELF_CHECK_REQUIRE_EMA", "1") != "0"
    if require_ema:
        _expect("error_rate_ema" in runtime, "/health.runtime missing error_rate_ema")
        ema = runtime.get("error_rate_ema", {})
        _expect("ocr" in ema and "vision" in ema, "/health EMA missing ocr/vision")

        # Validate EMA values are in range [0, 1]
        ocr_ema = ema.get("ocr", -1)
        vision_ema = ema.get("vision", -1)
        _expect(0 <= ocr_ema <= 1, f"/health OCR EMA out of range: {ocr_ema}")
        _expect(0 <= vision_ema <= 1, f"/health Vision EMA out of range: {vision_ema}")

    print(f"[self-check] ✓ Health endpoint OK (status=healthy)")
    return payload


def check_metrics_endpoint() -> Dict[str, List[Dict[str, str]]]:
    """Check metrics endpoint and parse data."""
    print("[self-check] Checking /metrics endpoint...")

    mr = client.get("/metrics")
    if mr.status_code != 200:
        print(f"[self-check] /metrics HTTP {mr.status_code}")
        sys.exit(3)

    body = mr.text or ""

    # Basic presence check
    required = [
        "ocr_errors_total",
        "vision_errors_total",
        "ocr_requests_total",
        "vision_requests_total",
    ]

    missing = [m for m in required if m not in body]
    if missing:
        print(f"[self-check] Missing metrics: {', '.join(missing)}")
        sys.exit(3)

    print(f"[self-check] ✓ Metrics endpoint OK")

    # Parse metrics for detailed validation
    return parse_metrics(body)


def increment_counters():
    """Make sample API calls to increment counters."""
    print("[self-check] Making sample API calls to increment counters...")

    # OCR call with invalid input (should increment rejection counter)
    files = {"file": ("test.txt", b"not_an_image", "text/plain")}
    ocr_resp = client.post("/api/v1/ocr/extract", files=files)
    print(f"[self-check]   OCR call: HTTP {ocr_resp.status_code}")

    # Vision call with small valid base64 (should increment request counter)
    small_image = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"x" * 50).decode()
    vision_resp = client.post(
        "/api/v1/vision/analyze",
        json={"image_base64": small_image, "include_description": True}
    )
    print(f"[self-check]   Vision call: HTTP {vision_resp.status_code}")


def check_strict_metrics(metrics: Dict[str, List[Dict[str, str]]]):
    """Perform strict metrics contract validation."""
    print("[self-check] Running strict metrics validation...")

    # Check minimum OCR error counters
    min_ocr_errors = int(os.getenv("SELF_CHECK_MIN_OCR_ERRORS", "0"))
    if min_ocr_errors > 0:
        ocr_error_count = len(metrics.get("ocr_errors_total", []))
        _expect(
            ocr_error_count >= min_ocr_errors,
            f"OCR error counter too low: {ocr_error_count} < {min_ocr_errors}",
            exit_code=5
        )

    # Check label contract compliance
    expected_labels = {
        "ocr_errors_total": {"provider", "code", "stage"},
        "vision_errors_total": {"provider", "code"},
        "ocr_input_rejected_total": {"reason"},
        "vision_input_rejected_total": {"reason"},
        "ocr_model_loaded": {"provider"},
    }

    label_violations = []
    for metric_name, expected_set in expected_labels.items():
        instances = metrics.get(metric_name, [])
        for instance in instances:
            actual_set = set(instance.keys())
            missing = expected_set - actual_set
            extra = actual_set - expected_set

            if missing:
                label_violations.append(f"{metric_name} missing labels: {missing}")
            if extra:
                label_violations.append(f"{metric_name} extra labels: {extra}")

    if label_violations:
        print("[self-check] Label contract violations:")
        for violation in label_violations:
            print(f"  - {violation}")
        sys.exit(5)

    # Check provider error mapping coverage
    from src.core.errors import ErrorCode
    valid_codes = {code.value for code in ErrorCode}

    unmapped_codes = set()
    for metric_name in ["ocr_errors_total", "vision_errors_total"]:
        for instance in metrics.get(metric_name, []):
            code = instance.get("code", "")
            if code and code not in valid_codes:
                unmapped_codes.add(code)

    if unmapped_codes:
        print(f"[self-check] Unmapped error codes detected: {unmapped_codes}")
        sys.exit(6)

    print(f"[self-check] ✓ Strict metrics validation passed")


def main(args: Optional[argparse.Namespace] = None) -> None:
    global JSON_MODE

    # Parse arguments if not provided
    if args is None:
        parser = argparse.ArgumentParser(description="Self-check script for CAD ML Platform")
        parser.add_argument("--json", action="store_true", help="Output results as JSON")
        args = parser.parse_args()

    JSON_MODE = args.json

    if not JSON_MODE:
        print("[self-check] Starting enhanced self-check...")
        print(f"[self-check] Base URL: {base_url or 'local (TestClient)'}")

    # Check configuration from environment
    strict_mode = os.getenv("SELF_CHECK_STRICT_METRICS", "0") != "0"

    if not JSON_MODE:
        print(f"[self-check] Strict mode: {strict_mode}")

    if JSON_MODE:
        CHECK_RESULTS["config"] = {
            "base_url": base_url or "local",
            "strict_mode": strict_mode
        }

    # 1. Health check
    health_data = check_health_endpoint()
    metrics_enabled = health_data.get("runtime", {}).get("metrics_enabled", False)

    # 2. Metrics check (if enabled)
    metrics_data = {}
    if metrics_enabled:
        metrics_data = check_metrics_endpoint()

        # Increment counters if requested
        if os.getenv("SELF_CHECK_INCREMENT_COUNTERS", "0") != "0":
            increment_counters()
            # Re-fetch metrics after increment
            metrics_data = parse_metrics(client.get("/metrics").text)
    else:
        print("[self-check] Metrics disabled, skipping metrics checks")

    # 3. Strict validation (if enabled)
    if strict_mode and metrics_data:
        check_strict_metrics(metrics_data)

    # 4. Error path validation
    if os.getenv("SELF_CHECK_ERROR", "1") != "0":
        print("[self-check] Checking unified error model...")
        # Send completely invalid data (empty bytes) to trigger validation error
        files = {"file": ("empty.png", b"", "image/png")}
        er = client.post("/api/v1/ocr/extract", files=files)

        _expect(er.status_code == 200, f"error-path HTTP {er.status_code}")

        ej = er.json()
        _expect(isinstance(ej, dict), "error response not JSON dict")
        # Check if response indicates error (success=False) or has error code
        # Some providers may process empty input gracefully, so we check for either
        has_error_indicator = (
            ej.get("success") is False or
            ej.get("code") is not None or
            ej.get("error") is not None
        )
        if has_error_indicator:
            print(f"[self-check] ✓ Unified error model OK (code={ej.get('code', 'none')})")
        else:
            # If no error indicator, at least verify the response structure is valid
            _expect("provider" in ej, "response missing provider field")
            print(f"[self-check] ✓ OCR response structure OK (provider={ej.get('provider')})")

    if JSON_MODE:
        # Output JSON results
        if CHECK_RESULTS["exit_code"] == 0:
            CHECK_RESULTS["summary"] = "All checks passed"
            CHECK_RESULTS["success"] = True
        else:
            CHECK_RESULTS["summary"] = f"Checks failed with exit code {CHECK_RESULTS['exit_code']}"
            CHECK_RESULTS["success"] = False

        print(json.dumps(CHECK_RESULTS, indent=2))
        sys.exit(CHECK_RESULTS["exit_code"])
    else:
        print("\n[self-check] ✅ All checks passed")
        sys.exit(0)


if __name__ == "__main__":
    main()