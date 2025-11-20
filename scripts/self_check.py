"""
Lightweight self-check script for local/CI verification.

Checks core health endpoint and (optionally) metrics exposure using FastAPI's TestClient.

Exit codes (aligned with CI failure routing):
 - 0: OK
 - 2: Critical failure (app import or /health contract)
 - 3: Metrics missing or malformed when expected
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

try:
    from fastapi.testclient import TestClient
except Exception as e:  # pragma: no cover - env issue
    print(f"[self-check] fastapi TestClient import failed: {e}")
    sys.exit(2)


def _load_app():
    try:
        from src.main import app  # type: ignore
    except Exception as e:
        print(f"[self-check] Failed to import app: {e}")
        sys.exit(2)
    return app


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        print(f"[self-check] {msg}")
        sys.exit(2)


def main() -> None:
    app = _load_app()
    client = TestClient(app)

    # 1) Health contract
    r = client.get("/health")
    _expect(r.status_code == 200, f"/health HTTP {r.status_code}")
    try:
        payload: Dict[str, Any] = r.json()
    except json.JSONDecodeError:
        print("[self-check] /health returned non-JSON body")
        sys.exit(2)

    _expect("status" in payload and payload["status"] == "healthy", "/health status!=healthy")
    _expect("runtime" in payload, "/health missing runtime")
    runtime = payload["runtime"]
    _expect("python_version" in runtime, "/health.runtime missing python_version")
    _expect("metrics_enabled" in runtime, "/health.runtime missing metrics_enabled")

    # 2) Metrics presence if enabled
    metrics_enabled = bool(runtime.get("metrics_enabled", False))
    check_metrics = os.getenv("SELF_CHECK_METRICS", "1") != "0"
    if metrics_enabled and check_metrics:
        mr = client.get("/metrics")
        if mr.status_code != 200:
            print(f"[self-check] /metrics HTTP {mr.status_code}")
            sys.exit(3)
        body = mr.text or ""
        # Check a few expected metric names exist (best-effort)
        required = [
            "ocr_error_rate_ema",
            "vision_error_rate_ema",
            "ocr_requests_total",
            "vision_requests_total",
        ]
        missing = [m for m in required if m not in body]
        if missing:
            print(f"[self-check] missing metrics: {', '.join(missing)}")
            sys.exit(3)

    # 3) Minimal unified error-path validation (optional)
    check_error = os.getenv("SELF_CHECK_ERROR", "1") != "0"
    if check_error:
        # Use invalid MIME to trigger unified INPUT_ERROR path deterministically
        files = {"file": ("bad.txt", b"hello", "text/plain")}
        er = client.post("/api/v1/ocr/extract", files=files)
        if er.status_code != 200:
            print(f"[self-check] error-path HTTP {er.status_code}")
            sys.exit(2)
        ej = er.json()
        if not isinstance(ej, dict) or ej.get("success", True) is True or not ej.get("code"):
            print("[self-check] unified error model missing or malformed")
            sys.exit(4)

    print("[self-check] OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
