"""Self-check script for CI smoke validation.

Performs:
1. /health endpoint check
2. /metrics presence of key counters
3. Vision analyze minimal request (base64 tiny PNG)
4. OCR extract invalid MIME to ensure INPUT_ERROR path

Exit codes:
0 - All checks pass
2 - Critical failure (unhealthy health or request crash)
3 - Missing metrics counters
4 - Unexpected error code schema
"""

from __future__ import annotations

import base64
import sys
import json
from typing import Any

import requests

BASE = "http://localhost:8000"


def _fail(code: int, msg: str) -> None:
    print(json.dumps({"success": False, "error": msg, "code": code}))
    sys.exit(code)


def main() -> None:
    # 1. Health
    try:
        r = requests.get(f"{BASE}/health", timeout=3)
    except Exception as e:
        _fail(2, f"health request failed: {e}")
    if r.status_code != 200:
        _fail(2, f"health non-200: {r.status_code}")
    data = r.json()
    if data.get("status") not in ("healthy", "ok"):
        _fail(2, "health status not healthy")

    # 2. Metrics presence
    try:
        mr = requests.get(f"{BASE}/metrics", timeout=3)
    except Exception as e:
        _fail(3, f"metrics request failed: {e}")
    if mr.status_code != 200:
        _fail(3, f"metrics non-200: {mr.status_code}")
    mtext = mr.text
    for key in ["vision_requests_total", "ocr_errors_total", "ocr_input_rejected_total"]:
        if key not in mtext:
            _fail(3, f"missing metric {key}")

    # 3. Vision minimal request
    tiny_png = base64.b64encode(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00\x00\x00\x00:~\x9bU\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    ).decode()
    v_payload = {
        "image_base64": tiny_png,
        "include_description": False,
        "include_ocr": False,
    }
    vr = requests.post(f"{BASE}/api/v1/vision/analyze", json=v_payload, timeout=5)
    if vr.status_code != 200:
        _fail(2, f"vision analyze non-200: {vr.status_code}")
    vdata = vr.json()
    if not vdata.get("success"):
        _fail(2, "vision analyze unexpected failure")

    # 4. OCR invalid MIME (should yield INPUT_ERROR)
    files = {"file": ("fake.txt", b"notimg", "text/plain")}
    orr = requests.post(f"{BASE}/api/v1/ocr/extract", files=files, timeout=5)
    if orr.status_code != 200:
        _fail(2, f"ocr extract non-200: {orr.status_code}")
    odata = orr.json()
    if odata.get("success") is True:
        _fail(4, "ocr invalid mime returned success")
    if odata.get("code") != "INPUT_ERROR":
        _fail(4, f"ocr code mismatch: {odata.get('code')}")

    print(json.dumps({"success": True, "error": None, "code": 0}))
    sys.exit(0)


if __name__ == "__main__":
    main()

