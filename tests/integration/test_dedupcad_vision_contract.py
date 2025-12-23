import os
from pathlib import Path
from typing import Dict

import httpx
import pytest


VISION_URL = os.environ.get("DEDUPCAD_VISION_URL", "http://localhost:58001")
TIMEOUT = float(os.environ.get("E2E_HTTP_TIMEOUT", "20"))
PNG_PATH = Path(os.environ.get("E2E_PNG_PATH", "data/dxf_fixtures_subset_out/mixed.png"))
VISION_REQUIRED = os.environ.get("DEDUPCAD_VISION_REQUIRED", "0") == "1"


def _skip_or_fail(message: str) -> None:
    if VISION_REQUIRED:
        pytest.fail(message)
    pytest.skip(message)


def _check_health() -> Dict[str, object]:
    try:
        resp = httpx.get(f"{VISION_URL}/health", timeout=5.0)
    except Exception:
        _skip_or_fail("dedupcad-vision not reachable; skipping contract tests")
    if resp.status_code != 200:
        _skip_or_fail(f"dedupcad-vision health returned {resp.status_code}")
    payload = resp.json()
    if "status" not in payload:
        _skip_or_fail("dedupcad-vision health payload missing status")
    return payload


def test_vision_health_contract() -> None:
    payload = _check_health()
    assert "status" in payload


def test_vision_search_contract() -> None:
    _check_health()
    if not PNG_PATH.exists():
        pytest.skip("PNG fixture missing; skipping contract tests")

    with PNG_PATH.open("rb") as handle:
        files = {"file": (PNG_PATH.name, handle, "application/octet-stream")}
        data = {
            "mode": "balanced",
            "max_results": "5",
            "compute_diff": "false",
        }
        try:
            resp = httpx.post(
                f"{VISION_URL}/api/v2/search",
                files=files,
                data=data,
                timeout=TIMEOUT,
            )
        except Exception:
            _skip_or_fail("dedupcad-vision search not reachable; skipping contract tests")

    if resp.status_code in {404, 502, 503, 504}:
        _skip_or_fail(f"dedupcad-vision search unavailable ({resp.status_code})")

    assert resp.status_code == 200
    payload = resp.json()
    for key in (
        "success",
        "total_matches",
        "duplicates",
        "similar",
        "final_level",
        "timing",
        "level_stats",
        "warnings",
        "error",
    ):
        assert key in payload
    timing = payload.get("timing", {})
    for key in ("total_ms", "l1_ms", "l2_ms", "l3_ms", "l4_ms"):
        assert key in timing
