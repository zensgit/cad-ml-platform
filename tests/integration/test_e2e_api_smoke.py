import os
import uuid
from pathlib import Path
from typing import Dict, Optional

import httpx
import pytest


BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "test")
TIMEOUT = float(os.environ.get("E2E_HTTP_TIMEOUT", "10"))
VISION_REQUIRED = os.environ.get("DEDUPCAD_VISION_REQUIRED", "0") == "1"

DXF_PATH = Path(os.environ.get("E2E_DXF_PATH", "data/dxf_fixtures_subset/mixed.dxf"))
PNG_PATH = Path(os.environ.get("E2E_PNG_PATH", "data/dxf_fixtures_subset_out/mixed.png"))


def _headers() -> Dict[str, str]:
    return {"X-API-Key": API_KEY}


def _check_health() -> None:
    try:
        resp = httpx.get(f"{BASE_URL}/health", headers=_headers(), timeout=2.0)
    except Exception:
        pytest.skip("API not reachable; skipping E2E smoke")
    if resp.status_code != 200:
        pytest.skip(f"API health check failed ({resp.status_code}); skipping E2E smoke")


def _post_file(path: Path, endpoint: str, params: Optional[Dict[str, str]] = None) -> httpx.Response:
    with path.open("rb") as handle:
        files = {"file": (path.name, handle, "application/octet-stream")}
        return httpx.post(
            f"{BASE_URL}{endpoint}",
            headers=_headers(),
            files=files,
            params=params,
            timeout=TIMEOUT,
        )


def test_e2e_core_api_smoke() -> None:
    _check_health()

    if not DXF_PATH.exists():
        pytest.skip("DXF fixture missing; skipping E2E smoke")

    analyze_resp = _post_file(DXF_PATH, "/api/v1/analyze/")
    if analyze_resp.status_code == 404:
        pytest.skip("Analyze endpoint not found; skipping E2E smoke")
    assert analyze_resp.status_code == 200
    analyze_data = analyze_resp.json()
    combined = (
        analyze_data.get("results", {})
        .get("features", {})
        .get("combined")
    )
    assert isinstance(combined, list)
    assert len(combined) >= 5
    if all(value == 0 for value in combined):
        combined = [float(idx + 1) / 10 for idx in range(len(combined))]

    vector_id = f"e2e-{uuid.uuid4().hex[:8]}"
    register_payload = {
        "id": vector_id,
        "vector": combined,
        "meta": {
            "material": "steel",
            "complexity": "low",
            "format": "dxf",
        },
    }
    register_resp = httpx.post(
        f"{BASE_URL}/api/v1/vectors/register",
        headers=_headers(),
        json=register_payload,
        timeout=TIMEOUT,
    )
    if register_resp.status_code == 404:
        pytest.skip("Vector endpoints not found; skipping E2E smoke")
    assert register_resp.status_code == 200
    assert register_resp.json().get("status") == "accepted"

    search_resp = httpx.post(
        f"{BASE_URL}/api/v1/vectors/search",
        headers=_headers(),
        json={"vector": combined, "k": 5},
        timeout=TIMEOUT,
    )
    assert search_resp.status_code == 200
    search_ids = [item.get("id") for item in search_resp.json().get("results", [])]
    assert vector_id in search_ids

    list_resp = httpx.get(
        f"{BASE_URL}/api/v1/vectors/",
        headers=_headers(),
        params={"source": "auto", "limit": 10},
        timeout=TIMEOUT,
    )
    assert list_resp.status_code == 200
    list_payload = list_resp.json()
    assert list_payload.get("total", 0) >= 1

    stats_resp = httpx.get(
        f"{BASE_URL}/api/v1/vectors_stats/stats",
        headers=_headers(),
        timeout=TIMEOUT,
    )
    assert stats_resp.status_code == 200
    stats_payload = stats_resp.json()
    assert stats_payload.get("total", 0) >= 1
    assert stats_payload.get("backend") in {"memory", "redis"}

    knowledge_resp = httpx.get(
        f"{BASE_URL}/api/v1/maintenance/knowledge/status",
        headers=_headers(),
        timeout=TIMEOUT,
    )
    assert knowledge_resp.status_code == 200
    knowledge_payload = knowledge_resp.json()
    assert "version" in knowledge_payload
    assert "by_category" in knowledge_payload

    try:
        httpx.post(
            f"{BASE_URL}/api/v1/vectors/delete",
            headers=_headers(),
            json={"id": vector_id},
            timeout=TIMEOUT,
        )
    except Exception:
        pass


def test_e2e_dedup_search_smoke() -> None:
    _check_health()

    if not PNG_PATH.exists():
        pytest.skip("PNG fixture missing; skipping dedup smoke")

    try:
        resp = _post_file(
            PNG_PATH,
            "/api/v1/dedup/2d/search",
            params={"mode": "balanced", "max_results": "5", "compute_diff": "false"},
        )
    except Exception:
        if VISION_REQUIRED:
            pytest.fail("Dedup search not reachable")
        pytest.skip("Dedup search not reachable; skipping smoke")

    if resp.status_code in {404, 502, 503, 504}:
        if VISION_REQUIRED:
            pytest.fail(f"Dedup search unavailable ({resp.status_code})")
        pytest.skip(f"Dedup search unavailable ({resp.status_code}); skipping smoke")

    assert resp.status_code == 200
    payload = resp.json()
    for key in ("success", "total_matches", "duplicates", "similar", "final_level", "timing"):
        assert key in payload
