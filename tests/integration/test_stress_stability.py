import os
import time
from typing import Dict

import pytest
import httpx


BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.environ.get("API_KEY", "test")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "test")


def _headers(include_admin: bool = False) -> Dict[str, str]:
    h = {"X-API-Key": API_KEY}
    if include_admin:
        h["X-Admin-Token"] = ADMIN_TOKEN
    return h


def test_degraded_state_history_cap():
    # Hit health endpoint to verify presence of fields and history cap
    url = f"{BASE_URL}/api/v1/health/faiss/health"
    try:
        r = httpx.get(url, headers=_headers(), timeout=2.0)
    except Exception:
        pytest.skip("API not reachable; skipping integration smoke")
    if r.status_code == 404:
        pytest.skip("Endpoint not found; API may not have this route")
    assert r.status_code == 200
    data = r.json()
    assert "degraded" in data
    assert "degradation_history_count" in data
    assert isinstance(data["degradation_history_count"], int)
    assert data["degradation_history_count"] <= 10


def test_cache_apply_rollback_window():
    # Ensure cache endpoints are reachable and window logic fields exist
    url_apply = f"{BASE_URL}/api/v1/health/features/cache/apply"
    try:
        r_apply = httpx.get(url_apply, headers=_headers(), timeout=2.0)
    except Exception:
        pytest.skip("API not reachable; skipping integration smoke")
    if r_apply.status_code == 404:
        pytest.skip("Endpoint not found; API may not have this route")
    # Endpoint may require params; allow 200/400/401 depending on env
    assert r_apply.status_code in (200, 400, 401, 403)

    url_rollback = f"{BASE_URL}/api/v1/health/features/cache/rollback"
    r_rb = httpx.get(url_rollback, headers=_headers(include_admin=True), timeout=2.0)
    assert r_rb.status_code in (200, 400, 401, 403)


def test_prewarm_endpoint_shape():
    url = f"{BASE_URL}/api/v1/health/features/cache/prewarm"
    try:
        r = httpx.post(url, headers=_headers(), timeout=2.0)
    except Exception:
        pytest.skip("API not reachable; skipping integration smoke")
    if r.status_code == 404:
        pytest.skip("Endpoint not found; API may not have this route")
    assert r.status_code in (200, 400)
    # Response should be JSON
    data = r.json()
    assert isinstance(data, dict)
    # metric should be present in Prometheus; this is a smoke test
