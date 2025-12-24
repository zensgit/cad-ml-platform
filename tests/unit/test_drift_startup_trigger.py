"""Tests: drift baseline startup refresh trigger metric.

Goals (TODO):
1. Ensure application startup loads baselines and records drift_baseline_refresh_total{trigger="startup"}.
2. Call /api/v1/analyze/drift/baseline/status and inspect fields.
3. Potentially expose metrics endpoint (/metrics) and parse to confirm counter increment.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


@pytest.mark.skip(reason="TODO: implement metrics scrape & verification of startup trigger")
def test_drift_startup_trigger_metric_present() -> None:
    response = client.get("/api/v1/analyze/drift/baseline/status")
    assert response.status_code in {200, 204}
