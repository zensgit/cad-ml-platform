"""Tests for cache tuning recommendation POST endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.utils.analysis_metrics import (
    feature_cache_tuning_recommended_capacity,
    feature_cache_tuning_recommended_ttl_seconds,
    feature_cache_tuning_requests_total,
)

client = TestClient(app)


def _counter_value(counter, status: str) -> float:
    for sample in counter.collect()[0].samples:
        if sample.labels.get("status") == status:
            return sample.value
    return 0.0


@pytest.mark.parametrize(
    "hit_rate,expected_capacity,expected_ttl,expected_reason",
    [
        (0.35, "increase", "same", "low hit rate"),
        (0.39, "increase", "same", "low hit rate"),
        (0.40, "same", "decrease", "moderate hit rate"),
        (0.70, "same", "same", "target band"),
        (0.85, "same", "same", "target band"),
        (0.90, "decrease", "same", "high hit rate"),
    ],
)
def test_cache_tuning_boundaries(
    hit_rate: float, expected_capacity: str, expected_ttl: str, expected_reason: str
) -> None:
    payload = {"hit_rate": hit_rate, "capacity": 1000, "ttl": 1000, "window_hours": 2}
    response = client.post(
        "/api/v1/features/cache/tuning", json=payload, headers={"X-API-Key": "test"}
    )
    assert response.status_code == 200
    data = response.json()

    if expected_capacity == "increase":
        assert data["recommended_capacity"] > payload["capacity"]
    elif expected_capacity == "decrease":
        assert data["recommended_capacity"] < payload["capacity"]
    else:
        assert data["recommended_capacity"] == payload["capacity"]

    if expected_ttl == "decrease":
        assert data["recommended_ttl"] < payload["ttl"]
    else:
        assert data["recommended_ttl"] == payload["ttl"]

    assert 0.0 <= data["confidence"] <= 1.0
    assert data["reasoning"], "reasoning should not be empty"
    assert any(expected_reason in reason.lower() for reason in data["reasoning"])
    assert data["experimental"] is True


def test_cache_tuning_metrics_increment() -> None:
    if not hasattr(feature_cache_tuning_requests_total, "collect"):
        pytest.skip("prometheus client disabled in this environment")

    payload = {"hit_rate": 0.35, "capacity": 1000, "ttl": 1000, "window_hours": 2}
    before = _counter_value(feature_cache_tuning_requests_total, "ok")
    response = client.post(
        "/api/v1/features/cache/tuning", json=payload, headers={"X-API-Key": "test"}
    )
    assert response.status_code == 200
    after = _counter_value(feature_cache_tuning_requests_total, "ok")
    assert after > before
    cap_value = feature_cache_tuning_recommended_capacity.collect()[0].samples[0].value
    ttl_value = feature_cache_tuning_recommended_ttl_seconds.collect()[0].samples[0].value
    assert cap_value == response.json()["recommended_capacity"]
    assert ttl_value == response.json()["recommended_ttl"]
