import pytest
from fastapi.testclient import TestClient


def test_preview_response_structure_keys():
    from src.main import app

    client = TestClient(app)
    # Minimal query; adjust params if required by implementation
    resp = client.get(
        "/api/v1/vectors/migrate/preview",
        params={"to_version": "v4", "limit": 5},
        headers={"X-API-Key": "test"},
    )
    assert resp.status_code in (200, 401, 403)
    if resp.status_code == 200:
        data = resp.json()
        # Structure expectations: stats keys and warnings
        assert "avg_delta" in data
        assert "median_delta" in data
        assert "warnings" in data
        assert isinstance(data["warnings"], list)
