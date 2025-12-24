"""Tests: vector migration dimension mismatch structured error.

Goals (TODO):
1. Seed a vector with incorrect length vs declared feature_version.
2. Migrate (e.g. to v3) expecting DIMENSION_MISMATCH (HTTP 409) structured error.
3. Validate error schema fields: code, stage, expected, found, id.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


@pytest.mark.skip(reason="TODO: implement mismatch seeding & migration call")
def test_vector_migrate_dimension_mismatch_error() -> None:
    response = client.post(
        "/api/v1/vectors/migrate", json={"ids": ["nonexistent"], "to_version": "v3"}
    )
    assert response.status_code in {200, 404, 409}
