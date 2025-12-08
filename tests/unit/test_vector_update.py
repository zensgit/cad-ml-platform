import os
import uuid
from fastapi.testclient import TestClient

from src.main import app
from src.core.similarity import register_vector


client = TestClient(app)


def _auth_headers():
    return {"X-API-Key": os.getenv("TEST_API_KEY", "test")}


def test_vector_update_append_and_replace():
    vid = str(uuid.uuid4())
    register_vector(vid, [0.1, 0.2, 0.3], meta={"material": "steel"})
    # Append (enforcement off)
    resp = client.post(
        "/api/v1/vectors/update",
        json={"id": vid, "append": [0.4, 0.5], "complexity": "low"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "updated"
    assert data["dimension"] == 5
    # Replace wrong dimension (enforcement off)
    resp2 = client.post(
        "/api/v1/vectors/update",
        json={"id": vid, "replace": [0.9, 0.8]},
        headers=_auth_headers(),
    )
    assert resp2.status_code == 200
    d2 = resp2.json()
    assert d2["status"] in {"dimension_mismatch", "rejected"}


def test_vector_update_enforcement():
    vid = str(uuid.uuid4())
    register_vector(vid, [0.11, 0.22, 0.33])
    os.environ["ANALYSIS_VECTOR_DIM_CHECK"] = "1"
    # Append should be rejected - API returns 409 Conflict for dimension violations
    r = client.post(
        "/api/v1/vectors/update",
        json={"id": vid, "append": [0.44]},
        headers=_auth_headers(),
    )
    assert r.status_code in (200, 409)  # Either 200 with rejected status or 409 Conflict
    if r.status_code == 200:
        assert r.json()["status"] == "rejected"
    # Replace wrong dimension rejected
    r2 = client.post(
        "/api/v1/vectors/update",
        json={"id": vid, "replace": [0.55]},
        headers=_auth_headers(),
    )
    assert r2.status_code in (200, 409)
    if r2.status_code == 200:
        assert r2.json()["status"] == "rejected"
    os.environ.pop("ANALYSIS_VECTOR_DIM_CHECK", None)
