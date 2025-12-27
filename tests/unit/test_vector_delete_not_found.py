import uuid

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_vector_delete_not_found():
    missing_id = str(uuid.uuid4())
    r = client.post(
        "/api/v1/vectors/delete",
        json={"id": missing_id},
        headers={"X-API-Key": "test"},
    )
    assert r.status_code == 404
    body = r.json()
    assert "detail" in body and body["detail"]["code"] == "DATA_NOT_FOUND"
