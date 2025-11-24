from fastapi.testclient import TestClient
from src.main import app
from src.core.similarity import register_vector  # type: ignore

client = TestClient(app)


def test_features_diff_basic():
    register_vector("diff_a", [0.1, 0.2, 0.3])
    register_vector("diff_b", [0.1, 0.25, 0.35])
    r = client.get("/api/v1/features/diff?id_a=diff_a&id_b=diff_b", headers={"api-key": "test"})
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"
    diffs = data.get("diffs")
    assert isinstance(diffs, list)
    # delta checks
    deltas = [d.get("delta") for d in diffs]
    assert deltas[1] == 0.05
    assert deltas[2] == 0.04999999999999999 or deltas[2] == 0.05

