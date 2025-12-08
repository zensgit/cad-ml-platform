from fastapi.testclient import TestClient
from src.main import app
from src.core.similarity import register_vector  # type: ignore

client = TestClient(app)


def test_features_diff_basic():
    register_vector("diff_a", [0.1, 0.2, 0.3])
    register_vector("diff_b", [0.1, 0.25, 0.35])
    r = client.get("/api/v1/features/diff?id_a=diff_a&id_b=diff_b", headers={"x-api-key": "test"})
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"
    diffs = data.get("diffs")
    assert isinstance(diffs, list)
    # abs_diff checks (diffs are sorted by abs_diff descending)
    # The response field is "abs_diff", not "delta"
    abs_diffs = [d.get("abs_diff") for d in diffs]
    # Original vectors: [0.1, 0.2, 0.3] vs [0.1, 0.25, 0.35]
    # Differences: 0, 0.05, 0.05 - sorted descending so order may vary
    assert 0.05 in abs_diffs or 0.04999999999999999 in abs_diffs

