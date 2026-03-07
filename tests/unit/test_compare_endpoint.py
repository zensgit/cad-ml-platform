import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def _headers() -> dict[str, str]:
    return {"X-API-Key": "test"}


def test_compare_endpoint_success():
    vector = [0.1, 0.2, 0.3]
    vid = "compare-test"
    register = client.post(
        "/api/v1/vectors/register",
        json={"id": vid, "vector": vector},
        headers=_headers(),
    )
    assert register.status_code == 200

    resp = client.post(
        "/api/compare",
        json={"query_features": vector, "candidate_hash": vid},
        headers=_headers(),
        follow_redirects=False,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["reference_id"] == vid
    assert data["method"] == "cosine"
    assert data["dimension"] == len(vector)
    assert data["score"] == data["similarity"]
    assert data["similarity"] == pytest.approx(1.0, abs=1e-4)
    assert data["feature_distance"] == pytest.approx(0.0, abs=1e-4)


def test_compare_endpoint_exposes_reference_label_contract():
    from src.core import similarity as sim_module

    vector = [0.2, 0.3, 0.4]
    vid = "compare-contract"
    register = client.post(
        "/api/v1/vectors/register",
        json={"id": vid, "vector": vector},
        headers=_headers(),
    )
    assert register.status_code == 200

    original_meta = sim_module._VECTOR_META.get(vid, {}).copy()
    sim_module._VECTOR_META[vid] = {
        **original_meta,
        "part_type": "人孔",
        "fine_part_type": "人孔",
        "coarse_part_type": "开孔件",
        "final_decision_source": "hybrid",
        "is_coarse_label": "false",
    }

    try:
        resp = client.post(
            "/api/compare",
            json={"query_features": vector, "candidate_hash": vid},
            headers=_headers(),
            follow_redirects=False,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["reference_id"] == vid
        assert data["reference_part_type"] == "人孔"
        assert data["reference_fine_part_type"] == "人孔"
        assert data["reference_coarse_part_type"] == "开孔件"
        assert data["reference_decision_source"] == "hybrid"
        assert data["reference_is_coarse_label"] is False
    finally:
        sim_module._VECTOR_STORE.pop(vid, None)
        sim_module._VECTOR_META.pop(vid, None)


def test_compare_endpoint_missing_candidate():
    resp = client.post(
        "/api/compare",
        json={"query_features": [0.1, 0.2], "candidate_hash": "missing"},
        headers=_headers(),
    )
    assert resp.status_code == 404
    detail = resp.json().get("detail") or {}
    assert detail.get("code") == "DATA_NOT_FOUND"


def test_compare_endpoint_dimension_mismatch():
    vector = [0.1, 0.2, 0.3]
    vid = "compare-dim"
    register = client.post(
        "/api/v1/vectors/register",
        json={"id": vid, "vector": vector},
        headers=_headers(),
    )
    assert register.status_code == 200

    resp = client.post(
        "/api/compare",
        json={"query_features": [0.1, 0.2], "candidate_hash": vid},
        headers=_headers(),
    )
    assert resp.status_code == 400
    detail = resp.json().get("detail") or {}
    assert detail.get("code") == "DIMENSION_MISMATCH"
