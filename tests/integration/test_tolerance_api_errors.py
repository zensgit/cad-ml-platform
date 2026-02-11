from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


def test_tolerance_it_unsupported_grade_returns_400():
    resp = client.get("/api/v1/tolerance/it", params={"diameter_mm": 10, "grade": "IT99"})
    assert resp.status_code == 400
    data = resp.json()
    assert "detail" in data
    assert "Unsupported size/grade" in data["detail"]


def test_tolerance_it_invalid_diameter_returns_422():
    resp = client.get("/api/v1/tolerance/it", params={"diameter_mm": 0, "grade": "IT7"})
    assert resp.status_code == 422
    data = resp.json()
    assert "detail" in data


def test_tolerance_limit_deviations_not_found_returns_404():
    resp = client.get(
        "/api/v1/tolerance/limit-deviations",
        params={"symbol": "ZZ", "grade": 7, "diameter_mm": 10},
    )
    assert resp.status_code == 404
    data = resp.json()
    assert data.get("detail") == "Limit deviations not found for given symbol/grade/size."


def test_tolerance_fit_not_supported_returns_404():
    resp = client.get(
        "/api/v1/tolerance/fit",
        params={"fit_code": "ZZ7/yy6", "diameter_mm": 10},
    )
    assert resp.status_code == 404
    data = resp.json()
    assert data.get("detail") == "Fit code not supported or out of range."

