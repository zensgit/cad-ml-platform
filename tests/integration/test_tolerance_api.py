from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_tolerance_it_it7_25mm():
    resp = client.get("/api/v1/tolerance/it", params={"diameter_mm": 25, "grade": "IT7"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["grade"] == "IT7"
    assert data["nominal_size_mm"] == 25.0
    assert data["tolerance_um"] == 21.0


def test_tolerance_limit_deviations_h7_25mm():
    resp = client.get(
        "/api/v1/tolerance/limit-deviations",
        params={"symbol": "H", "grade": 7, "diameter_mm": 25},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "H"
    assert data["grade"] == 7
    assert data["lower_deviation_um"] == 0.0
    assert data["upper_deviation_um"] == 21.0
    assert data["label"].upper() == "H7"


def test_tolerance_fit_deviations_h7_g6_25mm():
    resp = client.get(
        "/api/v1/tolerance/fit",
        params={"fit_code": "h7/G6", "diameter_mm": 25},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["fit_code"] == "H7/g6"
    assert data["fit_type"] == "clearance"
    assert data["nominal_size_mm"] == 25.0
    assert data["hole_lower_deviation_um"] == 0.0
    assert data["hole_upper_deviation_um"] == 21.0
    assert data["shaft_upper_deviation_um"] == -7.0
    assert data["shaft_lower_deviation_um"] == -20.0
    assert data["min_clearance_um"] == 7.0
    assert data["max_clearance_um"] == 41.0

