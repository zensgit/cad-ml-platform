from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_standards_status():
    resp = client.get("/api/v1/standards/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "counts" in data
    assert data["counts"]["threads"] > 0
    assert data["counts"]["bearings"] > 0
    assert data["counts"]["orings"] > 0


def test_standards_thread_m10():
    resp = client.get("/api/v1/standards/thread", params={"designation": "M10"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["designation"] == "M10"
    assert data["nominal_diameter_mm"] == 10.0
    assert data["pitch_mm"] == 1.5
    assert data["tap_drill_mm"] == 8.5


def test_standards_bearing_6205():
    resp = client.get("/api/v1/standards/bearing", params={"designation": "6205"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["designation"].startswith("6205")
    assert data["bore_mm"] == 25.0
    assert data["outer_d_mm"] == 52.0
    assert data["width_mm"] == 15.0


def test_standards_bearing_by_bore_25mm():
    resp = client.get("/api/v1/standards/bearing/by-bore", params={"bore_mm": 25})
    assert resp.status_code == 200
    data = resp.json()
    assert data["bore_mm"] == 25.0
    assert data["total"] > 0


def test_standards_oring_20x3():
    resp = client.get("/api/v1/standards/oring", params={"designation": "20x3"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["designation"] in {"20x3", "20x3.0"}
    assert data["inner_diameter_mm"] == 20.0
    assert data["cross_section_mm"] == 3.0
