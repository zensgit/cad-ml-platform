from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_design_standards_status():
    resp = client.get("/api/v1/design-standards/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["counts"]["surface_finish_grades"] > 0
    assert data["counts"]["linear_tolerance_ranges"] > 0
    assert data["counts"]["preferred_diameters"] > 0


def test_design_standards_surface_finish_grade_n7():
    resp = client.get("/api/v1/design-standards/surface-finish/grade", params={"grade": "N7"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["grade"] == "N7"
    assert data["ra_um"] == 1.6
    assert data["rz_um"] == 6.3


def test_design_standards_surface_finish_grades_list_contains_n7():
    resp = client.get("/api/v1/design-standards/surface-finish/grades")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0
    grades = data["grades"]
    assert any(item["grade"] == "N7" and item["ra_um"] == 1.6 for item in grades)


def test_design_standards_surface_finish_application_bearing_journal():
    resp = client.get(
        "/api/v1/design-standards/surface-finish/application",
        params={"application": "bearing_journal"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["application"] == "bearing_journal"
    assert data["grade"] == "N6"
    assert data["ra_value_um"] == 0.8
    assert data["ra_range_min_um"] == 0.4
    assert data["ra_range_max_um"] == 1.6


def test_design_standards_general_linear_tolerance_50_m():
    resp = client.get(
        "/api/v1/design-standards/general-tolerances/linear",
        params={"dimension_mm": 50, "tolerance_class": "m"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["dimension_mm"] == 50.0
    assert data["tolerance_class"] == "m"
    assert data["tolerance_plus_mm"] == 0.3
    assert data["tolerance_minus_mm"] == -0.3


def test_design_standards_general_angular_tolerance_80_m():
    resp = client.get(
        "/api/v1/design-standards/general-tolerances/angular",
        params={"length_mm": 80, "tolerance_class": "m"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["length_mm"] == 80.0
    assert data["tolerance_class"] == "m"
    assert "°" in data["tolerance"]


def test_design_standards_preferred_diameter_nearest_23mm():
    resp = client.get(
        "/api/v1/design-standards/preferred-diameter",
        params={"target_mm": 23, "direction": "nearest"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_mm"] == 23.0
    assert data["direction"] == "nearest"
    assert data["preferred_mm"] == 22.0


def test_design_standards_preferred_diameters_list_range():
    resp = client.get(
        "/api/v1/design-standards/design-features/preferred-diameters",
        params={"min_mm": 20, "max_mm": 25},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["min_mm"] == 20.0
    assert data["max_mm"] == 25.0
    assert data["total"] > 0
    values = set(data["diameters_mm"])
    assert 20.0 in values
    assert 22.0 in values
    assert 24.0 in values
    assert 25.0 in values


def test_design_standards_standard_chamfer_near_1_8mm():
    resp = client.get(
        "/api/v1/design-standards/design-features/chamfer",
        params={"target_size_mm": 1.8},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["designation"] == "C2"
    assert data["size_mm"] == 2.0


def test_design_standards_standard_fillet_near_2_3mm():
    resp = client.get(
        "/api/v1/design-standards/design-features/fillet",
        params={"target_radius_mm": 2.3},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["designation"] == "R2.5"
    assert data["size_mm"] == 2.5
