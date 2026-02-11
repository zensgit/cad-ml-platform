from fastapi.testclient import TestClient

from src.api.v1.tolerance import _normalize_fit_code
from src.main import app


client = TestClient(app)


def test_normalize_fit_code_basic_case():
    assert _normalize_fit_code("h7/G6") == "H7/g6"


def test_normalize_fit_code_with_whitespace():
    assert _normalize_fit_code("  h7  /  G6  ") == "H7/g6"


def test_normalize_fit_code_without_slash_kept_raw():
    assert _normalize_fit_code("H7g6") == "H7g6"


def test_normalize_fit_code_invalid_symbol_kept_raw():
    assert _normalize_fit_code("H_7/g6") == "H_7/g6"


def test_normalize_fit_code_three_letter_symbol_supported():
    assert _normalize_fit_code("js7/H6") == "JS7/h6"


def test_tolerance_fit_endpoint_accepts_whitespace_and_mixed_case():
    resp = client.get(
        "/api/v1/tolerance/fit",
        params={"fit_code": "  h7  /  G6  ", "diameter_mm": 25},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["fit_code"] == "H7/g6"
    assert data["fit_type"] == "clearance"

