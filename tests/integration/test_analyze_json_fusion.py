import io
import json
import os

from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


def test_analyze_json_triggers_l2_fusion():
    payload = {
        "entities": [
            {"type": "LINE", "layer": "0", "start": [0, 0], "end": [10, 0]},
            {"type": "CIRCLE", "layer": "A", "center": [5, 5], "radius": 2},
        ],
        "text_content": ["Bolt_M6x20"],
        "meta": {"drawing_number": "Bolt_M6x20"},
    }
    options = {"extract_features": True, "classify_parts": True}
    resp = client.post(
        "/api/v1/analyze/",
        files={"file": ("Bolt_M6x20.json", io.BytesIO(json.dumps(payload).encode()), "application/json")},
        data={"options": json.dumps(options)},
        headers={"x-api-key": os.getenv("API_KEY", "test")},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    classification = data.get("results", {}).get("classification", {})
    assert classification.get("part_type") == "bolt"
    assert classification.get("confidence_source") == "fusion"
    assert classification.get("rule_version") == "L2-Fusion-v1"


def test_analyze_json_no_keyword_falls_back():
    payload = {
        "entities": [
            {"type": "LINE", "layer": "0", "start": [0, 0], "end": [10, 0]},
            {"type": "CIRCLE", "layer": "A", "center": [5, 5], "radius": 2},
        ],
        "text_content": ["Untitled"],
    }
    options = {"extract_features": True, "classify_parts": True}
    resp = client.post(
        "/api/v1/analyze/",
        files={"file": ("generic.json", io.BytesIO(json.dumps(payload).encode()), "application/json")},
        data={"options": json.dumps(options)},
        headers={"x-api-key": os.getenv("API_KEY", "test")},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    classification = data.get("results", {}).get("classification", {})
    assert classification.get("rule_version") != "L2-Fusion-v1"
