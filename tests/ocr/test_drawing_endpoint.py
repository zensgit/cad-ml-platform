"""Smoke test for drawing recognition endpoint."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.v1 import drawing
from src.core.ocr.base import DimensionInfo, DimensionType, OcrResult, TitleBlock


class DummyManager:
    async def extract(self, image_bytes: bytes, strategy: str = "auto", trace_id: str | None = None) -> OcrResult:
        return OcrResult(
            title_block=TitleBlock(
                drawing_number="DWG-123",
                revision="A",
                part_name="Bracket",
                material="Aluminum",
                scale="1:2",
                sheet="1/2",
            ),
            title_block_confidence={"drawing_number": 0.93, "material": 0.71},
            dimensions=[DimensionInfo(type=DimensionType.diameter, value=20.0)],
            symbols=[],
            confidence=0.9,
            processing_time_ms=12,
        )


def test_drawing_recognize_smoke(monkeypatch) -> None:
    monkeypatch.setattr(drawing, "get_manager", lambda: DummyManager())

    app = FastAPI()
    app.include_router(drawing.router, prefix="/api/v1/drawing")
    client = TestClient(app)

    files = {"file": ("test.png", b"fake_image_bytes", "image/png")}
    resp = client.post("/api/v1/drawing/recognize?provider=auto", files=files)
    assert resp.status_code == 200

    data = resp.json()
    field_map = {field["key"]: field for field in data["fields"]}
    title_block = data["title_block"]
    assert field_map["drawing_number"]["value"] == "DWG-123"
    assert field_map["drawing_number"]["confidence"] == 0.93
    assert field_map["material"]["confidence"] == 0.71
    assert field_map["part_name"]["confidence"] == 0.9
    assert field_map["part_name"]["value"] == "Bracket"
    assert title_block["drawing_number"] == "DWG-123"
    assert title_block["material"] == "Aluminum"
    assert data["dimensions"]


def test_drawing_fields_catalog() -> None:
    app = FastAPI()
    app.include_router(drawing.router, prefix="/api/v1/drawing")
    client = TestClient(app)

    resp = client.get("/api/v1/drawing/fields")
    assert resp.status_code == 200
    payload = resp.json()
    field_keys = {field["key"] for field in payload["fields"]}
    assert "drawing_number" in field_keys
    assert "revision" in field_keys
