"""Smoke test for drawing recognition endpoint."""

from typing import Optional

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.v1 import drawing
from src.core.ocr.base import (
    DimensionInfo,
    DimensionType,
    HeatTreatmentInfo,
    HeatTreatmentType,
    IdentifierInfo,
    OcrResult,
    ProcessRequirements,
    SurfaceTreatmentInfo,
    SurfaceTreatmentType,
    SymbolInfo,
    SymbolType,
    TitleBlock,
    WeldingInfo,
    WeldingType,
)

SAMPLE_BASE64_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMBAkYp"
    "9V0AAAAASUVORK5CYII="
)


class DummyManager:
    async def extract(
        self,
        image_bytes: bytes,
        strategy: str = "auto",
        trace_id: Optional[str] = None,
    ) -> OcrResult:
        return OcrResult(
            text="材料 Aluminum 技术要求：未注公差按GB/T1804-m执行 氩弧焊 焊丝ER50-6 表面Ra3.2",
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
            symbols=[
                SymbolInfo(type=SymbolType.surface_roughness, value="3.2"),
                SymbolInfo(type=SymbolType.position, value="0.05"),
            ],
            process_requirements=ProcessRequirements(
                heat_treatments=[
                    HeatTreatmentInfo(
                        type=HeatTreatmentType.quenching,
                        hardness="HRC58-62",
                        hardness_min=58.0,
                        hardness_max=62.0,
                        hardness_unit="HRC",
                        raw="整体淬火 HRC58-62",
                        confidence=0.8,
                    )
                ],
                surface_treatments=[
                    SurfaceTreatmentInfo(
                        type=SurfaceTreatmentType.galvanizing,
                        thickness=10.0,
                        standard="GB/T 13912",
                        raw="表面镀锌 镀层厚度10μm",
                        confidence=0.7,
                    )
                ],
                welding=[
                    WeldingInfo(
                        type=WeldingType.tig_welding,
                        filler_material="ER50-6",
                        raw="氩弧焊 焊丝ER50-6",
                        confidence=0.7,
                    )
                ],
                general_notes=["未注公差按GB/T1804-m执行", "去毛刺倒钝"],
                raw_text="未注公差按GB/T1804-m执行; 氩弧焊 焊丝ER50-6; 表面镀锌 GB/T 13912",
            ),
            identifiers=[
                IdentifierInfo(
                    identifier_type="drawing_number",
                    label="Drawing Number",
                    value="DWG-123",
                    normalized_value="DWG-123",
                    source_text="图号: DWG-123",
                    bbox=[10, 10, 80, 12],
                    confidence=0.93,
                    source="ocr_line",
                ),
                IdentifierInfo(
                    identifier_type="material",
                    label="Material",
                    value="Aluminum",
                    normalized_value="Aluminum",
                    source_text="材料: Aluminum",
                    bbox=[10, 30, 80, 12],
                    confidence=0.71,
                    source="ocr_line",
                ),
            ],
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
    field_confidence = data["field_confidence"]
    assert field_map["drawing_number"]["value"] == "DWG-123"
    assert field_map["drawing_number"]["confidence"] == 0.93
    assert field_map["material"]["confidence"] == 0.71
    assert field_map["part_name"]["confidence"] == 0.9
    assert field_map["part_name"]["value"] == "Bracket"
    assert title_block["drawing_number"] == "DWG-123"
    assert title_block["material"] == "Aluminum"
    assert field_confidence["drawing_number"] == 0.93
    assert field_confidence["material"] == 0.71
    assert field_confidence["part_name"] == 0.9
    assert data["identifiers"][0]["identifier_type"] == "drawing_number"
    assert data["identifiers"][0]["bbox"] == [10, 10, 80, 12]
    assert data["identifiers"][1]["identifier_type"] == "material"
    assert data["field_evidence"]["drawing_number"]["value"] == "DWG-123"
    assert data["field_evidence"]["drawing_number"]["bbox"] == [10, 10, 80, 12]
    assert data["dimensions"]
    assert data["process_requirements"]["heat_treatments"][0]["type"] == "quenching"
    assert data["process_requirements"]["surface_treatments"][0]["standard"] == "GB/T 13912"
    assert data["process_requirements"]["general_notes"][0] == "未注公差按GB/T1804-m执行"
    assert data["field_coverage"]["recognized_count"] == 6
    assert "drawing_number" in data["field_coverage"]["recognized_keys"]
    assert "company" in data["field_coverage"]["missing_keys"]
    assert data["engineering_signals"]["dimension_count"] == 1
    assert data["engineering_signals"]["symbol_count"] == 2
    assert data["engineering_signals"]["has_surface_finish"] is True
    assert data["engineering_signals"]["has_gdt"] is True
    assert "position" in data["engineering_signals"]["gdt_symbol_types"]
    assert "Aluminum" in data["engineering_signals"]["materials_detected"]
    assert data["review_hints"]["review_recommended"] is False
    assert data["review_hints"]["missing_critical_fields"] == []
    assert data["review_hints"]["review_priority"] == "low"
    assert data["review_hints"]["automation_ready"] is True
    assert data["review_hints"]["recommended_actions"] == []
    assert data["review_hints"]["readiness_band"] == "high"
    assert "GB/T1804-M" in [
        candidate.upper() for candidate in data["engineering_signals"]["standards_candidates"]
    ]
    assert "GB/T13912" in data["engineering_signals"]["standards_candidates"]


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


def test_drawing_recognize_base64_smoke(monkeypatch) -> None:
    monkeypatch.setattr(drawing, "get_manager", lambda: DummyManager())

    app = FastAPI()
    app.include_router(drawing.router, prefix="/api/v1/drawing")
    client = TestClient(app)

    payload = {"image_base64": SAMPLE_BASE64_PNG, "provider": "auto"}
    resp = client.post("/api/v1/drawing/recognize-base64", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert data["success"] is True
    assert data["title_block"]["drawing_number"] == "DWG-123"
    assert data["engineering_signals"]["process_requirement_counts"]["welding"] == 1
    assert data["field_evidence"]["material"]["value"] == "Aluminum"
    assert data["review_hints"]["readiness_score"] >= 0.8
    assert data["review_hints"]["automation_ready"] is True
