"""Smoke test for /api/v1/ocr/extract endpoint using FastAPI TestClient."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi import APIRouter, UploadFile, File
from src.core.ocr.manager import OcrManager
from src.core.ocr.providers.paddle import PaddleOcrProvider
from src.core.ocr.providers.deepseek_hf import DeepSeekHfProvider


app = FastAPI()

manager = OcrManager(confidence_fallback=0.85)
manager.register_provider("paddle", PaddleOcrProvider())
manager.register_provider("deepseek_hf", DeepSeekHfProvider())

router = APIRouter(prefix="/v1/ocr")

@router.post("/extract")
async def extract(file: UploadFile = File(...), provider: str = "auto"):
    data = await file.read()
    result = await manager.extract(data, strategy=provider)
    return {
        "provider": result.provider,
        "dimensions": [d.model_dump() for d in result.dimensions],
        "symbols": [s.model_dump() for s in result.symbols],
        "processing_time_ms": result.processing_time_ms,
    }

app.include_router(router, prefix="/api")
client = TestClient(app)


def test_ocr_extract_smoke():
    files = {"file": ("test.png", b"fake_image_bytes", "image/png")}
    resp = client.post("/api/v1/ocr/extract?provider=auto", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["provider"] in ("paddle", "deepseek_hf", "auto")
    assert isinstance(data["dimensions"], list)
    assert "processing_time_ms" in data
