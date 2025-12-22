from __future__ import annotations

import json
from pathlib import Path

from jsonschema import validate

from src.core.vision import VisionAnalyzeResponse, VisionDescription


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "contracts").exists() and (parent / "src").exists():
            return parent
    raise RuntimeError("Repository root not found")


def _load_schema(name: str) -> dict:
    schema_path = _repo_root() / "contracts" / name
    return json.loads(schema_path.read_text(encoding="utf-8"))


def test_vision_analyze_contract_schema() -> None:
    schema = _load_schema("cad_ml_vision_analyze.schema.json")
    response = VisionAnalyzeResponse(
        success=True,
        description=VisionDescription(
            summary="Mechanical part with cylindrical features",
            details=["Main diameter: 20mm"],
            confidence=0.9,
        ),
        ocr=None,
        provider="stub",
        processing_time_ms=123.4,
        error=None,
        code=None,
    )
    validate(instance=response.model_dump(), schema=schema)
