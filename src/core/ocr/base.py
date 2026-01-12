"""OCR core data models and abstract client interface.

Week1 scaffolding: minimal Pydantic schemas + abstract provider base.
Implementation keeps surface area small; providers will extend OcrClient.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Protocol

from pydantic import BaseModel, Field


class DimensionType(str, Enum):
    diameter = "diameter"
    radius = "radius"
    length = "length"
    thread = "thread"


class SymbolType(str, Enum):
    surface_roughness = "surface_roughness"
    perpendicular = "perpendicular"
    parallel = "parallel"
    angularity = "angularity"
    position = "position"
    concentricity = "concentricity"
    flatness = "flatness"
    straightness = "straightness"
    circularity = "circularity"
    cylindricity = "cylindricity"
    symmetry = "symmetry"
    runout = "runout"
    total_runout = "total_runout"
    profile_line = "profile_line"
    profile_surface = "profile_surface"


class DimensionInfo(BaseModel):
    type: DimensionType
    value: float
    unit: str = Field("mm", description="Normalized unit")
    tolerance: Optional[float] = None
    tol_pos: Optional[float] = Field(None, description="Positive tolerance if dual")
    tol_neg: Optional[float] = Field(
        None, description="Negative tolerance if dual (absolute value)"
    )
    pitch: Optional[float] = Field(None, description="Thread pitch if thread")
    raw: Optional[str] = Field(None, description="Raw extracted text segment")
    bbox: Optional[list[int]] = Field(None, description="[x,y,w,h] normalized bbox")
    confidence: Optional[float] = Field(None, description="Per-dimension confidence if available")


class SymbolInfo(BaseModel):
    type: SymbolType
    value: str
    normalized_form: Optional[str] = None
    bbox: Optional[list[int]] = None
    confidence: Optional[float] = Field(None, description="Per-symbol confidence if available")


class TitleBlock(BaseModel):
    drawing_number: Optional[str] = None
    revision: Optional[str] = None
    material: Optional[str] = None
    part_name: Optional[str] = None
    scale: Optional[str] = None
    sheet: Optional[str] = None
    date: Optional[str] = None
    weight: Optional[str] = None
    company: Optional[str] = None
    projection: Optional[str] = None


class OcrStage(str, Enum):
    preprocess = "preprocess"
    infer = "infer"
    parse = "parse"
    postprocess = "postprocess"


class OcrResult(BaseModel):
    text: Optional[str] = None
    dimensions: List[DimensionInfo] = Field(default_factory=list)
    symbols: List[SymbolInfo] = Field(default_factory=list)
    title_block: TitleBlock = Field(default_factory=TitleBlock)
    title_block_confidence: Dict[str, float] = Field(default_factory=dict)
    confidence: Optional[float] = None
    calibrated_confidence: Optional[float] = None
    completeness: Optional[float] = None  # parsing coverage ratio
    provider: Optional[str] = None
    fallback_level: Optional[str] = None
    extraction_mode: Optional[str] = None  # provider_native|json_only|regex_only|json+regex_merge
    processing_time_ms: Optional[int] = None
    stages_latency_ms: Dict[str, int] = Field(default_factory=dict)
    image_hash: Optional[str] = None
    trace_id: Optional[str] = None


class OcrClient(Protocol):
    """Provider interface each concrete OCR backend must implement."""

    name: str

    async def warmup(self) -> None:  # optional no-op
        ...

    async def extract(self, image_bytes: bytes, trace_id: str | None = None) -> OcrResult:
        """Run OCR on raw image bytes and return structured result."""
        ...

    async def health_check(self) -> bool:
        return True
