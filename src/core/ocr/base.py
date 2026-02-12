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


class HeatTreatmentType(str, Enum):
    """热处理类型"""
    quenching = "quenching"  # 淬火
    tempering = "tempering"  # 回火
    annealing = "annealing"  # 退火
    normalizing = "normalizing"  # 正火
    carburizing = "carburizing"  # 渗碳
    nitriding = "nitriding"  # 渗氮
    induction_hardening = "induction_hardening"  # 感应淬火
    flame_hardening = "flame_hardening"  # 火焰淬火
    stress_relief = "stress_relief"  # 去应力退火
    aging = "aging"  # 时效处理
    quench_temper = "quench_temper"  # 调质处理
    solution_treatment = "solution_treatment"  # 固溶处理
    general_heat_treatment = "general_heat_treatment"  # 通用热处理


class SurfaceTreatmentType(str, Enum):
    """表面处理类型"""
    electroplating = "electroplating"  # 电镀
    galvanizing = "galvanizing"  # 镀锌
    chromating = "chromating"  # 镀铬
    nickel_plating = "nickel_plating"  # 镀镍
    anodizing = "anodizing"  # 阳极氧化
    phosphating = "phosphating"  # 磷化
    blackening = "blackening"  # 发黑
    painting = "painting"  # 喷漆
    powder_coating = "powder_coating"  # 粉末喷涂
    polishing = "polishing"  # 抛光
    sandblasting = "sandblasting"  # 喷砂
    passivation = "passivation"  # 钝化


class WeldingType(str, Enum):
    """焊接类型"""
    arc_welding = "arc_welding"  # 电弧焊
    mig_welding = "mig_welding"  # MIG焊
    tig_welding = "tig_welding"  # TIG焊
    spot_welding = "spot_welding"  # 点焊
    seam_welding = "seam_welding"  # 缝焊
    laser_welding = "laser_welding"  # 激光焊
    electron_beam = "electron_beam"  # 电子束焊
    brazing = "brazing"  # 钎焊
    soldering = "soldering"  # 软钎焊


class HeatTreatmentInfo(BaseModel):
    """热处理工艺信息"""
    type: HeatTreatmentType
    hardness: Optional[str] = Field(None, description="硬度要求，如 HRC58-62")
    hardness_min: Optional[float] = Field(None, description="最小硬度值")
    hardness_max: Optional[float] = Field(None, description="最大硬度值")
    hardness_unit: Optional[str] = Field(None, description="硬度单位 HRC/HB/HV")
    depth: Optional[float] = Field(None, description="处理深度 mm")
    temperature: Optional[float] = Field(None, description="处理温度 ℃")
    raw: Optional[str] = Field(None, description="原始文本")
    confidence: Optional[float] = None


class SurfaceTreatmentInfo(BaseModel):
    """表面处理工艺信息"""
    type: SurfaceTreatmentType
    thickness: Optional[float] = Field(None, description="镀层/涂层厚度 μm")
    area: Optional[str] = Field(None, description="处理区域描述")
    standard: Optional[str] = Field(None, description="执行标准")
    raw: Optional[str] = Field(None, description="原始文本")
    confidence: Optional[float] = None


class WeldingInfo(BaseModel):
    """焊接工艺信息"""
    type: WeldingType
    weld_symbol: Optional[str] = Field(None, description="焊接符号")
    leg_size: Optional[float] = Field(None, description="焊脚尺寸 mm")
    penetration: Optional[str] = Field(None, description="焊透要求")
    filler_material: Optional[str] = Field(None, description="焊材")
    raw: Optional[str] = Field(None, description="原始文本")
    confidence: Optional[float] = None


class ProcessRequirements(BaseModel):
    """工艺要求汇总"""
    heat_treatments: List[HeatTreatmentInfo] = Field(default_factory=list)
    surface_treatments: List[SurfaceTreatmentInfo] = Field(default_factory=list)
    welding: List[WeldingInfo] = Field(default_factory=list)
    general_notes: List[str] = Field(default_factory=list, description="通用技术要求")
    raw_text: Optional[str] = Field(None, description="技术要求原始文本")


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
    process_requirements: ProcessRequirements = Field(default_factory=ProcessRequirements)
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

    async def extract(self, image_bytes: bytes, trace_id: Optional[str] = None) -> OcrResult:
        """Run OCR on raw image bytes and return structured result."""
        ...

    async def health_check(self) -> bool:
        return True
