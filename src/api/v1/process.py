"""Process rules audit and route generation endpoints."""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.api.dependencies import get_api_key
from src.utils.analysis_metrics import process_rules_audit_requests_total

logger = logging.getLogger(__name__)
router = APIRouter()


class ProcessRulesAuditResponse(BaseModel):
    version: str = Field(description="规则版本")
    source: str = Field(description="规则文件来源")
    hash: Optional[str] = Field(default=None, description="文件内容哈希前16位")
    materials: List[str]
    complexities: Dict[str, List[str]]
    raw: Dict[str, Any]


@router.get("/process/rules/audit", response_model=ProcessRulesAuditResponse)
async def process_rules_audit(raw: bool = True, api_key: str = Depends(get_api_key)):
    from src.core.process_rules import load_rules

    path = os.getenv("PROCESS_RULES_FILE", "config/process_rules.yaml")
    rules = load_rules(force_reload=True)
    version = rules.get("__meta__", {}).get("version", "v1")
    materials = sorted([m for m in rules.keys() if not m.startswith("__")])
    complexities: Dict[str, list[str]] = {}
    for m in materials:
        cm = rules.get(m, {})
        if isinstance(cm, dict):
            complexities[m] = sorted([c for c in cm.keys() if isinstance(cm.get(c), list)])
    file_hash: str | None = None
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        file_hash = None
    try:
        resp = ProcessRulesAuditResponse(
            version=version,
            source=path if os.path.exists(path) else "embedded-defaults",
            hash=file_hash,
            materials=materials,
            complexities=complexities,
            raw=rules if raw else {},
        )
        process_rules_audit_requests_total.labels(status="ok").inc()
        return resp
    except Exception:
        process_rules_audit_requests_total.labels(status="error").inc()
        raise


# ============================================================================
# Process Route Generation Endpoints
# ============================================================================


class ProcessStepResponse(BaseModel):
    """工序步骤响应"""
    sequence: int = Field(..., description="工序序号")
    stage: str = Field(..., description="工序阶段")
    name: str = Field(..., description="工序名称")
    description: Optional[str] = Field(None, description="工序描述")
    parameters: Dict = Field(default_factory=dict, description="工艺参数")
    source: Optional[str] = Field(None, description="来源")


class ProcessRouteResponse(BaseModel):
    """工艺路线响应"""
    success: bool = Field(True, description="是否成功")
    steps: List[ProcessStepResponse] = Field(default_factory=list, description="工序列表")
    total_steps: int = Field(0, description="总工序数")
    confidence: float = Field(0.0, description="置信度")
    warnings: List[str] = Field(default_factory=list, description="警告信息")
    material: Optional[str] = Field(None, description="材料")
    drawing_type: Optional[str] = Field(None, description="图纸类型")
    error: Optional[str] = Field(None, description="错误信息")


class ProcessRouteRequest(BaseModel):
    """工艺路线请求"""
    text: str = Field(..., description="技术要求文本", min_length=1)
    material: Optional[str] = Field(None, description="材料名称")


class ProcessRequirementsInput(BaseModel):
    """工艺要求输入（结构化）"""
    heat_treatments: List[Dict] = Field(default_factory=list, description="热处理列表")
    surface_treatments: List[Dict] = Field(default_factory=list, description="表面处理列表")
    welding: List[Dict] = Field(default_factory=list, description="焊接列表")
    general_notes: List[str] = Field(default_factory=list, description="通用技术要求")


class ProcessRouteStructuredRequest(BaseModel):
    """结构化工艺路线请求"""
    process_requirements: ProcessRequirementsInput = Field(..., description="工艺要求")
    material: Optional[str] = Field(None, description="材料名称")


@router.post("/process/route/from-text", response_model=ProcessRouteResponse)
async def generate_route_from_text(request: ProcessRouteRequest) -> ProcessRouteResponse:
    """
    从技术要求文本生成工艺路线

    解析文本中的工艺信息并生成推荐的制造工艺路线。

    支持的工艺类型：
    - 热处理：淬火、回火、渗碳、渗氮、调质、正火、退火等
    - 表面处理：镀锌、镀铬、镀镍、阳极氧化、喷漆、发黑等
    - 焊接：氩弧焊、MIG/TIG焊、点焊、激光焊、钎焊等
    """
    try:
        from src.core.ocr.parsing.process_parser import parse_process_requirements
        from src.core.process import generate_process_route

        # Parse process requirements from text
        proc = parse_process_requirements(request.text)

        # Generate route with material info
        route = generate_process_route(proc, material=request.material)

        # Convert to response
        steps = [
            ProcessStepResponse(
                sequence=s.sequence,
                stage=s.stage.value,
                name=s.name,
                description=s.description,
                parameters=s.parameters,
                source=s.source,
            )
            for s in route.steps
        ]

        logger.info(
            "process_route.generated_from_text",
            extra={
                "total_steps": len(steps),
                "confidence": route.confidence,
                "has_warnings": len(route.warnings) > 0,
                "material": request.material,
            },
        )

        return ProcessRouteResponse(
            success=True,
            steps=steps,
            total_steps=len(steps),
            confidence=route.confidence,
            warnings=route.warnings,
            material=route.material,
            drawing_type=route.drawing_type,
        )

    except Exception as e:
        logger.error("process_route.generation_failed", extra={"error": str(e)})
        return ProcessRouteResponse(
            success=False,
            error=str(e),
        )


@router.post("/process/route/from-requirements", response_model=ProcessRouteResponse)
async def generate_route_from_requirements(
    request: ProcessRouteStructuredRequest,
) -> ProcessRouteResponse:
    """
    从结构化工艺要求生成工艺路线

    接受结构化的工艺要求数据，生成推荐的制造工艺路线。
    """
    try:
        from src.core.ocr.base import (
            HeatTreatmentInfo,
            HeatTreatmentType,
            ProcessRequirements,
            SurfaceTreatmentInfo,
            SurfaceTreatmentType,
            WeldingInfo,
            WeldingType,
        )
        from src.core.process import generate_process_route

        # Convert input to ProcessRequirements
        heat_treatments = []
        for ht in request.process_requirements.heat_treatments:
            try:
                ht_type = HeatTreatmentType(ht.get("type", "general_heat_treatment"))
                heat_treatments.append(HeatTreatmentInfo(
                    type=ht_type,
                    hardness=ht.get("hardness"),
                    hardness_min=ht.get("hardness_min"),
                    hardness_max=ht.get("hardness_max"),
                    hardness_unit=ht.get("hardness_unit"),
                    depth=ht.get("depth"),
                ))
            except ValueError:
                continue

        surface_treatments = []
        for st in request.process_requirements.surface_treatments:
            try:
                st_type = SurfaceTreatmentType(st.get("type", "electroplating"))
                surface_treatments.append(SurfaceTreatmentInfo(
                    type=st_type,
                    thickness=st.get("thickness"),
                ))
            except ValueError:
                continue

        welding = []
        for w in request.process_requirements.welding:
            try:
                w_type = WeldingType(w.get("type", "arc_welding"))
                welding.append(WeldingInfo(
                    type=w_type,
                    filler_material=w.get("filler_material"),
                    leg_size=w.get("leg_size"),
                ))
            except ValueError:
                continue

        proc = ProcessRequirements(
            heat_treatments=heat_treatments,
            surface_treatments=surface_treatments,
            welding=welding,
            general_notes=request.process_requirements.general_notes,
        )

        # Generate route with material info
        route = generate_process_route(proc, material=request.material)

        steps = [
            ProcessStepResponse(
                sequence=s.sequence,
                stage=s.stage.value,
                name=s.name,
                description=s.description,
                parameters=s.parameters,
                source=s.source,
            )
            for s in route.steps
        ]

        logger.info(
            "process_route.generated_from_requirements",
            extra={
                "total_steps": len(steps),
                "confidence": route.confidence,
                "material": request.material,
            },
        )

        return ProcessRouteResponse(
            success=True,
            steps=steps,
            total_steps=len(steps),
            confidence=route.confidence,
            warnings=route.warnings,
            material=route.material,
        )

    except Exception as e:
        logger.error("process_route.generation_failed", extra={"error": str(e)})
        return ProcessRouteResponse(
            success=False,
            error=str(e),
        )


@router.get("/process/treatments/heat", response_model=Dict)
async def list_heat_treatments() -> Dict:
    """列出支持的热处理类型"""
    from src.core.ocr.base import HeatTreatmentType
    from src.core.process.route_generator import HEAT_TREATMENT_NAMES

    return {
        "treatments": [
            {
                "type": ht.value,
                "name": HEAT_TREATMENT_NAMES.get(ht, ht.value),
            }
            for ht in HeatTreatmentType
        ]
    }


@router.get("/process/treatments/surface", response_model=Dict)
async def list_surface_treatments() -> Dict:
    """列出支持的表面处理类型"""
    from src.core.ocr.base import SurfaceTreatmentType
    from src.core.process.route_generator import SURFACE_TREATMENT_NAMES

    return {
        "treatments": [
            {
                "type": st.value,
                "name": SURFACE_TREATMENT_NAMES.get(st, st.value),
            }
            for st in SurfaceTreatmentType
        ]
    }


@router.get("/process/treatments/welding", response_model=Dict)
async def list_welding_types() -> Dict:
    """列出支持的焊接类型"""
    from src.core.ocr.base import WeldingType
    from src.core.process.route_generator import WELDING_NAMES

    return {
        "treatments": [
            {
                "type": wt.value,
                "name": WELDING_NAMES.get(wt, wt.value),
            }
            for wt in WeldingType
        ]
    }


__all__ = ["router"]
