"""Material query and classification API endpoints."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# Response Models
# ============================================================================


class MaterialPropertiesResponse(BaseModel):
    """材料物理和机械属性"""
    density: Optional[float] = Field(None, description="密度 g/cm³")
    melting_point: Optional[float] = Field(None, description="熔点 ℃")
    thermal_conductivity: Optional[float] = Field(None, description="导热系数 W/(m·K)")
    tensile_strength: Optional[float] = Field(None, description="抗拉强度 MPa")
    yield_strength: Optional[float] = Field(None, description="屈服强度 MPa")
    hardness: Optional[str] = Field(None, description="硬度")
    elongation: Optional[float] = Field(None, description="延伸率 %")
    machinability: Optional[str] = Field(None, description="可加工性")
    weldability: Optional[str] = Field(None, description="可焊性")


class ProcessRecommendationResponse(BaseModel):
    """工艺推荐"""
    blank_forms: List[str] = Field(default_factory=list, description="毛坯形式")
    blank_hint: str = Field("", description="毛坯描述")
    heat_treatments: List[str] = Field(default_factory=list, description="推荐热处理")
    forbidden_heat_treatments: List[str] = Field(default_factory=list, description="禁止热处理")
    surface_treatments: List[str] = Field(default_factory=list, description="推荐表面处理")
    forbidden_surface_treatments: List[str] = Field(default_factory=list, description="禁止表面处理")
    special_tooling: bool = Field(False, description="是否需要特殊刀具")
    coolant_required: bool = Field(True, description="是否需要冷却液")
    warnings: List[str] = Field(default_factory=list, description="警告")
    recommendations: List[str] = Field(default_factory=list, description="建议")


class MaterialInfoResponse(BaseModel):
    """材料完整信息"""
    found: bool = Field(..., description="是否找到材料")
    grade: str = Field(..., description="牌号")
    name: str = Field(..., description="名称")
    aliases: List[str] = Field(default_factory=list, description="别名")
    category: Optional[str] = Field(None, description="大类")
    sub_category: Optional[str] = Field(None, description="子类")
    group: Optional[str] = Field(None, description="材料组")
    standards: List[str] = Field(default_factory=list, description="相关标准")
    description: str = Field("", description="描述")
    properties: Optional[MaterialPropertiesResponse] = Field(None, description="属性")
    process: Optional[ProcessRecommendationResponse] = Field(None, description="工艺建议")


class MaterialListItem(BaseModel):
    """材料列表项"""
    grade: str = Field(..., description="牌号")
    name: str = Field(..., description="名称")
    category: str = Field(..., description="大类")
    group: str = Field(..., description="材料组")


class MaterialListResponse(BaseModel):
    """材料列表响应"""
    total: int = Field(..., description="总数")
    materials: List[MaterialListItem] = Field(..., description="材料列表")


class MaterialClassifyRequest(BaseModel):
    """材料分类请求"""
    material: str = Field(..., description="材料名称", min_length=1)


class MaterialClassifyResponse(BaseModel):
    """材料分类响应"""
    input: str = Field(..., description="输入材料名称")
    found: bool = Field(..., description="是否找到")
    grade: Optional[str] = Field(None, description="匹配的标准牌号")
    name: Optional[str] = Field(None, description="材料名称")
    category: Optional[str] = Field(None, description="大类")
    group: Optional[str] = Field(None, description="材料组")


class MaterialGroupsResponse(BaseModel):
    """材料组列表响应"""
    groups: Dict[str, List[str]] = Field(..., description="按类别分组的材料组")


# ============================================================================
# API Endpoints
# ============================================================================


@router.get("", response_model=MaterialListResponse)
async def list_materials(
    category: Optional[str] = Query(None, description="按类别筛选: metal, non_metal, composite"),
    group: Optional[str] = Query(None, description="按材料组筛选"),
) -> MaterialListResponse:
    """
    列出所有材料

    可选筛选条件：
    - category: 材料类别 (metal, non_metal, composite)
    - group: 材料组 (carbon_steel, stainless_steel, aluminum, etc.)
    """
    from src.core.materials import MATERIAL_DATABASE

    materials = []
    for grade, info in MATERIAL_DATABASE.items():
        # Apply filters
        if category and info.category.value != category:
            continue
        if group and info.group.value != group:
            continue

        materials.append(MaterialListItem(
            grade=grade,
            name=info.name,
            category=info.category.value,
            group=info.group.value,
        ))

    # Sort by grade
    materials.sort(key=lambda m: m.grade)

    return MaterialListResponse(
        total=len(materials),
        materials=materials,
    )


@router.get("/groups", response_model=MaterialGroupsResponse)
async def list_material_groups() -> MaterialGroupsResponse:
    """
    列出所有材料组

    返回按类别分组的材料组列表
    """
    from src.core.materials import MATERIAL_DATABASE
    from src.core.materials.classifier import MaterialCategory

    groups: Dict[str, List[str]] = {
        "metal": [],
        "non_metal": [],
        "composite": [],
    }

    seen_groups: Dict[str, set] = {k: set() for k in groups}

    for info in MATERIAL_DATABASE.values():
        cat = info.category.value
        grp = info.group.value
        if grp not in seen_groups[cat]:
            seen_groups[cat].add(grp)
            groups[cat].append(grp)

    # Sort each list
    for cat in groups:
        groups[cat].sort()

    return MaterialGroupsResponse(groups=groups)


@router.get("/{grade}", response_model=MaterialInfoResponse)
async def get_material(grade: str) -> MaterialInfoResponse:
    """
    获取材料详细信息

    通过牌号或别名查询材料的完整信息，包括：
    - 分类信息
    - 物理/机械属性
    - 工艺建议
    """
    from src.core.materials import classify_material_detailed

    info = classify_material_detailed(grade)

    if not info:
        return MaterialInfoResponse(
            found=False,
            grade=grade,
            name=grade,
        )

    return MaterialInfoResponse(
        found=True,
        grade=info.grade,
        name=info.name,
        aliases=info.aliases,
        category=info.category.value,
        sub_category=info.sub_category.value,
        group=info.group.value,
        standards=info.standards,
        description=info.description,
        properties=MaterialPropertiesResponse(
            density=info.properties.density,
            melting_point=info.properties.melting_point,
            thermal_conductivity=info.properties.thermal_conductivity,
            tensile_strength=info.properties.tensile_strength,
            yield_strength=info.properties.yield_strength,
            hardness=info.properties.hardness,
            elongation=info.properties.elongation,
            machinability=info.properties.machinability,
            weldability=info.properties.weldability,
        ),
        process=ProcessRecommendationResponse(
            blank_forms=info.process.blank_forms,
            blank_hint=info.process.blank_hint,
            heat_treatments=info.process.heat_treatments,
            forbidden_heat_treatments=info.process.forbidden_heat_treatments,
            surface_treatments=info.process.surface_treatments,
            forbidden_surface_treatments=info.process.forbidden_surface_treatments,
            special_tooling=info.process.special_tooling,
            coolant_required=info.process.coolant_required,
            warnings=info.process.warnings,
            recommendations=info.process.recommendations,
        ),
    )


@router.post("/classify", response_model=MaterialClassifyResponse)
async def classify_material(request: MaterialClassifyRequest) -> MaterialClassifyResponse:
    """
    分类材料

    根据输入的材料名称进行分类，支持：
    - 精确匹配：S30408, Q235B
    - 别名匹配：304, SUS304
    - 模式匹配：C-22, 45#, 6061-T6
    """
    from src.core.materials import classify_material_detailed

    info = classify_material_detailed(request.material)

    if not info:
        logger.info(
            "material.classify.not_found",
            extra={"input": request.material},
        )
        return MaterialClassifyResponse(
            input=request.material,
            found=False,
        )

    logger.info(
        "material.classify.found",
        extra={
            "input": request.material,
            "grade": info.grade,
            "group": info.group.value,
        },
    )

    return MaterialClassifyResponse(
        input=request.material,
        found=True,
        grade=info.grade,
        name=info.name,
        category=info.category.value,
        group=info.group.value,
    )


@router.post("/batch-classify", response_model=List[MaterialClassifyResponse])
async def batch_classify_materials(
    materials: List[str] = Query(..., description="材料名称列表"),
) -> List[MaterialClassifyResponse]:
    """
    批量分类材料

    一次性分类多个材料名称
    """
    from src.core.materials import classify_material_detailed

    results = []
    for mat in materials:
        info = classify_material_detailed(mat)
        if info:
            results.append(MaterialClassifyResponse(
                input=mat,
                found=True,
                grade=info.grade,
                name=info.name,
                category=info.category.value,
                group=info.group.value,
            ))
        else:
            results.append(MaterialClassifyResponse(
                input=mat,
                found=False,
            ))

    return results


@router.get("/{grade}/process", response_model=ProcessRecommendationResponse)
async def get_material_process(grade: str) -> ProcessRecommendationResponse:
    """
    获取材料工艺建议

    返回指定材料的工艺推荐，包括：
    - 毛坯形式和描述
    - 推荐/禁止的热处理
    - 推荐/禁止的表面处理
    - 加工参数建议
    - 警告和注意事项
    """
    from src.core.materials import classify_material_detailed

    info = classify_material_detailed(grade)

    if not info:
        raise HTTPException(status_code=404, detail=f"Material '{grade}' not found")

    return ProcessRecommendationResponse(
        blank_forms=info.process.blank_forms,
        blank_hint=info.process.blank_hint,
        heat_treatments=info.process.heat_treatments,
        forbidden_heat_treatments=info.process.forbidden_heat_treatments,
        surface_treatments=info.process.surface_treatments,
        forbidden_surface_treatments=info.process.forbidden_surface_treatments,
        special_tooling=info.process.special_tooling,
        coolant_required=info.process.coolant_required,
        warnings=info.process.warnings,
        recommendations=info.process.recommendations,
    )


__all__ = ["router"]
