"""Material query and classification API endpoints."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# Response Models
# ============================================================================



from src.api.v1.materials_models import (
    MaterialPropertiesResponse,
    ProcessRecommendationResponse,
    MaterialInfoResponse,
    MaterialListItem,
    MaterialListResponse,
    MaterialClassifyRequest,
    MaterialClassifyResponse,
    MaterialGroupsResponse,
    MaterialEquivalenceResponse,
    MaterialStandardItem,
    MaterialSearchItem,
    MaterialSearchResponse,
    PropertySearchItem,
    PropertySearchResponse,
    ApplicationItem,
    ApplicationListResponse,
    RecommendationPropertiesItem,
    RecommendationItem,
    RecommendationResponse,
    AlternativeItem,
    AlternativeResponse,
    CostTierItem,
    CostTierListResponse,
    MaterialCostItem,
    CostCompareItem,
    CostCompareResponse,
    CostSearchItem,
    CostSearchResponse,
    MaterialRef,
    WeldCompatibilityResponse,
    GalvanicAnodeCathode,
    GalvanicCorrosionResponse,
    HeatTreatmentCompatibilityResponse,
    FullCompatibilityResponse,
)



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


@router.get("/search", response_model=MaterialSearchResponse)
async def search_materials_api(
    q: str = Query(..., description="搜索关键词（支持中文、英文、拼音）", min_length=1),
    category: Optional[str] = Query(None, description="限定类别: metal, non_metal, composite"),
    group: Optional[str] = Query(None, description="限定材料组"),
    limit: int = Query(10, description="返回数量限制", ge=1, le=50),
) -> MaterialSearchResponse:
    """
    搜索材料（支持模糊搜索和拼音）

    支持的搜索方式：
    - 精确牌号: S30408, Q235B, 304
    - 中文名称: 不锈钢, 黄铜, 铝合金
    - 拼音: buxiugang, huangtong, lvhejin
    - 拼音缩写: bxg (不锈钢), lhj (铝合金)
    - 模糊匹配: 304 → S30408, 钛 → TA2/TC4
    """
    from src.core.materials import search_materials

    results = search_materials(
        query=q,
        limit=limit,
        category=category,
        group=group,
    )

    return MaterialSearchResponse(
        query=q,
        total=len(results),
        results=[MaterialSearchItem(**r) for r in results],
    )


@router.get("/search/properties", response_model=PropertySearchResponse)
async def search_by_properties_api(
    density_min: Optional[float] = Query(None, description="最小密度 g/cm³"),
    density_max: Optional[float] = Query(None, description="最大密度 g/cm³"),
    strength_min: Optional[float] = Query(None, description="最小抗拉强度 MPa"),
    strength_max: Optional[float] = Query(None, description="最大抗拉强度 MPa"),
    hardness: Optional[str] = Query(None, description="硬度类型 (HRC, HB)"),
    machinability: Optional[str] = Query(None, description="可加工性: excellent, good, fair, poor"),
    category: Optional[str] = Query(None, description="限定类别"),
    limit: int = Query(20, description="返回数量限制", ge=1, le=100),
) -> PropertySearchResponse:
    """
    按属性搜索材料

    支持的筛选条件：
    - 密度范围: density_min ~ density_max (g/cm³)
    - 强度范围: strength_min ~ strength_max (MPa)
    - 硬度类型: HRC, HB 等
    - 可加工性: excellent, good, fair, poor
    """
    from src.core.materials import search_by_properties

    density_range = None
    if density_min is not None or density_max is not None:
        density_range = (density_min or 0, density_max or 100)

    strength_range = None
    if strength_min is not None or strength_max is not None:
        strength_range = (strength_min or 0, strength_max or 10000)

    results = search_by_properties(
        density_range=density_range,
        tensile_strength_range=strength_range,
        hardness_contains=hardness,
        machinability=machinability,
        category=category,
        limit=limit,
    )

    return PropertySearchResponse(
        total=len(results),
        results=[PropertySearchItem(**r) for r in results],
    )


@router.get("/cost/tiers", response_model=CostTierListResponse)
async def list_cost_tiers() -> CostTierListResponse:
    """
    列出成本等级定义

    返回 1-5 级成本等级的名称、描述和价格区间
    """
    from src.core.materials import get_cost_tier_info

    tiers = get_cost_tier_info()
    return CostTierListResponse(
        tiers=[CostTierItem(**t) for t in tiers],
    )


@router.get("/cost/search", response_model=CostSearchResponse)
async def search_by_cost_api(
    max_tier: Optional[int] = Query(None, description="最大成本等级 (1-5)", ge=1, le=5),
    max_cost_index: Optional[float] = Query(None, description="最大成本指数"),
    category: Optional[str] = Query(None, description="限定类别"),
    group: Optional[str] = Query(None, description="限定材料组"),
    include_estimated: bool = Query(False, description="是否包含按材料组估算的成本"),
    limit: int = Query(20, description="返回数量限制", ge=1, le=100),
) -> CostSearchResponse:
    """
    按成本筛选材料

    筛选条件：
    - max_tier: 最大成本等级 (1=经济型, 5=特种型)
    - max_cost_index: 最大成本指数 (Q235B=1.0)
    - category: 材料类别
    - group: 材料组
    - include_estimated: 是否包含按材料组估算的成本
    """
    from src.core.materials import search_by_cost

    results = search_by_cost(
        max_tier=max_tier,
        max_cost_index=max_cost_index,
        category=category,
        group=group,
        include_estimated=include_estimated,
        limit=limit,
    )

    return CostSearchResponse(
        total=len(results),
        results=[CostSearchItem(**r) for r in results],
    )


@router.post("/cost/compare", response_model=CostCompareResponse)
async def compare_costs(
    grades: List[str] = Query(..., description="要比较的材料牌号列表"),
) -> CostCompareResponse:
    """
    比较多个材料的成本

    输入材料牌号列表，返回按成本排序的比较结果
    未命中的材料会返回到 missing 列表
    """
    from src.core.materials import compare_material_costs

    results, missing = compare_material_costs(grades, include_missing=True)

    return CostCompareResponse(
        total=len(results),
        comparison=[CostCompareItem(**r) for r in results],
        missing=missing,
    )


@router.get("/{grade}/cost", response_model=MaterialCostItem)
async def get_material_cost_api(grade: str) -> MaterialCostItem:
    """
    获取材料成本信息

    返回材料的成本等级、成本指数和价格区间
    """
    from src.core.materials import get_material_cost
    from fastapi import HTTPException

    result = get_material_cost(grade)
    if not result:
        raise HTTPException(status_code=404, detail=f"Material '{grade}' not found")

    return MaterialCostItem(**result)


@router.get("/applications", response_model=ApplicationListResponse)
async def list_material_applications() -> ApplicationListResponse:
    """
    列出所有支持的材料用途

    返回用于材料推荐的用途代码列表
    """
    from src.core.materials import list_applications

    apps = list_applications()
    return ApplicationListResponse(
        total=len(apps),
        applications=[ApplicationItem(**a) for a in apps],
    )


@router.get("/recommend/{application}", response_model=RecommendationResponse)
async def recommend_materials(
    application: str,
    min_strength: Optional[float] = Query(None, description="最小抗拉强度 MPa"),
    max_density: Optional[float] = Query(None, description="最大密度 g/cm³"),
    machinability: Optional[str] = Query(None, description="可加工性要求: excellent, good, fair"),
    limit: int = Query(5, description="返回数量限制", ge=1, le=20),
) -> RecommendationResponse:
    """
    根据用途推荐材料

    支持的用途代码：
    - structural: 结构件
    - load_bearing: 承载件
    - corrosion_resistant: 耐腐蚀件
    - seawater: 海水环境
    - chemical: 化工环境
    - wear_resistant: 耐磨件
    - bearing: 轴承/轴瓦
    - electrical: 导电件
    - thermal: 导热件
    - spring: 弹簧/弹性件
    - lightweight: 轻量化
    - high_temperature: 高温环境
    - food_grade: 食品级
    - medical: 医疗器械
    - precision: 精密零件
    """
    from src.core.materials import get_material_recommendations

    requirements: Dict[str, Any] = {}
    if min_strength:
        requirements["min_strength"] = min_strength
    if max_density:
        requirements["max_density"] = max_density
    if machinability:
        requirements["machinability"] = machinability

    results = get_material_recommendations(
        application=application,
        requirements=requirements,
        limit=limit,
    )

    recommendations = []
    for r in results:
        recommendations.append(RecommendationItem(
            grade=r["grade"],
            name=r["name"],
            group=r["group"],
            score=r["score"],
            reason=r["reason"],
            properties=RecommendationPropertiesItem(**r["properties"]),
        ))

    return RecommendationResponse(
        application=application,
        total=len(recommendations),
        recommendations=recommendations,
    )


@router.get("/{grade}/alternatives", response_model=AlternativeResponse)
async def get_material_alternatives(
    grade: str,
    preference: str = Query("similar", description="替代偏好: similar, cheaper, better"),
) -> AlternativeResponse:
    """
    获取材料的替代建议

    替代偏好：
    - similar: 性能相近的替代（默认）
    - cheaper: 成本更低的替代
    - better: 性能更好的替代
    """
    from src.core.materials import get_alternative_materials

    results = get_alternative_materials(grade=grade, preference=preference)

    return AlternativeResponse(
        original_grade=grade,
        preference=preference,
        total=len(results),
        alternatives=[AlternativeItem(**r) for r in results],
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

    seen_groups: Dict[str, set[str]] = {k: set() for k in groups}

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
            aliases=[],
            category=None,
            sub_category=None,
            group=None,
            standards=[],
            description="",
            properties=None,
            process=None,
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
            grade=None,
            name=None,
            category=None,
            group=None,
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
                grade=None,
                name=None,
                category=None,
                group=None,
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


@router.get("/{grade}/equivalents", response_model=MaterialEquivalenceResponse)
async def get_material_equivalents(grade: str) -> MaterialEquivalenceResponse:
    """
    获取材料等价表

    查询材料在不同标准体系中的等价牌号：
    - CN: 中国 GB 标准
    - US: 美国 ASTM/AISI 标准
    - JP: 日本 JIS 标准
    - DE: 德国 DIN/EN 标准
    - UNS: 统一编号系统（部分材料）
    """
    from src.core.materials import get_material_equivalence

    equiv = get_material_equivalence(grade)

    if not equiv:
        return MaterialEquivalenceResponse(
            found=False,
            input=grade,
            name=None,
            equivalents={},
        )

    # Extract name and remove from equivalents dict
    name = equiv.get("name")
    equivalents = {k: v for k, v in equiv.items() if k != "name"}

    return MaterialEquivalenceResponse(
        found=True,
        input=grade,
        name=name,
        equivalents=equivalents,
    )


@router.get("/{grade}/convert/{target_standard}")
async def convert_material_standard(
    grade: str,
    target_standard: str,
) -> Dict[str, Any]:
    """
    转换材料牌号到目标标准

    将材料牌号转换为指定标准体系的等价牌号

    Args:
        grade: 材料牌号（任意标准）
        target_standard: 目标标准体系 (CN/US/JP/DE/UNS)

    Returns:
        转换结果
    """
    from src.core.materials import find_equivalent_material, get_material_equivalence

    equiv = get_material_equivalence(grade)
    if not equiv:
        return {
            "success": False,
            "input": grade,
            "target_standard": target_standard,
            "error": "Material not found in equivalence table",
        }

    target_grade = find_equivalent_material(grade, target_standard)
    if not target_grade:
        return {
            "success": False,
            "input": grade,
            "target_standard": target_standard,
            "error": f"No equivalent found for standard '{target_standard}'",
        }

    return {
        "success": True,
        "input": grade,
        "target_standard": target_standard,
        "result": target_grade,
        "name": equiv.get("name"),
    }


@router.get("/compatibility/weld", response_model=WeldCompatibilityResponse)
async def check_weld_compatibility_api(
    material1: str = Query(..., description="材料1牌号"),
    material2: str = Query(..., description="材料2牌号"),
) -> WeldCompatibilityResponse:
    """
    检查两种材料的焊接兼容性

    评估两种材料进行焊接的兼容性，返回：
    - 兼容等级: excellent/good/fair/poor/incompatible/not_recommended/unknown
    - 推荐焊接方法
    - 注意事项
    """
    from src.core.materials import check_weld_compatibility, classify_material_detailed

    result = check_weld_compatibility(material1, material2)

    # Get material info for references
    info1 = classify_material_detailed(material1)
    info2 = classify_material_detailed(material2)

    # Handle unknown material case
    rating = result.get("rating", "unknown")
    rating_cn = result.get("rating_cn", "未知")
    method = result.get("method", "-")
    notes = result.get("notes", result.get("error", ""))

    return WeldCompatibilityResponse(
        compatible=result["compatible"],
        rating=rating,
        rating_cn=rating_cn,
        method=method,
        notes=notes,
        material1=MaterialRef(
            grade=material1,
            name=info1.name if info1 else material1,
            group=info1.group.value if info1 else None,
        ),
        material2=MaterialRef(
            grade=material2,
            name=info2.name if info2 else material2,
            group=info2.group.value if info2 else None,
        ),
    )


@router.get("/compatibility/galvanic", response_model=GalvanicCorrosionResponse)
async def check_galvanic_corrosion_api(
    material1: str = Query(..., description="材料1牌号"),
    material2: str = Query(..., description="材料2牌号"),
) -> GalvanicCorrosionResponse:
    """
    检查两种材料的电偶腐蚀风险

    评估两种材料接触时的电化学腐蚀风险，返回：
    - 风险等级: safe/low/moderate/high/severe/unknown/none
    - 电位差
    - 阳极/阴极识别
    - 防护建议
    """
    from src.core.materials import check_galvanic_corrosion, classify_material_detailed

    result = check_galvanic_corrosion(material1, material2)

    # Get material info for references
    info1 = classify_material_detailed(material1)
    info2 = classify_material_detailed(material2)

    # Handle unknown material case
    risk_cn = result.get("risk_cn", "未知")

    anode = None
    cathode = None
    if result.get("anode"):
        anode = GalvanicAnodeCathode(
            grade=result["anode"]["grade"],
            name=result["anode"]["name"],
            role="anode",
        )
    if result.get("cathode"):
        cathode = GalvanicAnodeCathode(
            grade=result["cathode"]["grade"],
            name=result["cathode"]["name"],
            role="cathode",
        )

    return GalvanicCorrosionResponse(
        risk=result["risk"],
        risk_cn=risk_cn,
        potential_difference=result.get("potential_difference"),
        recommendation=result.get("recommendation"),
        anode=anode,
        cathode=cathode,
        material1=MaterialRef(
            grade=material1,
            name=info1.name if info1 else material1,
            group=info1.group.value if info1 else None,
        ),
        material2=MaterialRef(
            grade=material2,
            name=info2.name if info2 else material2,
            group=info2.group.value if info2 else None,
        ),
    )


@router.get("/compatibility/heat-treatment", response_model=HeatTreatmentCompatibilityResponse)
async def check_heat_treatment_compatibility_api(
    grade: str = Query(..., description="材料牌号"),
    treatment: str = Query(..., description="热处理工艺"),
) -> HeatTreatmentCompatibilityResponse:
    """
    检查材料与热处理工艺的兼容性

    评估指定材料是否适合某种热处理工艺，返回：
    - 兼容性状态: recommended/allowed/not_recommended/forbidden/unknown
    - 原因说明
    - 推荐/禁止的热处理列表
    """
    from src.core.materials import check_heat_treatment_compatibility, classify_material_detailed

    result = check_heat_treatment_compatibility(grade, treatment)

    # Get material info
    info = classify_material_detailed(grade)

    # Handle unknown material case
    status = result.get("status", "unknown")
    status_cn = result.get("status_cn", "未知")
    reason = result.get("reason", result.get("error", ""))

    return HeatTreatmentCompatibilityResponse(
        compatible=result["compatible"],
        status=status,
        status_cn=status_cn,
        grade=grade,
        name=info.name if info else grade,
        treatment=treatment,
        reason=reason,
        recommended_treatments=result.get("recommended_treatments", []),
        forbidden_treatments=result.get("forbidden_treatments", []),
    )


@router.get("/compatibility/full", response_model=FullCompatibilityResponse)
async def check_full_compatibility_api(
    material1: str = Query(..., description="材料1牌号"),
    material2: str = Query(..., description="材料2牌号"),
) -> FullCompatibilityResponse:
    """
    完整兼容性检查

    对两种材料进行全面兼容性分析，包括：
    - 焊接兼容性
    - 电偶腐蚀风险
    - 综合评估和建议
    """
    from src.core.materials import check_full_compatibility

    result = check_full_compatibility(material1, material2)

    return FullCompatibilityResponse(
        overall=result["overall"],
        overall_cn=result["overall_cn"],
        issues=result.get("issues", []),
        recommendations=result.get("recommendations", []),
        weld_compatibility=result["weld_compatibility"],
        galvanic_corrosion=result["galvanic_corrosion"],
    )


@router.get("/export/csv")
async def export_materials_to_csv() -> Response:
    """
    导出材料数据库为 CSV

    返回完整的材料数据库 CSV 文件，包含：
    - 材料基本信息（牌号、名称、别名）
    - 分类信息（类别、子类、材料组）
    - 物理属性（密度、抗拉强度、屈服强度、硬度）
    - 加工属性（可加工性、可焊性）
    - 工艺建议（热处理、表面处理、切削参数）
    """
    from src.core.materials import export_materials_csv

    csv_content = export_materials_csv()

    return Response(
        content=csv_content,
        media_type="text/csv; charset=utf-8-sig",
        headers={
            "Content-Disposition": "attachment; filename=materials.csv"
        }
    )


@router.get("/export/equivalence-csv")
async def export_equivalence_to_csv() -> Response:
    """
    导出材料等价表为 CSV

    返回材料等价表 CSV 文件，包含：
    - 牌号和名称
    - 各标准体系等价牌号（CN/US/JP/DE/UNS）
    """
    from src.core.materials import export_equivalence_csv

    csv_content = export_equivalence_csv()

    return Response(
        content=csv_content,
        media_type="text/csv; charset=utf-8-sig",
        headers={
            "Content-Disposition": "attachment; filename=material_equivalence.csv"
        }
    )


__all__ = ["router"]
