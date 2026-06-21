"""Pydantic request/response models for the materials API.

Extracted verbatim from materials.py (behavior-preserving router slimming);
re-exported by materials.py so existing references / external imports keep working.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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


class MaterialEquivalenceResponse(BaseModel):
    """材料等价表响应"""
    found: bool = Field(..., description="是否找到等价表")
    input: str = Field(..., description="输入材料")
    name: Optional[str] = Field(None, description="材料名称")
    equivalents: Dict[str, str] = Field(default_factory=dict, description="各标准体系等价牌号")


class MaterialStandardItem(BaseModel):
    """标准牌号项"""
    standard: str = Field(..., description="标准体系 (CN/US/JP/DE/UNS)")
    grade: str = Field(..., description="牌号")


class MaterialSearchItem(BaseModel):
    """材料搜索结果项"""
    grade: str = Field(..., description="牌号")
    name: str = Field(..., description="名称")
    category: str = Field(..., description="类别")
    group: str = Field(..., description="材料组")
    score: float = Field(..., description="匹配分数 (0-1)")
    match_type: str = Field(..., description="匹配类型")


class MaterialSearchResponse(BaseModel):
    """材料搜索响应"""
    query: str = Field(..., description="搜索关键词")
    total: int = Field(..., description="结果数量")
    results: List[MaterialSearchItem] = Field(..., description="搜索结果")


class PropertySearchItem(BaseModel):
    """属性搜索结果项"""
    grade: str = Field(..., description="牌号")
    name: str = Field(..., description="名称")
    category: str = Field(..., description="类别")
    group: str = Field(..., description="材料组")
    density: Optional[float] = Field(None, description="密度 g/cm³")
    tensile_strength: Optional[float] = Field(None, description="抗拉强度 MPa")
    hardness: Optional[str] = Field(None, description="硬度")
    machinability: Optional[str] = Field(None, description="可加工性")


class PropertySearchResponse(BaseModel):
    """属性搜索响应"""
    total: int = Field(..., description="结果数量")
    results: List[PropertySearchItem] = Field(..., description="搜索结果")


class ApplicationItem(BaseModel):
    """用途项"""
    code: str = Field(..., description="用途代码")
    name: str = Field(..., description="用途名称")


class ApplicationListResponse(BaseModel):
    """用途列表响应"""
    total: int = Field(..., description="用途数量")
    applications: List[ApplicationItem] = Field(..., description="用途列表")


class RecommendationPropertiesItem(BaseModel):
    """推荐材料属性"""
    density: Optional[float] = Field(None, description="密度 g/cm³")
    tensile_strength: Optional[float] = Field(None, description="抗拉强度 MPa")
    machinability: Optional[str] = Field(None, description="可加工性")


class RecommendationItem(BaseModel):
    """推荐材料项"""
    grade: str = Field(..., description="牌号")
    name: str = Field(..., description="名称")
    group: str = Field(..., description="材料组")
    score: float = Field(..., description="匹配分数")
    reason: str = Field(..., description="推荐理由")
    properties: RecommendationPropertiesItem = Field(..., description="属性")


class RecommendationResponse(BaseModel):
    """推荐响应"""
    application: str = Field(..., description="用途")
    total: int = Field(..., description="推荐数量")
    recommendations: List[RecommendationItem] = Field(..., description="推荐列表")


class AlternativeItem(BaseModel):
    """替代材料项"""
    grade: str = Field(..., description="牌号")
    name: str = Field(..., description="名称")
    group: str = Field(..., description="材料组")
    reason: str = Field(..., description="替代理由")
    cost_factor: float = Field(..., description="成本系数 (1.0=相同)")
    source: str = Field(..., description="来源 (predefined/auto)")


class AlternativeResponse(BaseModel):
    """替代材料响应"""
    original_grade: str = Field(..., description="原材料牌号")
    preference: str = Field(..., description="替代偏好")
    total: int = Field(..., description="替代数量")
    alternatives: List[AlternativeItem] = Field(..., description="替代列表")


class CostTierItem(BaseModel):
    """成本等级项"""
    tier: int = Field(..., description="等级 (1-5)")
    name: str = Field(..., description="等级名称")
    description: str = Field(..., description="等级描述")
    price_range: str = Field(..., description="价格区间")


class CostTierListResponse(BaseModel):
    """成本等级列表响应"""
    tiers: List[CostTierItem] = Field(..., description="成本等级列表")


class MaterialCostItem(BaseModel):
    """材料成本项"""
    grade: str = Field(..., description="牌号")
    name: str = Field(..., description="名称")
    tier: int = Field(..., description="成本等级 (1-5)")
    tier_name: str = Field(..., description="等级名称")
    tier_description: str = Field(..., description="等级描述")
    cost_index: float = Field(..., description="相对成本指数 (Q235B=1.0)")
    price_range: str = Field(..., description="价格区间")
    group: str = Field(..., description="材料组")


class CostCompareItem(BaseModel):
    """成本比较项"""
    grade: str = Field(..., description="牌号")
    name: str = Field(..., description="名称")
    tier: int = Field(..., description="成本等级")
    tier_name: str = Field(..., description="等级名称")
    cost_index: float = Field(..., description="相对成本指数")
    group: str = Field(..., description="材料组")
    relative_to_cheapest: float = Field(..., description="相对最便宜材料的倍数")


class CostCompareResponse(BaseModel):
    """成本比较响应"""
    total: int = Field(..., description="比较数量")
    comparison: List[CostCompareItem] = Field(..., description="比较结果")
    missing: List[str] = Field(default_factory=list, description="未命中的材料牌号")


class CostSearchItem(BaseModel):
    """成本搜索结果项"""
    grade: str = Field(..., description="牌号")
    name: str = Field(..., description="名称")
    category: str = Field(..., description="类别")
    group: str = Field(..., description="材料组")
    tier: int = Field(..., description="成本等级")
    tier_name: str = Field(..., description="等级名称")
    cost_index: float = Field(..., description="相对成本指数")


class CostSearchResponse(BaseModel):
    """成本搜索响应"""
    total: int = Field(..., description="结果数量")
    results: List[CostSearchItem] = Field(..., description="搜索结果")


class MaterialRef(BaseModel):
    """材料引用"""
    grade: str = Field(..., description="牌号")
    name: str = Field(..., description="名称")
    group: Optional[str] = Field(None, description="材料组")


class WeldCompatibilityResponse(BaseModel):
    """焊接兼容性响应"""
    compatible: bool = Field(..., description="是否兼容")
    rating: str = Field(..., description="兼容等级")
    rating_cn: str = Field(..., description="兼容等级(中文)")
    method: str = Field(..., description="推荐焊接方法")
    notes: str = Field(..., description="注意事项")
    material1: MaterialRef = Field(..., description="材料1")
    material2: MaterialRef = Field(..., description="材料2")


class GalvanicAnodeCathode(BaseModel):
    """阳极/阴极信息"""
    grade: str = Field(..., description="牌号")
    name: str = Field(..., description="名称")
    role: str = Field(..., description="角色")


class GalvanicCorrosionResponse(BaseModel):
    """电偶腐蚀响应"""
    risk: str = Field(..., description="风险等级")
    risk_cn: str = Field(..., description="风险等级(中文)")
    potential_difference: Optional[float] = Field(None, description="电位差(V)")
    recommendation: Optional[str] = Field(None, description="建议")
    anode: Optional[GalvanicAnodeCathode] = Field(None, description="阳极")
    cathode: Optional[GalvanicAnodeCathode] = Field(None, description="阴极")
    material1: MaterialRef = Field(..., description="材料1")
    material2: MaterialRef = Field(..., description="材料2")


class HeatTreatmentCompatibilityResponse(BaseModel):
    """热处理兼容性响应"""
    compatible: bool = Field(..., description="是否兼容")
    status: str = Field(..., description="状态")
    status_cn: str = Field(..., description="状态(中文)")
    grade: str = Field(..., description="牌号")
    name: str = Field(..., description="名称")
    treatment: str = Field(..., description="热处理工艺")
    reason: str = Field(..., description="原因")
    recommended_treatments: List[str] = Field(default_factory=list, description="推荐热处理")
    forbidden_treatments: List[str] = Field(default_factory=list, description="禁止热处理")


class FullCompatibilityResponse(BaseModel):
    """完整兼容性响应"""
    overall: str = Field(..., description="总体评估")
    overall_cn: str = Field(..., description="总体评估(中文)")
    issues: List[str] = Field(default_factory=list, description="问题列表")
    recommendations: List[str] = Field(default_factory=list, description="建议列表")
    weld_compatibility: Dict[str, Any] = Field(..., description="焊接兼容性")
    galvanic_corrosion: Dict[str, Any] = Field(..., description="电偶腐蚀")
