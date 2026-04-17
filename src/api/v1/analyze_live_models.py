from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class AnalysisOptions(BaseModel):
    """分析选项"""

    extract_features: bool = Field(default=True, description="是否提取特征")
    classify_parts: bool = Field(default=True, description="是否分类零件")
    calculate_similarity: bool = Field(default=False, description="是否计算相似度")
    reference_id: Optional[str] = Field(default=None, description="参考文件ID")
    quality_check: bool = Field(default=True, description="是否质量检查")
    process_recommendation: bool = Field(default=False, description="是否推荐工艺")
    estimate_cost: bool = Field(default=False, description="是否估算成本 (L4)")
    enable_ocr: bool = Field(
        default=False, description="是否启用OCR解析 (默认关闭保障向后兼容)"
    )
    ocr_provider: str = Field(
        default="auto", description="OCR provider策略 auto|paddle|deepseek_hf"
    )
    history_file_path: Optional[str] = Field(
        default=None,
        description="历史命令序列文件路径（可选，.h5）",
    )


class AnalysisResult(BaseModel):
    """分析结果"""

    id: str = Field(description="分析ID")
    timestamp: datetime = Field(description="分析时间")
    file_name: str = Field(description="文件名")
    file_format: str = Field(description="文件格式")
    results: Dict[str, Any] = Field(description="分析结果")
    processing_time: float = Field(description="处理时间(秒)")
    cache_hit: bool = Field(default=False, description="是否缓存命中")
    cad_document: Optional[Dict[str, Any]] = Field(
        default=None,
        description="统一的CAD文档结构 (序列化) 包含实体/图层/边界框/复杂度等, 便于下游直接使用。",
    )
    feature_version: str = Field(
        default="v1", description="特征版本 (用于兼容后续维度或语义扩展)"
    )


class BatchClassifyResultItem(BaseModel):
    """批量分类单个结果"""

    file_name: str = Field(description="文件名")
    category: Optional[str] = Field(default=None, description="分类类别")
    fine_category: Optional[str] = Field(default=None, description="细粒度分类类别")
    coarse_category: Optional[str] = Field(default=None, description="归一化粗粒度分类类别")
    is_coarse_label: Optional[bool] = Field(
        default=None, description="分类类别是否已经是粗粒度标签"
    )
    confidence: Optional[float] = Field(default=None, description="置信度")
    probabilities: Optional[Dict[str, float]] = Field(
        default=None, description="各类别概率"
    )
    needs_review: bool = Field(default=False, description="是否需要人工复核")
    review_reason: Optional[str] = Field(default=None, description="复核原因")
    top2_category: Optional[str] = Field(default=None, description="第二候选类别")
    top2_confidence: Optional[float] = Field(default=None, description="第二候选置信度")
    classifier: Optional[str] = Field(default=None, description="使用的分类器版本")
    error: Optional[str] = Field(default=None, description="错误信息")


class BatchClassifyResponse(BaseModel):
    """批量分类响应"""

    total: int = Field(description="总文件数")
    success: int = Field(description="成功分类数")
    failed: int = Field(description="失败数")
    processing_time: float = Field(description="处理时间(秒)")
    results: list[BatchClassifyResultItem] = Field(description="分类结果列表")


class SimilarityQuery(BaseModel):
    reference_id: str = Field(description="参考分析ID")
    target_id: str = Field(description="目标分析ID")


class SimilarityResult(BaseModel):
    reference_id: str
    target_id: str
    score: float
    method: str
    dimension: int
    reference_part_type: Optional[str] = None
    reference_fine_part_type: Optional[str] = None
    reference_coarse_part_type: Optional[str] = None
    reference_decision_source: Optional[str] = None
    reference_is_coarse_label: Optional[bool] = None
    target_part_type: Optional[str] = None
    target_fine_part_type: Optional[str] = None
    target_coarse_part_type: Optional[str] = None
    target_decision_source: Optional[str] = None
    target_is_coarse_label: Optional[bool] = None
    status: Optional[str] = None
    error: Optional[Dict[str, Any]] = None


class SimilarityTopKQuery(BaseModel):
    target_id: str = Field(description="用于检索相似向量的分析ID")
    k: int = Field(default=5, description="返回的最大数量 (包含自身)")
    exclude_self: bool = Field(default=False, description="是否排除自身向量")
    offset: int = Field(default=0, description="结果偏移用于分页")
    material_filter: Optional[str] = Field(default=None, description="按材料过滤")
    complexity_filter: Optional[str] = Field(default=None, description="按复杂度过滤")
    fine_part_type_filter: Optional[str] = Field(default=None, description="按细分类过滤")
    coarse_part_type_filter: Optional[str] = Field(default=None, description="按粗分类过滤")
    decision_source_filter: Optional[str] = Field(default=None, description="按决策来源过滤")
    is_coarse_label_filter: Optional[bool] = Field(
        default=None,
        description="是否按 coarse label 标记过滤",
    )


class SimilarityTopKItem(BaseModel):
    id: str
    score: float
    material: Optional[str] = None
    complexity: Optional[str] = None
    format: Optional[str] = None
    part_type: Optional[str] = None
    fine_part_type: Optional[str] = None
    coarse_part_type: Optional[str] = None
    decision_source: Optional[str] = None
    is_coarse_label: Optional[bool] = None


class SimilarityTopKResponse(BaseModel):
    target_id: str
    k: int
    results: list[SimilarityTopKItem]
    status: Optional[str] = None
    error: Optional[Dict[str, Any]] = None


__all__ = [
    "AnalysisOptions",
    "AnalysisResult",
    "BatchClassifyResponse",
    "BatchClassifyResultItem",
    "SimilarityQuery",
    "SimilarityResult",
    "SimilarityTopKItem",
    "SimilarityTopKQuery",
    "SimilarityTopKResponse",
]
