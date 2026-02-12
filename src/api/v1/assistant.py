"""CAD Intelligent Assistant API endpoints."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================


class QueryRequest(BaseModel):
    """Assistant query request."""

    query: str = Field(..., description="用户查询", min_length=1, max_length=2000)
    context: Optional[str] = Field(None, description="附加上下文信息")
    language: str = Field("zh", description="响应语言: zh 或 en")
    verbose: bool = Field(False, description="是否返回详细信息")


class QuerySource(BaseModel):
    """Knowledge source reference."""

    source: str = Field(..., description="知识来源")
    summary: str = Field(..., description="内容摘要")


class QueryResponse(BaseModel):
    """Assistant query response."""

    success: bool = Field(..., description="是否成功")
    answer: str = Field(..., description="回答内容")
    confidence: float = Field(..., description="置信度 (0-1)")
    sources: List[str] = Field(default_factory=list, description="参考来源")
    intent: Optional[str] = Field(None, description="识别的意图")
    entities: Optional[Dict[str, Any]] = Field(None, description="提取的实体")


class SuggestionResponse(BaseModel):
    """Query suggestion response."""

    suggestions: List[str] = Field(..., description="建议查询列表")


class SupportedQueriesResponse(BaseModel):
    """Supported query types response."""

    categories: Dict[str, List[str]] = Field(..., description="按类别分组的示例查询")


class ProviderInfo(BaseModel):
    """LLM provider information."""

    name: str = Field(..., description="提供商名称")
    available: bool = Field(..., description="是否可用")
    model: Optional[str] = Field(None, description="模型名称")


class StatusResponse(BaseModel):
    """Assistant status response."""

    status: str = Field(..., description="服务状态")
    provider: ProviderInfo = Field(..., description="当前 LLM 提供商")
    knowledge_modules: List[str] = Field(..., description="已加载的知识模块")


# ============================================================================
# Singleton Assistant Instance
# ============================================================================

_assistant_instance = None


def get_assistant():
    """Get or create assistant instance."""
    global _assistant_instance
    if _assistant_instance is None:
        from src.core.assistant import CADAssistant, AssistantConfig

        config = AssistantConfig(
            verbose=False,
            auto_select_provider=True,
        )
        _assistant_instance = CADAssistant(config=config)
    return _assistant_instance


# ============================================================================
# API Endpoints
# ============================================================================


@router.post("/query", response_model=QueryResponse)
async def query_assistant(request: QueryRequest) -> QueryResponse:
    """
    向智能助手提问

    支持的查询类型：
    - 材料查询：材料属性、性能比较
    - 公差配合：IT公差、配合选择
    - 标准件：螺纹、轴承、O形圈规格
    - 加工参数：切削参数、刀具选择
    - 设计标准：表面粗糙度、一般公差

    示例查询：
    - "304不锈钢的抗拉强度是多少？"
    - "IT7公差在25mm时的值是多少？"
    - "M10螺纹的底孔尺寸？"
    - "车削铝合金用什么转速？"
    """
    try:
        assistant = get_assistant()

        # Query with or without additional context
        if request.context:
            response = assistant.ask_with_context(
                query=request.query,
                additional_context=request.context,
            )
        else:
            response = assistant.ask(request.query)

        logger.info(
            "assistant.query",
            extra={
                "query": request.query[:100],
                "intent": response.metadata.get("intent"),
                "confidence": response.confidence,
            },
        )

        return QueryResponse(
            success=True,
            answer=response.answer,
            confidence=response.confidence,
            sources=response.sources,
            intent=response.metadata.get("intent") if request.verbose else None,
            entities=response.metadata.get("entities") if request.verbose else None,
        )

    except Exception as e:
        logger.error(f"assistant.query.error: {e}")
        return QueryResponse(
            success=False,
            answer=f"处理查询时发生错误: {str(e)}",
            confidence=0.0,
            sources=[],
        )


@router.get("/suggest", response_model=SuggestionResponse)
async def get_suggestions(
    q: str = Query(..., description="部分查询文本", min_length=1),
    limit: int = Query(5, description="返回数量", ge=1, le=10),
) -> SuggestionResponse:
    """
    获取查询建议（自动补全）

    根据用户输入的部分文本，返回可能的查询建议
    """
    try:
        assistant = get_assistant()
        suggestions = assistant.get_suggestions(q)
        return SuggestionResponse(suggestions=suggestions[:limit])
    except Exception as e:
        logger.error(f"assistant.suggest.error: {e}")
        return SuggestionResponse(suggestions=[])


@router.get("/supported-queries", response_model=SupportedQueriesResponse)
async def get_supported_queries() -> SupportedQueriesResponse:
    """
    获取支持的查询类型

    返回按类别分组的示例查询，帮助用户了解可以问什么问题
    """
    assistant = get_assistant()
    categories = assistant.get_supported_queries()
    return SupportedQueriesResponse(categories=categories)


@router.get("/status", response_model=StatusResponse)
async def get_assistant_status() -> StatusResponse:
    """
    获取助手状态

    返回当前助手的配置状态，包括：
    - 服务状态
    - LLM 提供商信息
    - 已加载的知识模块
    """
    try:
        assistant = get_assistant()

        # Get provider info
        provider_name = "offline"
        provider_available = False
        model_name = None

        if assistant._llm_provider is not None:
            provider_name = type(assistant._llm_provider).__name__.replace("Provider", "").lower()
            provider_available = assistant._llm_provider.is_available()
            model_name = assistant._llm_provider.config.model_name if hasattr(assistant._llm_provider, 'config') else None

        # List knowledge modules
        knowledge_modules = [
            "materials",  # 材料数据库
            "tolerance",  # 公差配合
            "standards",  # 标准件
            "machining",  # 加工参数
            "design_standards",  # 设计标准
        ]

        return StatusResponse(
            status="running",
            provider=ProviderInfo(
                name=provider_name,
                available=provider_available,
                model=model_name,
            ),
            knowledge_modules=knowledge_modules,
        )

    except Exception as e:
        logger.error(f"assistant.status.error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def submit_feedback(
    query: str = Query(..., description="原始查询"),
    answer: str = Query(..., description="助手回答"),
    rating: int = Query(..., description="评分 (1-5)", ge=1, le=5),
    comment: Optional[str] = Query(None, description="反馈评论"),
) -> Dict[str, Any]:
    """
    提交反馈

    对助手的回答进行评价，用于改进服务质量
    """
    logger.info(
        "assistant.feedback",
        extra={
            "query": query[:100],
            "rating": rating,
            "has_comment": comment is not None,
        },
    )

    return {
        "success": True,
        "message": "感谢您的反馈！",
    }


__all__ = ["router"]
