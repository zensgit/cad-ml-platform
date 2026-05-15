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


class QueryEvidence(BaseModel):
    """Structured evidence for a grounded assistant answer."""

    reference_id: str = Field(..., description="证据引用编号")
    source: str = Field(..., description="知识来源模块")
    summary: str = Field(..., description="证据摘要")
    relevance: float = Field(..., description="相关性分数 (0-1)")
    match_type: str = Field(..., description="命中方式")
    key_facts: List[str] = Field(default_factory=list, description="关键事实列表")


class QueryAlternative(BaseModel):
    """Alternative candidate for assistant explainability."""

    label: str = Field(..., description="候选标签或来源")
    confidence: float = Field(..., description="候选置信度 (0-1)")


class QueryKnowledgeCitation(BaseModel):
    """Rule-level citation extracted from shared decision evidence."""

    evidence_source: str = Field(..., description="证据来源")
    rule_source: str = Field(..., description="规则来源")
    rule_version: str = Field(..., description="规则版本")
    categories: List[str] = Field(default_factory=list, description="知识检查类别")
    standards: List[str] = Field(default_factory=list, description="标准候选类别")


class QueryUncertainty(BaseModel):
    """Uncertainty description for assistant explainability."""

    score: float = Field(..., description="不确定性分数 (0-1)")
    reasons: List[str] = Field(default_factory=list, description="不确定性来源")


class QueryExplainability(BaseModel):
    """Stable explainability contract for assistant responses."""

    summary: str = Field(..., description="可读解释摘要")
    decision_path: List[str] = Field(default_factory=list, description="决策路径")
    source_contributions: Dict[str, float] = Field(
        default_factory=dict,
        description="结构化来源贡献",
    )
    alternative_labels: List[QueryAlternative] = Field(
        default_factory=list,
        description="其他候选来源或标签",
    )
    uncertainty: QueryUncertainty = Field(..., description="不确定性摘要")
    contract_version: Optional[str] = Field(default=None, description="决策合同版本")
    decision_contract: Optional[Dict[str, Any]] = Field(
        default=None,
        description="共享最终决策合同",
    )
    decision_evidence: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="来自 DecisionService 的结构化决策证据",
    )
    knowledge_citations: List[QueryKnowledgeCitation] = Field(
        default_factory=list,
        description="来自 analyze 决策证据的知识规则引用",
    )
    rule_sources: List[str] = Field(default_factory=list, description="规则来源列表")
    rule_versions: List[str] = Field(default_factory=list, description="规则版本列表")
    fallback_flags: List[str] = Field(default_factory=list, description="降级/回退标记")
    review_reasons: List[str] = Field(default_factory=list, description="复核原因列表")


class QueryResponse(BaseModel):
    """Assistant query response."""

    success: bool = Field(..., description="是否成功")
    answer: str = Field(..., description="回答内容")
    confidence: float = Field(..., description="置信度 (0-1)")
    sources: List[str] = Field(default_factory=list, description="参考来源")
    evidence: List[QueryEvidence] = Field(default_factory=list, description="结构化证据")
    explainability: Optional[QueryExplainability] = Field(
        None,
        description="稳定 explainability 输出",
    )
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


def _safe_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _list_text(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        values = [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _decision_contract_from_metadata(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    contract = metadata.get("decision_contract")
    return dict(contract) if isinstance(contract, dict) else None


def _decision_evidence_from_metadata(
    metadata: Dict[str, Any],
    decision_contract: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    raw_evidence = metadata.get("decision_evidence")
    if raw_evidence is None and isinstance(decision_contract, dict):
        raw_evidence = decision_contract.get("evidence")
    if not isinstance(raw_evidence, list):
        return []
    rows: List[Dict[str, Any]] = []
    for item in raw_evidence:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "").strip()
        if not source:
            continue
        rows.append(dict(item))
    return rows


def _source_contributions_from_decision_evidence(
    evidence_rows: List[Dict[str, Any]],
) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for row in evidence_rows:
        source = str(row.get("source") or "").strip()
        if not source:
            continue
        score = _safe_float(row.get("contribution"), default=None)
        if score is None:
            score = _safe_float(row.get("confidence"), default=None)
        if score is None or score <= 0:
            continue
        totals[source] = totals.get(source, 0.0) + score
    total = sum(totals.values())
    if total <= 0:
        return {}
    return {
        key: round(value / total, 6)
        for key, value in sorted(totals.items(), key=lambda pair: -pair[1])
    }


def _decision_alternatives(
    evidence_rows: List[Dict[str, Any]],
    source_contributions: Dict[str, float],
) -> List[QueryAlternative]:
    alternatives: List[QueryAlternative] = []
    seen = set()
    for row in evidence_rows:
        source = str(row.get("source") or "").strip()
        if not source or source in seen:
            continue
        seen.add(source)
        label = str(row.get("label") or source).strip()
        confidence = source_contributions.get(source)
        if confidence is None:
            confidence = _safe_float(row.get("confidence"), default=0.0) or 0.0
        alternatives.append(QueryAlternative(label=label, confidence=confidence))
    return alternatives[:4]


def _knowledge_citations_from_decision_evidence(
    evidence_rows: List[Dict[str, Any]],
) -> List[QueryKnowledgeCitation]:
    citations: List[QueryKnowledgeCitation] = []
    seen: set[tuple[str, str, str]] = set()
    for row in evidence_rows:
        source = str(row.get("source") or "").strip()
        details = row.get("details") if isinstance(row.get("details"), dict) else {}
        rule_sources = _list_text(
            row.get("rule_source")
            or details.get("rule_sources")
            or details.get("rule_source")
        )
        rule_versions = _list_text(
            row.get("rule_version")
            or details.get("rule_versions")
            or details.get("rule_version")
        )
        if not rule_sources:
            continue
        categories = _list_text(
            row.get("category")
            or details.get("check_categories")
            or details.get("categories")
        )
        standards = _list_text(
            row.get("type")
            or details.get("standards_candidate_types")
            or details.get("standards")
        )
        rule_version = rule_versions[0] if rule_versions else "unknown"
        for rule_source in rule_sources:
            key = (source, rule_source, rule_version)
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                QueryKnowledgeCitation(
                    evidence_source=source or "knowledge",
                    rule_source=rule_source,
                    rule_version=rule_version,
                    categories=categories,
                    standards=standards,
                )
            )
    return citations


def _append_knowledge_citation_note(
    answer: str,
    explainability: QueryExplainability,
) -> str:
    citations = explainability.knowledge_citations
    if not citations:
        return answer
    citation_tokens = [
        f"{item.rule_source}@{item.rule_version}"
        for item in citations[:5]
    ]
    note = "知识规则引用: " + "; ".join(citation_tokens)
    if note in answer:
        return answer
    return f"{answer.rstrip()}\n\n{note}"


def _build_query_explainability(response: Any) -> QueryExplainability:
    """Build a stable explainability payload from assistant evidence and metadata."""
    metadata = dict(getattr(response, "metadata", {}) or {})
    evidence_items = list(getattr(response, "evidence", []) or [])
    decision_contract = _decision_contract_from_metadata(metadata)
    decision_evidence = _decision_evidence_from_metadata(metadata, decision_contract)
    retrieval_count = int(metadata.get("retrieval_count") or len(evidence_items) or 0)
    source_totals: Dict[str, float] = {}
    for item in evidence_items:
        source_name = str(getattr(item, "source", "") or "").strip()
        relevance = float(getattr(item, "relevance", 0.0) or 0.0)
        if source_name:
            source_totals[source_name] = source_totals.get(source_name, 0.0) + relevance
    total_relevance = sum(source_totals.values())
    if total_relevance > 0:
        source_contributions = {
            key: round(value / total_relevance, 6)
            for key, value in sorted(source_totals.items(), key=lambda pair: -pair[1])
        }
    else:
        source_contributions = {}

    decision_source_contributions = _source_contributions_from_decision_evidence(
        decision_evidence
    )
    if decision_source_contributions:
        source_contributions = decision_source_contributions

    if decision_evidence:
        alternative_labels = _decision_alternatives(
            decision_evidence,
            source_contributions,
        )
    else:
        alternative_labels = [
            QueryAlternative(label=key, confidence=value)
            for key, value in list(source_contributions.items())[1:4]
        ]

    uncertainty_reasons: List[str] = []
    uncertainty_score = max(0.0, min(1.0, 1.0 - float(getattr(response, "confidence", 0.0))))
    if retrieval_count <= 0:
        uncertainty_reasons.append("no_retrieval_hits")
        uncertainty_score = max(uncertainty_score, 0.8)
    elif retrieval_count < 2:
        uncertainty_reasons.append("limited_retrieval_coverage")
        uncertainty_score = max(uncertainty_score, 0.45)
    if len(source_contributions) <= 1:
        uncertainty_reasons.append("single_source_grounding")
        uncertainty_score = max(uncertainty_score, 0.35)
    fallback_flags = _list_text(
        (decision_contract or {}).get("fallback_flags") or metadata.get("fallback_flags")
    )
    review_reasons = _list_text(
        (decision_contract or {}).get("review_reasons") or metadata.get("review_reasons")
    )
    branch_conflicts = (
        (decision_contract or {}).get("branch_conflicts")
        if isinstance((decision_contract or {}).get("branch_conflicts"), dict)
        else {}
    )
    if review_reasons:
        uncertainty_reasons.extend(
            reason for reason in review_reasons if reason not in uncertainty_reasons
        )
        uncertainty_score = max(uncertainty_score, 0.45)
    if branch_conflicts:
        uncertainty_reasons.append("branch_conflict")
        uncertainty_score = max(uncertainty_score, 0.6)
    if fallback_flags:
        uncertainty_reasons.append("fallback_flags_present")
        uncertainty_score = max(uncertainty_score, 0.5)

    knowledge_citations = _knowledge_citations_from_decision_evidence(decision_evidence)
    rule_sources = list(
        dict.fromkeys(item.rule_source for item in knowledge_citations)
    )
    rule_versions = list(
        dict.fromkeys(item.rule_version for item in knowledge_citations)
    )

    decision_path = [
        "query_analyzed",
        "knowledge_retrieved" if retrieval_count > 0 else "knowledge_missing",
        "context_assembled",
        "response_generated",
    ]
    if decision_contract:
        decision_path.append("decision_contract_loaded")
    if decision_evidence:
        decision_path.append("decision_evidence_grounded")
    if source_contributions:
        decision_path.append("structured_evidence_grounded")
    if knowledge_citations:
        decision_path.append("knowledge_rule_citations_grounded")

    summary_parts = [
        f"intent={metadata.get('intent') or 'unknown'}",
        f"retrieval={retrieval_count}",
        f"evidence={len(evidence_items)}",
    ]
    if decision_contract:
        summary_parts.append(
            f"contract={decision_contract.get('contract_version') or 'unknown'}"
        )
        summary_parts.append(
            f"decision_source={decision_contract.get('decision_source') or 'unknown'}"
        )
        if decision_contract.get("fine_part_type"):
            summary_parts.append(f"fine={decision_contract.get('fine_part_type')}")
    if source_contributions:
        primary_source = next(iter(source_contributions))
        summary_parts.append(f"primary_source={primary_source}")
    if rule_versions:
        summary_parts.append(f"rule_version={rule_versions[0]}")
    if uncertainty_reasons:
        summary_parts.append(f"uncertainty={','.join(uncertainty_reasons[:2])}")

    return QueryExplainability(
        summary="; ".join(summary_parts),
        decision_path=decision_path,
        source_contributions=source_contributions,
        alternative_labels=alternative_labels,
        uncertainty=QueryUncertainty(
            score=round(uncertainty_score, 6),
            reasons=uncertainty_reasons,
        ),
        contract_version=(decision_contract or {}).get("contract_version"),
        decision_contract=decision_contract,
        decision_evidence=decision_evidence,
        knowledge_citations=knowledge_citations,
        rule_sources=rule_sources,
        rule_versions=rule_versions,
        fallback_flags=fallback_flags,
        review_reasons=review_reasons,
    )


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

        explainability = _build_query_explainability(response)

        return QueryResponse(
            success=True,
            answer=_append_knowledge_citation_note(response.answer, explainability),
            confidence=response.confidence,
            sources=response.sources,
            evidence=[QueryEvidence(**item.to_dict()) for item in response.evidence],
            explainability=explainability,
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
            evidence=[],
            explainability=QueryExplainability(
                summary="assistant_error",
                decision_path=["query_failed"],
                source_contributions={},
                alternative_labels=[],
                uncertainty=QueryUncertainty(score=1.0, reasons=["assistant_error"]),
            ),
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
            model_name = (
                assistant._llm_provider.config.model_name
                if hasattr(assistant._llm_provider, "config")
                else None
            )

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
