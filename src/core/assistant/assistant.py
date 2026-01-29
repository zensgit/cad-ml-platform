"""
CAD-ML Intelligent Assistant.

Main assistant class that orchestrates query analysis, knowledge retrieval,
context assembly, and response generation.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum

from .query_analyzer import QueryAnalyzer, AnalyzedQuery, QueryIntent
from .knowledge_retriever import KnowledgeRetriever, RetrievalResult
from .context_assembler import ContextAssembler, AssembledContext
from .llm_providers import (
    BaseLLMProvider,
    LLMConfig,
    get_provider,
    get_best_available_provider,
)


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    CLAUDE = "claude"
    GPT4 = "gpt4"
    QWEN = "qwen"
    LOCAL = "local"  # Local/offline mode


@dataclass
class AssistantConfig:
    """Configuration for CAD Assistant."""

    # LLM settings
    llm_provider: LLMProvider = LLMProvider.CLAUDE
    model_name: str = "claude-3-sonnet-20240229"
    temperature: float = 0.3
    max_tokens: int = 2000
    api_key: Optional[str] = None  # If None, uses environment variable

    # Retrieval settings
    max_retrieval_results: int = 5
    min_relevance_threshold: float = 0.5

    # Context settings
    max_context_tokens: int = 4000

    # Behavior settings
    language: str = "zh"  # "zh" or "en"
    verbose: bool = False
    auto_select_provider: bool = True  # Auto-select best available provider


@dataclass
class AssistantResponse:
    """Response from the assistant."""

    answer: str
    confidence: float
    sources: List[str] = field(default_factory=list)
    context_used: Optional[AssembledContext] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CADAssistant:
    """
    CAD-ML Intelligent Assistant with RAG architecture.

    Architecture:
        User Query → QueryAnalyzer → KnowledgeRetriever → ContextAssembler → LLM → Response

    Example:
        >>> assistant = CADAssistant()
        >>> response = assistant.ask("304不锈钢的抗拉强度是多少?")
        >>> print(response.answer)

        # With custom LLM callback
        >>> def my_llm(system_prompt, user_prompt):
        ...     return call_my_llm_api(system_prompt, user_prompt)
        >>> assistant = CADAssistant(llm_callback=my_llm)
    """

    def __init__(
        self,
        config: Optional[AssistantConfig] = None,
        llm_callback: Optional[Callable[[str, str], str]] = None,
    ):
        """
        Initialize CAD Assistant.

        Args:
            config: Assistant configuration
            llm_callback: Custom LLM callback function(system_prompt, user_prompt) -> response
        """
        self.config = config or AssistantConfig()

        # Initialize components
        self._query_analyzer = QueryAnalyzer()
        self._knowledge_retriever = KnowledgeRetriever()
        self._context_assembler = ContextAssembler(
            max_context_tokens=self.config.max_context_tokens
        )

        # Initialize LLM provider
        self._llm_provider: Optional[BaseLLMProvider] = None
        self._llm_callback = llm_callback

        if llm_callback is None:
            self._init_llm_provider()

    def _init_llm_provider(self) -> None:
        """Initialize the LLM provider based on configuration."""
        llm_config = LLMConfig(
            api_key=self.config.api_key,
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        if self.config.auto_select_provider:
            self._llm_provider = get_best_available_provider(llm_config)
        else:
            provider_map = {
                LLMProvider.CLAUDE: "claude",
                LLMProvider.GPT4: "openai",
                LLMProvider.QWEN: "qwen",
                LLMProvider.LOCAL: "ollama",
            }
            provider_name = provider_map.get(self.config.llm_provider, "offline")
            self._llm_provider = get_provider(provider_name, llm_config)

        if self.config.verbose:
            provider_type = type(self._llm_provider).__name__
            available = self._llm_provider.is_available()
            print(f"[LLM] Using {provider_type}, available: {available}")

    def ask(self, query: str) -> AssistantResponse:
        """
        Process a user query and return a response.

        Args:
            query: User's natural language query

        Returns:
            AssistantResponse with answer and metadata
        """
        # 1. Analyze query
        analyzed = self._query_analyzer.analyze(query)

        if self.config.verbose:
            print(f"[QueryAnalyzer] Intent: {analyzed.intent}, Confidence: {analyzed.confidence:.2f}")
            print(f"[QueryAnalyzer] Entities: {analyzed.entities}")

        # 2. Retrieve knowledge
        results = self._knowledge_retriever.retrieve(
            analyzed,
            max_results=self.config.max_retrieval_results,
        )

        # Filter by relevance threshold
        results = [r for r in results if r.relevance >= self.config.min_relevance_threshold]

        if self.config.verbose:
            print(f"[KnowledgeRetriever] Found {len(results)} relevant results")
            for r in results:
                print(f"  - {r.source.value}: {r.summary} (relevance: {r.relevance:.2f})")

        # 3. Assemble context
        context = self._context_assembler.assemble(analyzed, results)

        if self.config.verbose:
            print(f"[ContextAssembler] Token estimate: {context.token_estimate}")

        # 4. Generate response
        if results:
            answer = self._call_llm(context.system_prompt, context.user_prompt)
            confidence = min(analyzed.confidence, max(r.relevance for r in results))
        else:
            answer = self._generate_fallback_response(analyzed)
            confidence = 0.3

        # 5. Build response
        sources = [f"{r.source.value}: {r.summary}" for r in results]

        return AssistantResponse(
            answer=answer,
            confidence=confidence,
            sources=sources,
            context_used=context if self.config.verbose else None,
            metadata={
                "intent": analyzed.intent.value,
                "entities": analyzed.entities,
                "retrieval_count": len(results),
            },
        )

    def ask_with_context(
        self,
        query: str,
        additional_context: str = "",
    ) -> AssistantResponse:
        """
        Process query with additional user-provided context.

        Args:
            query: User's query
            additional_context: Additional context to include

        Returns:
            AssistantResponse
        """
        # Analyze and retrieve as normal
        analyzed = self._query_analyzer.analyze(query)
        results = self._knowledge_retriever.retrieve(
            analyzed,
            max_results=self.config.max_retrieval_results,
        )

        # Assemble context
        context = self._context_assembler.assemble(analyzed, results)

        # Append additional context to user prompt
        enhanced_prompt = context.user_prompt + f"\n\n附加背景信息:\n{additional_context}"

        # Generate response
        answer = self._call_llm(context.system_prompt, enhanced_prompt)
        confidence = analyzed.confidence * 0.9 if results else 0.5

        sources = [f"{r.source.value}: {r.summary}" for r in results]

        return AssistantResponse(
            answer=answer,
            confidence=confidence,
            sources=sources,
            metadata={
                "intent": analyzed.intent.value,
                "has_additional_context": True,
            },
        )

    def get_suggestions(self, partial_query: str) -> List[str]:
        """
        Get query suggestions for autocomplete.

        Args:
            partial_query: Partial user input

        Returns:
            List of suggested queries
        """
        return self._query_analyzer.get_suggested_queries(partial_query)

    def get_supported_queries(self) -> Dict[str, List[str]]:
        """
        Get examples of supported query types.

        Returns:
            Dictionary mapping query types to examples
        """
        return {
            "材料查询": [
                "304不锈钢的机械性能是什么?",
                "6061铝合金的密度是多少?",
                "比较Q235和45钢的强度",
            ],
            "公差配合": [
                "IT7公差等级在25mm时的值是多少?",
                "H7/g6配合的间隙范围?",
                "轴承配合应该用什么公差?",
            ],
            "标准件": [
                "M10螺纹的底孔尺寸?",
                "6205轴承的尺寸规格?",
                "20x3的O形圈沟槽尺寸?",
            ],
            "加工参数": [
                "车削304不锈钢用什么转速?",
                "加工钛合金用什么刀具?",
                "铝合金铣削的进给量?",
            ],
        }

    def _default_llm_callback(self, system_prompt: str, user_prompt: str) -> str:
        """
        Default LLM callback - returns knowledge-based answer without external LLM.

        In production, replace with actual LLM API call.
        """
        # Extract key information from user_prompt
        if "参考知识:" in user_prompt:
            knowledge_section = user_prompt.split("参考知识:")[1].split("请基于以上知识")[0]

            # Simple response based on retrieved knowledge
            if knowledge_section.strip() and knowledge_section.strip() != "未找到相关知识。":
                return f"根据知识库查询结果:\n\n{knowledge_section.strip()}\n\n如需更详细的信息，请提供更具体的查询条件。"

        return "抱歉，知识库中未找到与您问题直接相关的信息。请尝试更具体的查询，或联系技术支持获取帮助。"

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call LLM to generate response.

        Uses custom callback if provided, otherwise uses configured provider.
        """
        # Use custom callback if provided
        if self._llm_callback is not None:
            return self._llm_callback(system_prompt, user_prompt)

        # Use configured LLM provider
        if self._llm_provider is not None and self._llm_provider.is_available():
            try:
                return self._llm_provider.generate(system_prompt, user_prompt)
            except Exception as e:
                if self.config.verbose:
                    print(f"[LLM] Error: {e}, falling back to default")
                return self._default_llm_callback(system_prompt, user_prompt)

        # Fallback to default
        return self._default_llm_callback(system_prompt, user_prompt)

    def _generate_fallback_response(self, analyzed: AnalyzedQuery) -> str:
        """Generate fallback response when no knowledge is found."""
        intent_hints = {
            QueryIntent.MATERIAL_PROPERTY: "请提供具体的材料牌号（如304、Q235、6061等）以查询属性。",
            QueryIntent.TOLERANCE_LOOKUP: "请提供公差等级（如IT7）和尺寸（如25mm）以查询公差值。",
            QueryIntent.FIT_SELECTION: "请提供配合代号（如H7/g6）和尺寸以查询配合详情。",
            QueryIntent.THREAD_SPEC: "请提供螺纹规格（如M10、M10x1）以查询螺纹参数。",
            QueryIntent.BEARING_SPEC: "请提供轴承型号（如6205）或内径尺寸以查询轴承规格。",
            QueryIntent.CUTTING_PARAMETERS: "请提供材料类型以查询切削参数推荐。",
        }

        hint = intent_hints.get(
            analyzed.intent,
            "请尝试更具体的查询，例如指定材料牌号、尺寸或型号。"
        )

        return f"未能找到匹配的信息。{hint}"

    def set_llm_callback(self, callback: Callable[[str, str], str]) -> None:
        """
        Set custom LLM callback for response generation.

        Args:
            callback: Function(system_prompt, user_prompt) -> response_text
        """
        self._llm_callback = callback
