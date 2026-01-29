"""Tests for CAD-ML Assistant module."""

import pytest

from src.core.assistant import (
    CADAssistant,
    AssistantConfig,
    QueryAnalyzer,
    QueryIntent,
    AnalyzedQuery,
    KnowledgeRetriever,
    RetrievalResult,
    ContextAssembler,
    AssembledContext,
)
from src.core.assistant.knowledge_retriever import RetrievalSource


class TestQueryAnalyzer:
    """Tests for QueryAnalyzer."""

    def test_analyze_material_property(self):
        """Test material property query detection."""
        analyzer = QueryAnalyzer()
        result = analyzer.analyze("304不锈钢的抗拉强度是多少?")

        assert result.intent == QueryIntent.MATERIAL_PROPERTY
        assert "304" in result.entities.get("material_grade", "")

    def test_analyze_thread_spec(self):
        """Test thread specification query detection."""
        analyzer = QueryAnalyzer()
        result = analyzer.analyze("M10螺纹的底孔尺寸?")

        assert result.intent == QueryIntent.THREAD_SPEC
        assert result.entities.get("thread_diameter") == "10"

    def test_analyze_fit_selection(self):
        """Test fit selection query detection."""
        analyzer = QueryAnalyzer()
        result = analyzer.analyze("H7/g6配合的间隙范围?")

        assert result.intent == QueryIntent.FIT_SELECTION
        assert result.entities.get("hole_tolerance") == "H7"
        assert result.entities.get("shaft_tolerance") == "g6"

    def test_analyze_bearing_spec(self):
        """Test bearing specification query detection."""
        analyzer = QueryAnalyzer()
        result = analyzer.analyze("6205轴承的尺寸规格?")

        assert result.intent == QueryIntent.BEARING_SPEC
        assert result.entities.get("bearing") == "6205"

    def test_analyze_cutting_parameters(self):
        """Test cutting parameters query detection."""
        analyzer = QueryAnalyzer()
        result = analyzer.analyze("车削不锈钢的切削速度?")

        assert result.intent == QueryIntent.CUTTING_PARAMETERS

    def test_analyze_tolerance_lookup(self):
        """Test tolerance lookup query detection."""
        analyzer = QueryAnalyzer()
        result = analyzer.analyze("IT7公差在25mm时的值?")

        assert result.intent == QueryIntent.TOLERANCE_LOOKUP
        assert result.entities.get("it_grade") == "7"

    def test_analyze_unknown_query(self):
        """Test unknown query handling."""
        analyzer = QueryAnalyzer()
        result = analyzer.analyze("今天天气怎么样?")

        # Should still return a result, but with lower confidence
        assert result.intent in [QueryIntent.UNKNOWN, QueryIntent.GENERAL_QUESTION]
        assert result.confidence < 0.5

    def test_get_suggested_queries(self):
        """Test query suggestions."""
        analyzer = QueryAnalyzer()
        suggestions = analyzer.get_suggested_queries("")

        assert len(suggestions) > 0
        assert all(isinstance(s, str) for s in suggestions)


class TestKnowledgeRetriever:
    """Tests for KnowledgeRetriever."""

    def test_retrieve_thread_spec(self):
        """Test thread specification retrieval."""
        analyzer = QueryAnalyzer()
        retriever = KnowledgeRetriever()

        query = analyzer.analyze("M10螺纹规格?")
        results = retriever.retrieve(query)

        assert len(results) > 0
        assert any(r.source == RetrievalSource.THREADS for r in results)

    def test_retrieve_bearing_spec(self):
        """Test bearing specification retrieval."""
        analyzer = QueryAnalyzer()
        retriever = KnowledgeRetriever()

        query = analyzer.analyze("6205轴承尺寸?")
        results = retriever.retrieve(query)

        assert len(results) > 0
        assert any(r.source == RetrievalSource.BEARINGS for r in results)

    def test_retrieve_tolerance(self):
        """Test tolerance retrieval."""
        analyzer = QueryAnalyzer()
        retriever = KnowledgeRetriever()

        query = analyzer.analyze("H7/g6配合?")
        results = retriever.retrieve(query)

        assert len(results) > 0
        assert any(r.source == RetrievalSource.TOLERANCE for r in results)

    def test_retrieve_max_results(self):
        """Test max results limit."""
        analyzer = QueryAnalyzer()
        retriever = KnowledgeRetriever()

        query = analyzer.analyze("配合选择?")
        results = retriever.retrieve(query, max_results=2)

        assert len(results) <= 2

    def test_retrieval_result_has_required_fields(self):
        """Test retrieval results have all required fields."""
        analyzer = QueryAnalyzer()
        retriever = KnowledgeRetriever()

        query = analyzer.analyze("M10螺纹?")
        results = retriever.retrieve(query)

        for result in results:
            assert isinstance(result.source, RetrievalSource)
            assert 0 <= result.relevance <= 1
            assert isinstance(result.data, dict)
            assert isinstance(result.summary, str)


class TestContextAssembler:
    """Tests for ContextAssembler."""

    def test_assemble_basic(self):
        """Test basic context assembly."""
        analyzer = QueryAnalyzer()
        retriever = KnowledgeRetriever()
        assembler = ContextAssembler()

        query = analyzer.analyze("M10螺纹底孔?")
        results = retriever.retrieve(query)
        context = assembler.assemble(query, results)

        assert isinstance(context, AssembledContext)
        assert len(context.system_prompt) > 0
        assert len(context.user_prompt) > 0
        assert context.token_estimate > 0

    def test_assemble_empty_results(self):
        """Test assembly with no results."""
        assembler = ContextAssembler()
        query = AnalyzedQuery(
            original_query="测试",
            intent=QueryIntent.UNKNOWN,
            confidence=0.3,
        )

        context = assembler.assemble(query, [])

        assert "未找到相关知识" in context.knowledge_context

    def test_assemble_includes_sources(self):
        """Test that assembled context tracks sources."""
        analyzer = QueryAnalyzer()
        retriever = KnowledgeRetriever()
        assembler = ContextAssembler()

        query = analyzer.analyze("M10螺纹?")
        results = retriever.retrieve(query)
        context = assembler.assemble(query, results)

        assert len(context.sources_used) > 0


class TestCADAssistant:
    """Tests for CADAssistant main class."""

    def test_ask_thread_query(self):
        """Test assistant with thread query."""
        assistant = CADAssistant()
        response = assistant.ask("M10螺纹的底孔尺寸?")

        assert response.answer is not None
        assert len(response.answer) > 0
        assert response.confidence > 0

    def test_ask_bearing_query(self):
        """Test assistant with bearing query."""
        assistant = CADAssistant()
        response = assistant.ask("6205轴承的尺寸?")

        assert response.answer is not None
        assert "6205" in response.answer or "25" in response.answer

    def test_ask_returns_sources(self):
        """Test that responses include sources."""
        assistant = CADAssistant()
        response = assistant.ask("M10螺纹规格?")

        # Should have sources when knowledge is found
        assert isinstance(response.sources, list)

    def test_ask_returns_metadata(self):
        """Test that responses include metadata."""
        assistant = CADAssistant()
        response = assistant.ask("M10螺纹?")

        assert "intent" in response.metadata
        assert "entities" in response.metadata

    def test_custom_llm_callback(self):
        """Test custom LLM callback."""
        def mock_llm(system_prompt: str, user_prompt: str) -> str:
            return "Mock response from custom LLM"

        assistant = CADAssistant(llm_callback=mock_llm)
        response = assistant.ask("M10螺纹?")

        assert "Mock response" in response.answer

    def test_get_suggestions(self):
        """Test query suggestions."""
        assistant = CADAssistant()
        suggestions = assistant.get_suggestions("")

        assert len(suggestions) > 0

    def test_get_supported_queries(self):
        """Test getting supported query examples."""
        assistant = CADAssistant()
        supported = assistant.get_supported_queries()

        assert "材料查询" in supported
        assert "公差配合" in supported
        assert "标准件" in supported
        assert "加工参数" in supported

    def test_config_verbose_mode(self):
        """Test verbose configuration."""
        config = AssistantConfig(verbose=True)
        assistant = CADAssistant(config=config)
        response = assistant.ask("M10螺纹?")

        # In verbose mode, context should be included
        assert response.context_used is not None


class TestIntegration:
    """Integration tests for the full assistant pipeline."""

    def test_full_pipeline_thread(self):
        """Test full pipeline with thread query."""
        assistant = CADAssistant()
        response = assistant.ask("M10粗牙螺纹的攻丝底孔直径是多少?")

        # Should find thread info
        assert response.confidence > 0.5
        # Answer should mention the drill size (8.5mm for M10)
        assert "8.5" in response.answer or "底孔" in response.answer

    def test_full_pipeline_tolerance(self):
        """Test full pipeline with tolerance query."""
        assistant = CADAssistant()
        response = assistant.ask("H7/g6配合是什么类型的配合?")

        # Should identify as clearance fit
        assert response.confidence > 0.3
        assert len(response.sources) > 0

    def test_full_pipeline_bearing(self):
        """Test full pipeline with bearing query."""
        assistant = CADAssistant()
        response = assistant.ask("内径25mm的轴承有哪些型号?")

        # Should find bearings with 25mm bore
        assert response.confidence > 0.3
        # Common 25mm bore bearings: 6005, 6205, 6305
        answer_lower = response.answer.lower()
        assert "6005" in answer_lower or "6205" in answer_lower or "6305" in answer_lower or "25" in response.answer

    def test_fallback_for_unknown_query(self):
        """Test graceful fallback for unknown queries."""
        assistant = CADAssistant()
        response = assistant.ask("这是一个完全无关的问题")

        # Should still return a response
        assert response.answer is not None
        assert response.confidence < 0.5  # Lower confidence for unknown
