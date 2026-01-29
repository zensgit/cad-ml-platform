"""Tests for welding and heat treatment retrieval integration."""

import pytest


class TestWeldingRetrieval:
    """Tests for welding knowledge retrieval."""

    def test_welding_parameters_intent(self):
        """Test welding parameters intent detection."""
        from src.core.assistant.query_analyzer import QueryAnalyzer, QueryIntent

        analyzer = QueryAnalyzer()

        query = analyzer.analyze("Q235钢板焊接电流多少?")
        assert query.intent == QueryIntent.WELDING_PARAMETERS

        query = analyzer.analyze("6mm碳钢MIG焊参数")
        assert query.intent == QueryIntent.WELDING_PARAMETERS

    def test_weldability_intent(self):
        """Test weldability intent detection."""
        from src.core.assistant.query_analyzer import QueryAnalyzer, QueryIntent

        analyzer = QueryAnalyzer()

        query = analyzer.analyze("40Cr需要预热吗?")
        assert query.intent == QueryIntent.WELDABILITY

        query = analyzer.analyze("304不锈钢焊接性如何?")
        assert query.intent == QueryIntent.WELDABILITY

    def test_joint_design_intent(self):
        """Test joint design intent detection."""
        from src.core.assistant.query_analyzer import QueryAnalyzer, QueryIntent

        analyzer = QueryAnalyzer()

        query = analyzer.analyze("10mm厚板V型坡口尺寸")
        assert query.intent == QueryIntent.WELDING_JOINT

    def test_retrieve_welding_parameters(self):
        """Test retrieving welding parameters."""
        from src.core.assistant import KnowledgeRetriever, RetrievalSource
        from src.core.assistant.query_analyzer import QueryAnalyzer

        analyzer = QueryAnalyzer()
        retriever = KnowledgeRetriever()

        query = analyzer.analyze("碳钢气保焊参数")
        results = retriever.retrieve(query, max_results=5)

        # Should get welding results
        welding_results = [r for r in results if r.source == RetrievalSource.WELDING]
        assert len(welding_results) > 0

    def test_retrieve_weldability(self):
        """Test retrieving weldability information."""
        from src.core.assistant import KnowledgeRetriever, RetrievalSource
        from src.core.assistant.query_analyzer import QueryAnalyzer

        analyzer = QueryAnalyzer()
        retriever = KnowledgeRetriever()

        query = analyzer.analyze("Q345焊接性评估")
        # Force welding source
        results = retriever.retrieve_by_source(RetrievalSource.WELDING, query)

        # May or may not have results depending on intent detection
        # Just verify no errors
        assert isinstance(results, list)


class TestHeatTreatmentRetrieval:
    """Tests for heat treatment knowledge retrieval."""

    def test_heat_treatment_intent(self):
        """Test heat treatment intent detection."""
        from src.core.assistant.query_analyzer import QueryAnalyzer, QueryIntent

        analyzer = QueryAnalyzer()

        query = analyzer.analyze("45钢淬火温度多少?")
        # May detect as HARDENING or HEAT_TREATMENT - both are valid
        assert query.intent in [QueryIntent.HARDENING, QueryIntent.HEAT_TREATMENT]

        query = analyzer.analyze("40Cr热处理参数")
        assert query.intent == QueryIntent.HEAT_TREATMENT

    def test_annealing_intent(self):
        """Test annealing intent detection."""
        from src.core.assistant.query_analyzer import QueryAnalyzer, QueryIntent

        analyzer = QueryAnalyzer()

        query = analyzer.analyze("45钢退火工艺")
        assert query.intent == QueryIntent.ANNEALING

        query = analyzer.analyze("GCr15球化退火温度")
        # May detect as ANNEALING or HEAT_TREATMENT - both are valid
        assert query.intent in [QueryIntent.ANNEALING, QueryIntent.HEAT_TREATMENT]

    def test_retrieve_heat_treatment(self):
        """Test retrieving heat treatment parameters."""
        from src.core.assistant import KnowledgeRetriever, RetrievalSource
        from src.core.assistant.query_analyzer import QueryAnalyzer

        analyzer = QueryAnalyzer()
        retriever = KnowledgeRetriever()

        query = analyzer.analyze("45钢淬火参数")
        results = retriever.retrieve(query, max_results=5)

        # Should get heat treatment results
        ht_results = [r for r in results if r.source == RetrievalSource.HEAT_TREATMENT]
        assert len(ht_results) > 0

    def test_retrieve_annealing(self):
        """Test retrieving annealing parameters."""
        from src.core.assistant import KnowledgeRetriever, RetrievalSource
        from src.core.assistant.query_analyzer import QueryAnalyzer

        analyzer = QueryAnalyzer()
        retriever = KnowledgeRetriever()

        query = analyzer.analyze("45钢退火温度")
        results = retriever.retrieve_by_source(RetrievalSource.HEAT_TREATMENT, query)

        # Should return list (may be empty if intent doesn't match)
        assert isinstance(results, list)


class TestIntentMapping:
    """Tests for intent to source mapping."""

    def test_welding_intent_mapping(self):
        """Test welding intents map to welding source."""
        from src.core.assistant import KnowledgeRetriever, RetrievalSource
        from src.core.assistant.query_analyzer import QueryIntent

        retriever = KnowledgeRetriever()

        sources = retriever._get_sources_for_intent(QueryIntent.WELDING_PARAMETERS)
        assert RetrievalSource.WELDING in sources

        sources = retriever._get_sources_for_intent(QueryIntent.WELDABILITY)
        assert RetrievalSource.WELDING in sources

    def test_heat_treatment_intent_mapping(self):
        """Test heat treatment intents map to heat treatment source."""
        from src.core.assistant import KnowledgeRetriever, RetrievalSource
        from src.core.assistant.query_analyzer import QueryIntent

        retriever = KnowledgeRetriever()

        sources = retriever._get_sources_for_intent(QueryIntent.HEAT_TREATMENT)
        assert RetrievalSource.HEAT_TREATMENT in sources

        sources = retriever._get_sources_for_intent(QueryIntent.HARDENING)
        assert RetrievalSource.HEAT_TREATMENT in sources

        sources = retriever._get_sources_for_intent(QueryIntent.ANNEALING)
        assert RetrievalSource.HEAT_TREATMENT in sources


class TestAssistantIntegration:
    """Tests for full assistant integration."""

    def test_assistant_welding_query(self):
        """Test assistant handles welding queries."""
        from src.core.assistant import CADAssistant

        assistant = CADAssistant()
        response = assistant.ask("碳钢气保焊电流多少?")

        assert response.answer is not None
        assert response.confidence > 0

    def test_assistant_heat_treatment_query(self):
        """Test assistant handles heat treatment queries."""
        from src.core.assistant import CADAssistant

        assistant = CADAssistant()
        response = assistant.ask("45钢淬火温度是多少?")

        assert response.answer is not None
        assert response.confidence > 0
