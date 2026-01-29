"""Tests for surface treatment knowledge retrieval integration."""

import pytest

from src.core.assistant.query_analyzer import QueryAnalyzer, QueryIntent
from src.core.assistant.knowledge_retriever import (
    KnowledgeRetriever,
    RetrievalSource,
    RetrievalMode,
)


class TestSurfaceTreatmentIntentDetection:
    """Test intent detection for surface treatment queries."""

    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = QueryAnalyzer()

    def test_detect_electroplating_intent(self):
        """Test electroplating intent detection."""
        query = self.analyzer.analyze("镀锌层厚度选择标准是什么？")

        assert query.intent in [QueryIntent.ELECTROPLATING, QueryIntent.GENERAL_QUESTION]

    def test_detect_anodizing_intent(self):
        """Test anodizing intent detection."""
        query = self.analyzer.analyze("铝合金阳极氧化类型有哪些？")

        assert query.intent in [QueryIntent.ANODIZING, QueryIntent.GENERAL_QUESTION]

    def test_detect_coating_intent(self):
        """Test coating intent detection."""
        query = self.analyzer.analyze("C4级环境用什么涂层？")

        assert query.intent in [QueryIntent.COATING, QueryIntent.GENERAL_QUESTION]

    def test_detect_salt_spray_plating(self):
        """Test plating query with salt spray requirement."""
        query = self.analyzer.analyze("盐雾试验500小时需要什么镀层？")

        assert query.intent in [QueryIntent.ELECTROPLATING, QueryIntent.GENERAL_QUESTION]

    def test_detect_hard_chrome(self):
        """Test hard chrome plating query."""
        query = self.analyzer.analyze("镀硬铬的厚度和硬度参数？")

        assert query.intent in [QueryIntent.ELECTROPLATING, QueryIntent.GENERAL_QUESTION]


class TestSurfaceTreatmentRetrieval:
    """Test surface treatment knowledge retrieval."""

    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = QueryAnalyzer()
        self.retriever = KnowledgeRetriever(mode=RetrievalMode.KEYWORD)

    def test_retrieve_electroplating_info(self):
        """Test retrieval of electroplating information."""
        query = self.analyzer.analyze("电镀层厚度选择")
        query.intent = QueryIntent.ELECTROPLATING

        results = self.retriever.retrieve(query, max_results=10)

        # Should retrieve plating types
        plating_results = [
            r for r in results if r.source == RetrievalSource.SURFACE_TREATMENT
        ]
        assert len(plating_results) > 0

        # Check for expected data fields
        for r in plating_results:
            if "plating_type" in r.data:
                assert "thickness_typical" in r.data

    def test_retrieve_anodizing_info(self):
        """Test retrieval of anodizing information."""
        query = self.analyzer.analyze("铝合金阳极氧化工艺")
        query.intent = QueryIntent.ANODIZING

        results = self.retriever.retrieve(query, max_results=10)

        # Should retrieve anodize types
        anodize_results = [
            r for r in results if r.source == RetrievalSource.SURFACE_TREATMENT
        ]
        assert len(anodize_results) > 0

        # Check for anodize type data
        has_anodize_type = any("anodize_type" in r.data for r in anodize_results)
        assert has_anodize_type

    def test_retrieve_coating_info(self):
        """Test retrieval of coating information."""
        query = self.analyzer.analyze("涂层选择推荐")
        query.intent = QueryIntent.COATING

        results = self.retriever.retrieve(query, max_results=10)

        # Should retrieve coating types
        coating_results = [
            r for r in results if r.source == RetrievalSource.SURFACE_TREATMENT
        ]
        assert len(coating_results) > 0

        # Check for coating type data
        has_coating_type = any("coating_type" in r.data for r in coating_results)
        assert has_coating_type

    def test_retrieve_anodize_colors(self):
        """Test retrieval of anodize color information."""
        query = self.analyzer.analyze("阳极氧化可以做什么颜色")
        query.intent = QueryIntent.ANODIZING

        results = self.retriever.retrieve(query, max_results=10)

        # Should include color information
        has_colors = any("colors" in r.data for r in results)
        assert has_colors

    def test_retrieve_by_source_surface_treatment(self):
        """Test direct retrieval from surface treatment source."""
        query = self.analyzer.analyze("电镀参数")
        query.intent = QueryIntent.ELECTROPLATING

        results = self.retriever.retrieve_by_source(
            RetrievalSource.SURFACE_TREATMENT,
            query
        )

        assert len(results) > 0
        assert all(r.source == RetrievalSource.SURFACE_TREATMENT for r in results)


class TestSurfaceTreatmentDataQuality:
    """Test data quality of surface treatment retrieval."""

    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = QueryAnalyzer()
        self.retriever = KnowledgeRetriever(mode=RetrievalMode.KEYWORD)

    def test_plating_thickness_values(self):
        """Test that plating thickness values are reasonable."""
        query = self.analyzer.analyze("电镀厚度")
        query.intent = QueryIntent.ELECTROPLATING

        results = self.retriever.retrieve(query)

        for r in results:
            if "thickness_typical" in r.data:
                thickness = r.data["thickness_typical"]
                # Typical plating thickness: 1-100 μm
                assert 0 < thickness < 500

    def test_anodize_temperature_values(self):
        """Test that anodize temperature values are reasonable."""
        query = self.analyzer.analyze("阳极氧化参数")
        query.intent = QueryIntent.ANODIZING

        results = self.retriever.retrieve(query)

        for r in results:
            if "temperature" in r.data:
                temp = r.data["temperature"]
                # Anodizing temperature: -10 to 50°C
                assert -10 < temp < 60

    def test_coating_dft_values(self):
        """Test that coating DFT values are reasonable."""
        query = self.analyzer.analyze("涂层厚度")
        query.intent = QueryIntent.COATING

        results = self.retriever.retrieve(query)

        for r in results:
            if "dft_recommended" in r.data:
                dft = r.data["dft_recommended"]
                # Typical DFT: 20-500 μm
                assert 0 < dft < 1000

    def test_result_summaries_not_empty(self):
        """Test that all results have non-empty summaries."""
        for intent in [QueryIntent.ELECTROPLATING, QueryIntent.ANODIZING, QueryIntent.COATING]:
            query = self.analyzer.analyze("表面处理")
            query.intent = intent

            results = self.retriever.retrieve(query)

            for r in results:
                assert r.summary, f"Empty summary for {r.data}"
                assert len(r.summary) > 5
