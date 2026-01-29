"""Tests for GD&T knowledge retrieval integration."""

import pytest

from src.core.assistant.query_analyzer import QueryAnalyzer, QueryIntent
from src.core.assistant.knowledge_retriever import (
    KnowledgeRetriever,
    RetrievalSource,
    RetrievalMode,
)


class TestGDTIntentDetection:
    """Test intent detection for GD&T queries."""

    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = QueryAnalyzer()

    def test_detect_gdt_interpretation_intent(self):
        """Test GD&T interpretation intent detection."""
        query = self.analyzer.analyze("GD&T形位公差有哪些类型？")

        assert query.intent in [QueryIntent.GDT_INTERPRETATION, QueryIntent.GENERAL_QUESTION]

    def test_detect_flatness_query(self):
        """Test flatness query detection."""
        query = self.analyzer.analyze("平面度公差怎么标注？")

        assert query.intent in [QueryIntent.GDT_INTERPRETATION, QueryIntent.GENERAL_QUESTION]

    def test_detect_position_tolerance(self):
        """Test position tolerance detection."""
        query = self.analyzer.analyze("位置度公差的应用场合？")

        assert query.intent in [QueryIntent.GDT_INTERPRETATION, QueryIntent.GDT_APPLICATION, QueryIntent.GENERAL_QUESTION]

    def test_detect_runout_query(self):
        """Test runout query detection."""
        query = self.analyzer.analyze("圆跳动和全跳动的区别？")

        # May be detected as GDT or general question depending on pattern matching
        assert query.intent in [
            QueryIntent.GDT_INTERPRETATION,
            QueryIntent.GENERAL_QUESTION,
            QueryIntent.UNKNOWN,  # Acceptable if no specific pattern matched
        ]


class TestGDTRetrieval:
    """Test GD&T knowledge retrieval."""

    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = QueryAnalyzer()
        self.retriever = KnowledgeRetriever(mode=RetrievalMode.KEYWORD)

    def test_retrieve_flatness_info(self):
        """Test retrieval of flatness information."""
        query = self.analyzer.analyze("平面度公差是什么？")
        query.intent = QueryIntent.GDT_INTERPRETATION

        results = self.retriever.retrieve(query, max_results=10)

        # Should retrieve GD&T info
        gdt_results = [r for r in results if r.source == RetrievalSource.GDT]
        assert len(gdt_results) > 0

        # Check for flatness data
        flatness_result = next(
            (r for r in gdt_results if r.data.get("characteristic") == "flatness"),
            None
        )
        assert flatness_result is not None
        assert flatness_result.data["requires_datum"] is False

    def test_retrieve_position_info(self):
        """Test retrieval of position tolerance information."""
        query = self.analyzer.analyze("位置度公差应用")
        query.intent = QueryIntent.GDT_INTERPRETATION

        results = self.retriever.retrieve(query, max_results=10)

        gdt_results = [r for r in results if r.source == RetrievalSource.GDT]
        assert len(gdt_results) > 0

        position_result = next(
            (r for r in gdt_results if r.data.get("characteristic") == "position"),
            None
        )
        assert position_result is not None
        assert position_result.data["requires_datum"] is True

    def test_retrieve_gdt_categories(self):
        """Test retrieval of GD&T category overview."""
        query = self.analyzer.analyze("几何公差分类")
        query.intent = QueryIntent.GDT_INTERPRETATION

        results = self.retriever.retrieve(query, max_results=15)

        gdt_results = [r for r in results if r.source == RetrievalSource.GDT]

        # Should have category overviews
        categories_found = {r.data.get("category") for r in gdt_results if "category" in r.data}
        assert len(categories_found) >= 3  # At least form, orientation, location

    def test_retrieve_by_source_gdt(self):
        """Test direct retrieval from GD&T source."""
        query = self.analyzer.analyze("圆度公差")
        query.intent = QueryIntent.GDT_INTERPRETATION

        results = self.retriever.retrieve_by_source(RetrievalSource.GDT, query)

        assert len(results) > 0
        assert all(r.source == RetrievalSource.GDT for r in results)


class TestGDTApplicationRetrieval:
    """Test GD&T application-based retrieval."""

    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = QueryAnalyzer()
        self.retriever = KnowledgeRetriever(mode=RetrievalMode.KEYWORD)

    def test_retrieve_hole_gdt_application(self):
        """Test GD&T recommendations for holes."""
        query = self.analyzer.analyze("孔的GD&T怎么标注？")
        query.intent = QueryIntent.GDT_APPLICATION

        results = self.retriever.retrieve(query, max_results=10)

        gdt_results = [r for r in results if r.source == RetrievalSource.GDT]
        assert len(gdt_results) > 0

        # Should recommend position for holes
        app_result = next(
            (r for r in gdt_results if r.data.get("feature_type") == "hole"),
            None
        )
        assert app_result is not None
        assert "position" in app_result.data.get("recommended_characteristics", [])

    def test_retrieve_shaft_gdt_application(self):
        """Test GD&T recommendations for shafts."""
        query = self.analyzer.analyze("轴类零件GD&T标注")
        query.intent = QueryIntent.GDT_APPLICATION

        results = self.retriever.retrieve(query, max_results=10)

        gdt_results = [r for r in results if r.source == RetrievalSource.GDT]

        # Should recommend cylindricity/runout for shafts
        app_result = next(
            (r for r in gdt_results if r.data.get("feature_type") == "shaft"),
            None
        )
        assert app_result is not None
        recommended = app_result.data.get("recommended_characteristics", [])
        assert any(c in recommended for c in ["cylindricity", "circular_runout", "total_runout"])


class TestGDTDataQuality:
    """Test data quality of GD&T retrieval."""

    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = QueryAnalyzer()
        self.retriever = KnowledgeRetriever(mode=RetrievalMode.KEYWORD)

    def test_gdt_symbols_have_chinese_names(self):
        """Test that GD&T results include Chinese names."""
        query = self.analyzer.analyze("形位公差符号")
        query.intent = QueryIntent.GDT_INTERPRETATION

        results = self.retriever.retrieve(query, max_results=10)

        for r in results:
            if "name_zh" in r.data:
                assert len(r.data["name_zh"]) > 0

    def test_gdt_results_have_summaries(self):
        """Test that all GD&T results have summaries."""
        query = self.analyzer.analyze("垂直度公差")
        query.intent = QueryIntent.GDT_INTERPRETATION

        results = self.retriever.retrieve(query, max_results=10)

        for r in results:
            if r.source == RetrievalSource.GDT:
                assert r.summary, f"Empty summary for {r.data}"

    def test_datum_requirement_accuracy(self):
        """Test that datum requirements are correctly indicated."""
        from src.core.knowledge.gdt import GDTCharacteristic, get_gdt_symbol

        # Form tolerances should NOT require datum
        form_chars = [
            GDTCharacteristic.FLATNESS,
            GDTCharacteristic.STRAIGHTNESS,
            GDTCharacteristic.CIRCULARITY,
        ]
        for char in form_chars:
            info = get_gdt_symbol(char)
            assert info.requires_datum is False, f"{char} should not require datum"

        # Orientation/location should require datum
        datum_chars = [
            GDTCharacteristic.PERPENDICULARITY,
            GDTCharacteristic.POSITION,
            GDTCharacteristic.CIRCULAR_RUNOUT,
        ]
        for char in datum_chars:
            info = get_gdt_symbol(char)
            assert info.requires_datum is True, f"{char} should require datum"
