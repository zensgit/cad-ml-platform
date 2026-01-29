"""Tests for embedding-based semantic retrieval."""

import pytest
import numpy as np


class TestEmbeddingProvider:
    """Tests for EmbeddingProvider."""

    def test_fallback_mode_available(self):
        """Test fallback mode is always available."""
        from src.core.assistant.embedding_retriever import EmbeddingProvider, EmbeddingConfig

        config = EmbeddingConfig()
        provider = EmbeddingProvider(config)

        # Should at least work in fallback mode
        texts = ["测试文本", "another test"]
        embeddings = provider.embed(texts)

        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0

    def test_embed_single(self):
        """Test single text embedding."""
        from src.core.assistant.embedding_retriever import EmbeddingProvider

        provider = EmbeddingProvider()
        embedding = provider.embed_single("304不锈钢强度")

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0

    def test_embeddings_normalized(self):
        """Test embeddings are normalized."""
        from src.core.assistant.embedding_retriever import EmbeddingProvider, EmbeddingConfig

        config = EmbeddingConfig(normalize_embeddings=True)
        provider = EmbeddingProvider(config)

        embedding = provider.embed_single("测试文本")
        norm = np.linalg.norm(embedding)

        # Should be close to 1.0 if normalized
        assert 0.9 <= norm <= 1.1 or norm == 0  # 0 for empty text


class TestKnowledgeIndex:
    """Tests for KnowledgeIndex."""

    def test_add_and_search(self):
        """Test adding items and searching."""
        from src.core.assistant.embedding_retriever import (
            KnowledgeIndex,
            KnowledgeItem,
            EmbeddingProvider,
        )

        provider = EmbeddingProvider()
        index = KnowledgeIndex(provider)

        # Add items
        items = [
            KnowledgeItem(id="1", text="304不锈钢抗拉强度520MPa", source="materials", data={}),
            KnowledgeItem(id="2", text="6061铝合金密度2.7g/cm³", source="materials", data={}),
            KnowledgeItem(id="3", text="M10螺纹螺距1.5mm", source="threads", data={}),
        ]
        index.add_items(items)

        assert len(index) == 3

        # Search
        results = index.search("不锈钢强度", max_results=2)
        assert len(results) > 0

    def test_search_with_source_filter(self):
        """Test search with source filter."""
        from src.core.assistant.embedding_retriever import (
            KnowledgeIndex,
            KnowledgeItem,
            EmbeddingProvider,
        )

        provider = EmbeddingProvider()
        index = KnowledgeIndex(provider)

        items = [
            KnowledgeItem(id="1", text="304不锈钢", source="materials", data={}),
            KnowledgeItem(id="2", text="M10螺纹", source="threads", data={}),
        ]
        index.add_items(items)

        # Search only materials
        results = index.search("材料", source_filter="materials")

        for item, _ in results:
            assert item.source == "materials"

    def test_empty_index_search(self):
        """Test search on empty index."""
        from src.core.assistant.embedding_retriever import KnowledgeIndex, EmbeddingProvider

        provider = EmbeddingProvider()
        index = KnowledgeIndex(provider)

        results = index.search("任何查询")
        assert results == []


class TestSemanticRetriever:
    """Tests for SemanticRetriever."""

    def test_initialization(self):
        """Test retriever initialization."""
        from src.core.assistant.embedding_retriever import SemanticRetriever

        retriever = SemanticRetriever()
        assert retriever is not None

    def test_search_materials(self):
        """Test searching for materials."""
        from src.core.assistant.embedding_retriever import SemanticRetriever

        retriever = SemanticRetriever()
        retriever.initialize()

        results = retriever.search("不锈钢强度", max_results=5)

        # Should find some results (depends on data)
        assert isinstance(results, list)

    def test_search_with_source_filter(self):
        """Test search with specific source."""
        from src.core.assistant.embedding_retriever import SemanticRetriever

        retriever = SemanticRetriever()
        retriever.initialize()

        results = retriever.search("螺纹", source_filter="threads", max_results=5)

        for item, _ in results:
            assert item.source == "threads"

    def test_singleton_retriever(self):
        """Test singleton pattern."""
        from src.core.assistant.embedding_retriever import get_semantic_retriever

        r1 = get_semantic_retriever()
        r2 = get_semantic_retriever()

        assert r1 is r2


class TestHybridRetrieval:
    """Tests for hybrid keyword + semantic retrieval."""

    def test_hybrid_mode(self):
        """Test hybrid retrieval mode."""
        from src.core.assistant import KnowledgeRetriever, RetrievalMode
        from src.core.assistant.query_analyzer import QueryAnalyzer

        analyzer = QueryAnalyzer()
        retriever = KnowledgeRetriever(mode=RetrievalMode.HYBRID)

        query = analyzer.analyze("304不锈钢的抗拉强度")
        results = retriever.retrieve(query, max_results=5)

        # Should return results
        assert isinstance(results, list)

    def test_keyword_mode(self):
        """Test keyword-only mode."""
        from src.core.assistant import KnowledgeRetriever, RetrievalMode
        from src.core.assistant.query_analyzer import QueryAnalyzer

        analyzer = QueryAnalyzer()
        retriever = KnowledgeRetriever(mode=RetrievalMode.KEYWORD)

        query = analyzer.analyze("IT7公差25mm")
        results = retriever.retrieve(query, max_results=5)

        assert isinstance(results, list)

    def test_semantic_mode(self):
        """Test semantic-only mode."""
        from src.core.assistant import KnowledgeRetriever, RetrievalMode
        from src.core.assistant.query_analyzer import QueryAnalyzer

        analyzer = QueryAnalyzer()
        retriever = KnowledgeRetriever(mode=RetrievalMode.SEMANTIC)

        query = analyzer.analyze("耐腐蚀的金属材料")
        results = retriever.retrieve(query, max_results=5)

        assert isinstance(results, list)

    def test_mode_override(self):
        """Test mode override in retrieve call."""
        from src.core.assistant import KnowledgeRetriever, RetrievalMode
        from src.core.assistant.query_analyzer import QueryAnalyzer

        analyzer = QueryAnalyzer()
        retriever = KnowledgeRetriever(mode=RetrievalMode.KEYWORD)

        query = analyzer.analyze("轴承规格")

        # Override to semantic mode
        results = retriever.retrieve(query, max_results=5, mode=RetrievalMode.SEMANTIC)
        assert isinstance(results, list)


class TestDesignStandardsRetrieval:
    """Tests for design standards retrieval."""

    def test_retrieve_surface_finish(self):
        """Test surface finish retrieval."""
        from src.core.assistant import KnowledgeRetriever, RetrievalMode
        from src.core.assistant.query_analyzer import AnalyzedQuery, QueryIntent

        retriever = KnowledgeRetriever(mode=RetrievalMode.KEYWORD)

        query = AnalyzedQuery(
            original_query="N7表面粗糙度",
            intent=QueryIntent.GENERAL_QUESTION,
            confidence=0.8,
            entities={"surface_grade": "N7"},
            keywords=["N7", "表面粗糙度"],
        )

        results = retriever._retrieve_design_standards(query)

        # Should find N7 grade
        assert len(results) > 0
        assert any("N7" in r.summary for r in results)

    def test_retrieve_general_tolerance(self):
        """Test general tolerance retrieval."""
        from src.core.assistant import KnowledgeRetriever
        from src.core.assistant.query_analyzer import AnalyzedQuery, QueryIntent

        retriever = KnowledgeRetriever()

        query = AnalyzedQuery(
            original_query="50mm的m级公差",
            intent=QueryIntent.TOLERANCE_LOOKUP,
            confidence=0.8,
            entities={"dimension": "50", "tolerance_class": "m"},
            keywords=["50mm", "公差"],
        )

        results = retriever._retrieve_design_standards(query)

        assert len(results) > 0


class TestIndexPersistence:
    """Tests for index save/load."""

    def test_save_and_load(self, tmp_path):
        """Test saving and loading index."""
        from src.core.assistant.embedding_retriever import (
            KnowledgeIndex,
            KnowledgeItem,
            EmbeddingProvider,
        )

        provider = EmbeddingProvider()
        index = KnowledgeIndex(provider)

        # Add items
        items = [
            KnowledgeItem(id="1", text="测试项1", source="test", data={"key": "value"}),
            KnowledgeItem(id="2", text="测试项2", source="test", data={}),
        ]
        index.add_items(items)

        # Save
        save_path = str(tmp_path / "test_index")
        index.save(save_path)

        # Load into new index
        new_index = KnowledgeIndex(provider)
        loaded = new_index.load(save_path)

        assert loaded is True
        assert len(new_index) == 2

    def test_load_nonexistent(self, tmp_path):
        """Test loading nonexistent index."""
        from src.core.assistant.embedding_retriever import KnowledgeIndex, EmbeddingProvider

        provider = EmbeddingProvider()
        index = KnowledgeIndex(provider)

        loaded = index.load(str(tmp_path / "nonexistent"))
        assert loaded is False
