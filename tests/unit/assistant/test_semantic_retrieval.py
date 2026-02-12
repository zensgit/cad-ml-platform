"""Tests for semantic retrieval module."""

import json
import os
import shutil
import tempfile

import pytest

from src.core.assistant.semantic_retrieval import (
    EmbeddingResult,
    SemanticSearchResult,
    SimpleEmbeddingProvider,
    VectorStore,
    SemanticRetriever,
    create_semantic_retriever,
)


class TestSimpleEmbeddingProvider:
    """Tests for SimpleEmbeddingProvider."""

    def setup_method(self):
        """Setup test fixtures."""
        self.provider = SimpleEmbeddingProvider(dimension=128)

    def test_embed_text(self):
        """Test embedding a single text."""
        text = "304不锈钢的抗拉强度"
        vector = self.provider.embed_text(text)

        assert len(vector) == 128
        assert isinstance(vector, list)
        assert all(isinstance(v, float) for v in vector)

    def test_embed_normalized(self):
        """Test that embeddings are L2 normalized."""
        text = "测试文本"
        vector = self.provider.embed_text(text)

        # L2 norm should be approximately 1
        norm = sum(v * v for v in vector) ** 0.5
        assert abs(norm - 1.0) < 0.01 or norm == 0

    def test_embed_batch(self):
        """Test embedding multiple texts."""
        texts = ["文本1", "文本2", "文本3"]
        vectors = self.provider.embed_batch(texts)

        assert len(vectors) == 3
        assert all(len(v) == 128 for v in vectors)

    def test_similar_texts_similar_vectors(self):
        """Test that similar texts have similar embeddings."""
        text1 = "304不锈钢的强度"
        text2 = "304不锈钢的抗拉强度"
        text3 = "铝合金的密度"

        vec1 = self.provider.embed_text(text1)
        vec2 = self.provider.embed_text(text2)
        vec3 = self.provider.embed_text(text3)

        # Calculate cosine similarities
        def cosine_sim(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            if norm_a == 0 or norm_b == 0:
                return 0
            return dot / (norm_a * norm_b)

        sim_12 = cosine_sim(vec1, vec2)
        sim_13 = cosine_sim(vec1, vec3)

        # Similar texts should have higher similarity
        assert sim_12 > sim_13

    def test_empty_text(self):
        """Test embedding empty text."""
        vector = self.provider.embed_text("")
        assert len(vector) == 128
        assert all(v == 0 for v in vector)

    def test_dimension_property(self):
        """Test dimension property."""
        assert self.provider.dimension == 128

        provider_256 = SimpleEmbeddingProvider(dimension=256)
        assert provider_256.dimension == 256


class TestVectorStore:
    """Tests for VectorStore."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.store = VectorStore(dimension=128)

    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_vector(self):
        """Test adding a vector."""
        vector = [0.1] * 128
        idx = self.store.add("测试文本", vector, "test_source")

        assert idx == 0
        assert len(self.store) == 1

    def test_add_batch(self):
        """Test adding multiple vectors."""
        texts = ["文本1", "文本2", "文本3"]
        vectors = [[0.1 * (i + 1)] * 128 for i in range(3)]
        sources = ["source1", "source2", "source3"]

        indices = self.store.add_batch(texts, vectors, sources)

        assert indices == [0, 1, 2]
        assert len(self.store) == 3

    def test_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        vector = [0.1] * 64  # Wrong dimension

        with pytest.raises(ValueError):
            self.store.add("测试", vector)

    def test_search(self):
        """Test searching for similar vectors."""
        # Add vectors
        texts = ["304不锈钢", "316不锈钢", "铝合金"]
        vectors = [
            [1.0] + [0.0] * 127,  # Similar to query
            [0.9] + [0.1] * 127,  # Somewhat similar
            [0.0] + [1.0] * 127,  # Different
        ]

        self.store.add_batch(texts, vectors)

        # Search
        query = [1.0] + [0.0] * 127
        results = self.store.search(query, top_k=2)

        assert len(results) == 2
        assert results[0].text == "304不锈钢"
        assert results[0].score > results[1].score

    def test_search_with_source_filter(self):
        """Test searching with source filter."""
        self.store.add("文本1", [1.0] + [0.0] * 127, "source_a")
        self.store.add("文本2", [0.9] + [0.1] * 127, "source_b")
        self.store.add("文本3", [0.8] + [0.2] * 127, "source_a")

        query = [1.0] + [0.0] * 127
        results = self.store.search(query, source_filter="source_a")

        assert len(results) == 2
        assert all(r.source == "source_a" for r in results)

    def test_search_with_min_score(self):
        """Test searching with minimum score threshold."""
        self.store.add("高相似", [1.0] + [0.0] * 127)
        self.store.add("低相似", [0.0] + [1.0] + [0.0] * 126)

        query = [1.0] + [0.0] * 127
        results = self.store.search(query, min_score=0.5)

        assert len(results) == 1
        assert results[0].text == "高相似"

    def test_search_empty_store(self):
        """Test searching empty store."""
        query = [1.0] + [0.0] * 127
        results = self.store.search(query)

        assert len(results) == 0

    def test_persistence(self):
        """Test saving and loading store."""
        storage_path = os.path.join(self.temp_dir, "vectors.json")
        store = VectorStore(dimension=128, storage_path=storage_path)

        # Add data
        store.add("测试1", [0.1] * 128, "source1", {"key": "value1"})
        store.add("测试2", [0.2] * 128, "source2", {"key": "value2"})

        # Save
        assert store.save() is True

        # Load in new store
        new_store = VectorStore(dimension=128, storage_path=storage_path)

        assert len(new_store) == 2

        # Search to verify data
        results = new_store.search([0.1] * 128, top_k=1)
        assert results[0].text == "测试1"

    def test_clear(self):
        """Test clearing the store."""
        self.store.add("测试", [0.1] * 128)
        assert len(self.store) == 1

        self.store.clear()
        assert len(self.store) == 0


class TestSemanticRetriever:
    """Tests for SemanticRetriever."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.retriever = SemanticRetriever()

    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_index_text(self):
        """Test indexing a single text."""
        idx = self.retriever.index_text(
            "304不锈钢的抗拉强度约为520MPa",
            source="materials",
            metadata={"material": "304"},
        )

        assert idx == 0
        assert self.retriever.indexed_count == 1

    def test_index_batch(self):
        """Test indexing multiple texts."""
        texts = [
            "304不锈钢的强度",
            "316L不锈钢的耐腐蚀性",
            "6061铝合金的密度",
        ]
        sources = ["materials"] * 3

        indices = self.retriever.index_batch(texts, sources)

        assert len(indices) == 3
        assert self.retriever.indexed_count == 3

    def test_index_knowledge_base(self):
        """Test indexing from knowledge base format."""
        knowledge_items = [
            {"content": "304不锈钢的强度是520MPa", "source": "materials", "type": "property"},
            {"content": "TIG焊接适用于薄板", "source": "welding", "type": "application"},
            {"content": "平面度公差标注方法", "source": "gdt", "type": "tolerance"},
        ]

        count = self.retriever.index_knowledge_base(knowledge_items)

        assert count == 3
        assert self.retriever.indexed_count == 3

    def test_search(self):
        """Test semantic search."""
        # Index some content
        texts = [
            "304不锈钢的抗拉强度约为520MPa",
            "316L不锈钢具有更好的耐腐蚀性",
            "TIG焊接主要参数包括电流和电压",
        ]
        self.retriever.index_batch(texts, ["materials", "materials", "welding"])

        # Search
        results = self.retriever.search("不锈钢强度", top_k=2)

        assert len(results) <= 2
        # First result should be about 304 strength
        if results:
            assert "304" in results[0].text or "强度" in results[0].text

    def test_search_with_source_filter(self):
        """Test search with source filter."""
        self.retriever.index_text("材料A", source="materials")
        self.retriever.index_text("焊接B", source="welding")
        self.retriever.index_text("材料C", source="materials")

        results = self.retriever.search("材料", source_filter="materials")

        assert all(r.source == "materials" for r in results)

    def test_hybrid_search(self):
        """Test hybrid search combining semantic and keyword results."""
        # Index content
        texts = [
            "304不锈钢的抗拉强度约为520MPa",
            "316L不锈钢的强度略低于304",
            "铝合金6061的强度",
        ]
        self.retriever.index_batch(texts)

        # Keyword results (simulated)
        keyword_results = [
            {"content": "304不锈钢的抗拉强度约为520MPa", "score": 0.9},
            {"content": "不锈钢的通用性能", "score": 0.7},
        ]

        # Hybrid search
        results = self.retriever.hybrid_search(
            "304不锈钢强度",
            keyword_results,
            top_k=3,
        )

        assert len(results) <= 3
        # 304 strength should be ranked high
        assert any("304" in r.text and "强度" in r.text for r in results[:2])

    def test_save_and_load(self):
        """Test persistence."""
        storage_path = os.path.join(self.temp_dir, "retriever.json")
        retriever = SemanticRetriever(storage_path=storage_path)

        retriever.index_text("测试内容", source="test")
        assert retriever.save() is True

        # New retriever should load data
        new_retriever = SemanticRetriever(storage_path=storage_path)
        assert new_retriever.indexed_count == 1

    def test_clear(self):
        """Test clearing indexed content."""
        self.retriever.index_text("测试")
        assert self.retriever.indexed_count == 1

        self.retriever.clear()
        assert self.retriever.indexed_count == 0


class TestCreateSemanticRetriever:
    """Tests for factory function."""

    def test_create_with_simple_provider(self):
        """Test creating retriever with simple provider."""
        retriever = create_semantic_retriever(use_transformers=False)

        assert retriever is not None
        assert isinstance(retriever.embedding_provider, SimpleEmbeddingProvider)

    def test_create_with_custom_weight(self):
        """Test creating retriever with custom hybrid weight."""
        retriever = create_semantic_retriever(hybrid_weight=0.5)

        assert retriever.hybrid_weight == 0.5


class TestSemanticRetrieverIntegration:
    """Integration tests for semantic retrieval."""

    def setup_method(self):
        """Setup test fixtures."""
        self.retriever = SemanticRetriever()

    def test_cad_knowledge_retrieval(self):
        """Test retrieval of CAD/manufacturing knowledge."""
        # Index CAD knowledge
        knowledge = [
            {"content": "304不锈钢的抗拉强度约为520MPa，屈服强度约为205MPa", "source": "materials"},
            {"content": "316L不锈钢具有优异的耐腐蚀性，适用于化工设备", "source": "materials"},
            {"content": "TIG焊接适用于不锈钢薄板，焊缝质量高", "source": "welding"},
            {"content": "平面度公差用于控制表面的平整程度", "source": "gdt"},
            {"content": "位置度公差需要指定基准，常用于孔位控制", "source": "gdt"},
            {"content": "阳极氧化可提高铝合金的耐腐蚀性和硬度", "source": "surface"},
        ]

        self.retriever.index_knowledge_base(knowledge)

        # Test various queries with very low threshold for simple n-gram embeddings
        # Material query
        results = self.retriever.search("不锈钢的强度", top_k=3, min_score=0.0)
        assert len(results) > 0

        # Welding query with exact match terms
        results = self.retriever.search("TIG焊接", top_k=3, min_score=0.0)
        assert any("TIG" in r.text or "焊接" in r.text for r in results)

        # GD&T query
        results = self.retriever.search("公差", top_k=3, min_score=0.0)
        assert any("公差" in r.text for r in results)

    def test_chinese_text_handling(self):
        """Test handling of Chinese text."""
        texts = [
            "氩弧焊参数设置",
            "埋弧焊自动化应用",
            "激光焊接精度控制",
        ]

        self.retriever.index_batch(texts, ["welding"] * 3)

        # Use min_score=0 to ensure results are returned
        results = self.retriever.search("焊接", top_k=3, min_score=0.0)
        assert len(results) == 3
        assert all("焊" in r.text for r in results)

    def test_mixed_language_handling(self):
        """Test handling of mixed Chinese and English."""
        texts = [
            "TIG焊接 (GTAW) 参数设置",
            "MIG焊接自动化",
            "GD&T形位公差标注规范",
        ]

        self.retriever.index_batch(texts)

        # Search with English - use lower threshold for simple embeddings
        results = self.retriever.search("TIG welding", top_k=2, min_score=0.1)
        assert any("TIG" in r.text for r in results)

        # Search with Chinese
        results = self.retriever.search("形位公差", top_k=2, min_score=0.1)
        assert any("GD&T" in r.text for r in results)
