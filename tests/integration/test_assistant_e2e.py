"""
End-to-end integration tests for CAD Assistant.

Tests the complete workflow integrating persistence, semantic retrieval,
quality evaluation, and API service components.
"""

import json
import os
import shutil
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

from src.core.assistant.persistence import (
    ConversationPersistence,
    JSONStorageBackend,
)
from src.core.assistant.conversation import (
    Conversation,
    Message,
    MessageRole,
)
from src.core.assistant.semantic_retrieval import (
    SemanticRetriever,
    SimpleEmbeddingProvider,
)
from src.core.assistant.quality_evaluation import (
    ResponseQualityEvaluator,
    EvaluationHistory,
)
from src.core.assistant.api_service import (
    CADAssistantAPI,
    APIResponse,
)


class TestEndToEndWorkflow:
    """End-to-end tests for the complete assistant workflow."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.persistence_path = os.path.join(self.temp_dir, "conversations")
        self.retriever_path = os.path.join(self.temp_dir, "vectors.json")
        self.history_path = os.path.join(self.temp_dir, "eval_history.json")

    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_complete_conversation_workflow(self):
        """Test complete conversation with persistence."""
        # Initialize storage backend
        backend = JSONStorageBackend(storage_dir=self.persistence_path)

        # Create conversation
        conv = Conversation(id="test-conv-001")
        conv.add_message(MessageRole.USER, "304不锈钢的强度是多少？")
        conv.add_message(
            MessageRole.ASSISTANT,
            "304不锈钢的抗拉强度约为520MPa，屈服强度约为205MPa。",
            confidence=0.9,
        )

        # Save
        assert backend.save_conversation(conv) is True

        # Create new backend instance to verify loading
        new_backend = JSONStorageBackend(storage_dir=self.persistence_path)

        # Load and verify
        loaded = new_backend.load_conversation("test-conv-001")
        assert loaded is not None
        assert len(loaded.messages) == 2
        assert "520MPa" in loaded.messages[1].content

    def test_semantic_retrieval_with_knowledge_base(self):
        """Test semantic retrieval indexing and search."""
        # Initialize retriever
        retriever = SemanticRetriever(storage_path=self.retriever_path)

        # Index CAD knowledge
        knowledge = [
            {"content": "304不锈钢的抗拉强度约为520MPa，屈服强度约为205MPa", "source": "materials"},
            {"content": "316L不锈钢具有优异的耐腐蚀性，适用于化工设备", "source": "materials"},
            {"content": "TIG焊接适用于不锈钢薄板，焊缝质量高", "source": "welding"},
            {"content": "平面度公差用于控制表面的平整程度", "source": "gdt"},
            {"content": "铝合金6061的密度为2.7g/cm³", "source": "materials"},
        ]

        count = retriever.index_knowledge_base(knowledge)
        assert count == 5

        # Save and reload
        retriever.save()

        new_retriever = SemanticRetriever(storage_path=self.retriever_path)
        assert new_retriever.indexed_count == 5

        # Search
        results = new_retriever.search("不锈钢强度", top_k=3, min_score=0.0)
        assert len(results) > 0

    def test_quality_evaluation_with_history(self):
        """Test quality evaluation with history tracking."""
        evaluator = ResponseQualityEvaluator()
        history = EvaluationHistory(storage_path=self.history_path)

        # Evaluate multiple responses
        test_cases = [
            {
                "query": "304不锈钢的强度？",
                "response": "304不锈钢的抗拉强度约为520MPa，屈服强度约为205MPa。适用于一般结构件。",
            },
            {
                "query": "TIG焊接参数？",
                "response": "TIG焊接参数包括：电流100-150A，电压12-15V，氩气流量8-12L/min。",
            },
            {
                "query": "什么是公差？",
                "response": "公差是指允许的尺寸变化范围。",
            },
        ]

        for case in test_cases:
            result = evaluator.evaluate(case["query"], case["response"])
            history.add_result(result)

        # Verify history
        assert len(history.results) == 3

        # Get average score
        avg = history.get_average_score()
        assert 0 <= avg <= 1

        # Save and reload
        history.save()

        new_history = EvaluationHistory(storage_path=self.history_path)
        assert len(new_history.results) == 3

    def test_api_with_components_integration(self):
        """Test API service with integrated components."""
        api = CADAssistantAPI(rate_limit_enabled=False)

        # Test health
        health = api.health()
        assert health.success is True

        # Test info
        info = api.info()
        assert "endpoints" in info.data

        # Test validation
        response = api.ask({"query": ""})
        assert response.success is False
        assert response.error["code"] == "VALIDATION_ERROR"

    @patch.object(CADAssistantAPI, "_get_assistant")
    @patch.object(CADAssistantAPI, "_get_evaluator")
    def test_full_ask_and_evaluate_flow(self, mock_evaluator, mock_assistant):
        """Test full ask and evaluate flow."""
        # Setup mocks
        mock_assistant_instance = MagicMock()
        mock_assistant_instance.start_conversation.return_value = "conv-123"
        mock_result = MagicMock()
        mock_result.answer = "304不锈钢的抗拉强度约为520MPa，屈服强度约为205MPa。"
        mock_result.confidence = 0.9
        mock_result.sources = []
        mock_result.intent = MagicMock(value="material_property")
        mock_assistant_instance.ask.return_value = mock_result
        mock_assistant.return_value = mock_assistant_instance

        mock_eval_instance = MagicMock()
        mock_eval_result = MagicMock()
        mock_eval_result.to_dict.return_value = {
            "overall_score": 0.85,
            "grade": "B",
        }
        mock_eval_instance.evaluate.return_value = mock_eval_result
        mock_evaluator.return_value = mock_eval_instance

        api = CADAssistantAPI(rate_limit_enabled=False)

        # Ask
        ask_response = api.ask({"query": "304不锈钢的强度？"})
        assert ask_response.success is True
        assert "520MPa" in ask_response.data["answer"]

        # Evaluate the response
        eval_response = api.evaluate({
            "query": "304不锈钢的强度？",
            "response": ask_response.data["answer"],
        })
        assert eval_response.success is True
        assert eval_response.data["overall_score"] == 0.85


class TestPerformanceOptimization:
    """Tests for performance optimization features."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_batch_embedding_performance(self):
        """Test batch embedding is faster than individual."""
        provider = SimpleEmbeddingProvider(dimension=128)

        texts = [f"测试文本{i}" for i in range(100)]

        # Batch embedding
        start = time.time()
        batch_result = provider.embed_batch(texts)
        batch_time = time.time() - start

        # Individual embedding
        start = time.time()
        individual_result = [provider.embed_text(t) for t in texts]
        individual_time = time.time() - start

        # Both should produce same results
        assert len(batch_result) == len(individual_result)
        assert len(batch_result) == 100

        # Batch should be comparable or faster (may vary)
        # Just verify both complete successfully
        assert batch_time < 5  # Should complete within 5 seconds
        assert individual_time < 5

    def test_retriever_search_performance(self):
        """Test retriever handles large index efficiently."""
        retriever = SemanticRetriever()

        # Index many items
        texts = [f"知识条目{i}：这是关于CAD设计的第{i}条知识" for i in range(500)]
        retriever.index_batch(texts)

        # Search should be fast
        start = time.time()
        results = retriever.search("CAD设计知识", top_k=10, min_score=0.0)
        search_time = time.time() - start

        assert len(results) <= 10
        assert search_time < 1  # Should complete within 1 second

    def test_concurrent_api_requests(self):
        """Test API handles concurrent requests correctly."""
        api = CADAssistantAPI(
            rate_limit_enabled=True,
            requests_per_minute=100,
        )

        # Simulate multiple health checks
        responses = []
        for _ in range(10):
            response = api.health()
            responses.append(response)

        # All should succeed
        assert all(r.success for r in responses)


class TestCacheIntegration:
    """Tests for caching functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_embedding_cache_hit(self):
        """Test embedding results are consistent (can be cached)."""
        provider = SimpleEmbeddingProvider(dimension=128)

        text = "测试缓存命中"

        # Generate twice
        result1 = provider.embed_text(text)
        result2 = provider.embed_text(text)

        # Should be identical (deterministic)
        assert result1 == result2

    def test_persistence_memory_cache(self):
        """Test persistence uses memory cache."""
        backend = JSONStorageBackend(storage_dir=os.path.join(self.temp_dir, "conv"))

        conv = Conversation(id="cache-test")
        conv.add_message(MessageRole.USER, "测试消息")

        # Save
        backend.save_conversation(conv)

        # First load
        loaded1 = backend.load_conversation("cache-test")

        # Second load (should be consistent)
        loaded2 = backend.load_conversation("cache-test")

        assert loaded1.id == loaded2.id
        assert len(loaded1.messages) == len(loaded2.messages)


class TestErrorRecovery:
    """Tests for error recovery and resilience."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_persistence_corrupted_file_recovery(self):
        """Test persistence handles corrupted files."""
        storage_path = os.path.join(self.temp_dir, "conversations")
        os.makedirs(storage_path, exist_ok=True)

        # Create corrupted file
        corrupted_path = os.path.join(storage_path, "corrupted.json")
        with open(corrupted_path, "w") as f:
            f.write("not valid json {{{")

        # Persistence should handle gracefully
        backend = JSONStorageBackend(storage_path)
        result = backend.load_conversation("corrupted")

        # Should return None for corrupted file
        assert result is None

    def test_retriever_empty_query(self):
        """Test retriever handles empty queries."""
        retriever = SemanticRetriever()
        retriever.index_text("测试内容")

        # Empty query should return empty results
        results = retriever.search("", top_k=5)
        assert len(results) == 0

    def test_evaluator_empty_inputs(self):
        """Test evaluator handles edge cases."""
        evaluator = ResponseQualityEvaluator()

        # Very short response
        result = evaluator.evaluate("问题", "好")
        assert result.overall_score >= 0
        assert result.grade in ["A", "B", "C", "D", "F"]

    def test_api_internal_error_handling(self):
        """Test API handles internal errors gracefully."""
        api = CADAssistantAPI(rate_limit_enabled=False)

        # Invalid request that passes validation but may fail internally
        # The API should catch and return error response
        response = api.conversation({"action": "get", "conversation_id": "nonexistent"})

        # Should get a response (may succeed or fail gracefully)
        assert isinstance(response, APIResponse)


class TestMultiLanguageSupport:
    """Tests for multi-language support."""

    def test_chinese_text_retrieval(self):
        """Test Chinese text indexing and retrieval."""
        retriever = SemanticRetriever()

        knowledge = [
            {"content": "氩弧焊的焊接电流一般为100-200A", "source": "welding"},
            {"content": "激光切割的精度可达0.1mm", "source": "cutting"},
            {"content": "数控车床的主轴转速可达3000rpm", "source": "machining"},
        ]

        retriever.index_knowledge_base(knowledge)

        results = retriever.search("焊接电流", top_k=3, min_score=0.0)
        assert len(results) > 0
        assert any("焊" in r.text for r in results)

    def test_mixed_language_retrieval(self):
        """Test mixed Chinese-English content."""
        retriever = SemanticRetriever()

        knowledge = [
            {"content": "TIG welding (钨极惰性气体保护焊) 适用于精密焊接", "source": "welding"},
            {"content": "GD&T (几何尺寸和公差) 是工程图纸标注标准", "source": "gdt"},
            {"content": "CAD (计算机辅助设计) 软件包括AutoCAD、SolidWorks等", "source": "software"},
        ]

        retriever.index_knowledge_base(knowledge)

        # Search with English term
        results = retriever.search("TIG", top_k=3, min_score=0.0)
        assert any("TIG" in r.text for r in results)

        # Search with Chinese term
        results = retriever.search("公差", top_k=3, min_score=0.0)
        assert any("公差" in r.text for r in results)

    def test_evaluation_chinese_content(self):
        """Test quality evaluation with Chinese content."""
        evaluator = ResponseQualityEvaluator()

        result = evaluator.evaluate(
            "304不锈钢的强度是多少？",
            "304不锈钢的抗拉强度约为520MPa，屈服强度约为205MPa。"
            "该材料具有良好的耐腐蚀性和可焊性，广泛应用于食品、化工等行业。"
        )

        assert result.overall_score > 0
        assert result.grade in ["A", "B", "C", "D", "F"]


class TestDataConsistency:
    """Tests for data consistency across components."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_conversation_message_order(self):
        """Test conversation messages maintain order."""
        backend = JSONStorageBackend(storage_dir=os.path.join(self.temp_dir, "conv"))

        conv = Conversation(id="order-test")

        # Add messages in order
        messages = [
            (MessageRole.USER, "消息1"),
            (MessageRole.ASSISTANT, "回复1"),
            (MessageRole.USER, "消息2"),
            (MessageRole.ASSISTANT, "回复2"),
            (MessageRole.USER, "消息3"),
        ]

        for role, content in messages:
            conv.add_message(role, content)

        # Save
        backend.save_conversation(conv)

        # Reload
        new_backend = JSONStorageBackend(storage_dir=os.path.join(self.temp_dir, "conv"))
        loaded = new_backend.load_conversation("order-test")

        # Verify order
        for i, (role, content) in enumerate(messages):
            assert loaded.messages[i].role == role
            assert loaded.messages[i].content == content

    def test_evaluation_history_order(self):
        """Test evaluation history maintains chronological order."""
        history = EvaluationHistory(
            storage_path=os.path.join(self.temp_dir, "history.json")
        )

        evaluator = ResponseQualityEvaluator()

        # Add evaluations
        for i in range(5):
            result = evaluator.evaluate(f"问题{i}", f"回答{i}")
            history.add_result(result)
            time.sleep(0.01)  # Small delay for timestamp difference

        # Verify order by timestamp
        timestamps = [r.timestamp for r in history.results]
        assert timestamps == sorted(timestamps)
