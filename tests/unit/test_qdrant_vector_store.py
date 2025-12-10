"""
Unit tests for Qdrant Vector Store implementation.

Tests cover:
- Configuration handling
- Vector registration and retrieval
- Similarity search with filters
- Batch operations
- Error handling and fallback behavior
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass

# Test the module's conditional import handling
from src.core.vector_stores.qdrant_store import (
    QdrantConfig,
    VectorSearchResult,
    QDRANT_AVAILABLE,
)


class TestQdrantConfig:
    """Test QdrantConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = QdrantConfig()

        assert config.host == "localhost"
        assert config.port == 6333
        assert config.grpc_port == 6334
        assert config.api_key is None
        assert config.https is False
        assert config.collection_name == "cad_vectors"
        assert config.vector_size == 128
        assert config.distance == "Cosine"
        assert config.on_disk is False
        assert config.timeout == 30.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = QdrantConfig(
            host="qdrant.example.com",
            port=6334,
            api_key="secret-key",
            https=True,
            collection_name="custom_collection",
            vector_size=256,
            distance="Euclidean",
        )

        assert config.host == "qdrant.example.com"
        assert config.port == 6334
        assert config.api_key == "secret-key"
        assert config.https is True
        assert config.collection_name == "custom_collection"
        assert config.vector_size == 256
        assert config.distance == "Euclidean"

    def test_from_env(self):
        """Test configuration from environment variables."""
        env_vars = {
            "QDRANT_HOST": "qdrant-server",
            "QDRANT_PORT": "6335",
            "QDRANT_API_KEY": "env-key",
            "QDRANT_HTTPS": "true",
            "QDRANT_COLLECTION": "env_collection",
            "QDRANT_VECTOR_SIZE": "512",
            "QDRANT_DISTANCE": "Dot",
            "QDRANT_ON_DISK": "true",
            "QDRANT_TIMEOUT": "60.0",
        }

        with patch.dict("os.environ", env_vars):
            config = QdrantConfig.from_env()

            assert config.host == "qdrant-server"
            assert config.port == 6335
            assert config.api_key == "env-key"
            assert config.https is True
            assert config.collection_name == "env_collection"
            assert config.vector_size == 512
            assert config.distance == "Dot"
            assert config.on_disk is True
            assert config.timeout == 60.0


class TestVectorSearchResult:
    """Test VectorSearchResult dataclass."""

    def test_basic_result(self):
        """Test basic search result."""
        result = VectorSearchResult(
            id="doc-123",
            score=0.95,
        )

        assert result.id == "doc-123"
        assert result.score == 0.95
        assert result.metadata == {}
        assert result.vector is None

    def test_full_result(self):
        """Test search result with all fields."""
        result = VectorSearchResult(
            id="doc-456",
            score=0.87,
            metadata={"material": "steel", "category": "fastener"},
            vector=[0.1, 0.2, 0.3],
        )

        assert result.id == "doc-456"
        assert result.score == 0.87
        assert result.metadata["material"] == "steel"
        assert result.vector == [0.1, 0.2, 0.3]


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
class TestQdrantVectorStore:
    """Test QdrantVectorStore implementation."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock Qdrant client."""
        with patch("src.core.vector_stores.qdrant_store.QdrantClient") as mock:
            client_instance = MagicMock()

            # Mock get_collections
            mock_collection = MagicMock()
            mock_collection.name = "cad_vectors"
            client_instance.get_collections.return_value.collections = [mock_collection]

            mock.return_value = client_instance
            yield client_instance

    @pytest.fixture
    def vector_store(self, mock_qdrant_client):
        """Create a QdrantVectorStore instance with mocked client."""
        from src.core.vector_stores.qdrant_store import QdrantVectorStore

        store = QdrantVectorStore()
        store._client = mock_qdrant_client
        return store

    def test_initialization(self, mock_qdrant_client):
        """Test store initialization."""
        from src.core.vector_stores.qdrant_store import QdrantVectorStore

        store = QdrantVectorStore()
        assert store.config is not None
        assert store._initialized is False

    def test_health_check_success(self, vector_store, mock_qdrant_client):
        """Test successful health check."""
        mock_qdrant_client.get_collections.return_value.collections = []

        result = vector_store.health_check()
        assert result is True

    def test_health_check_failure(self, vector_store, mock_qdrant_client):
        """Test health check failure."""
        mock_qdrant_client.get_collections.side_effect = Exception("Connection failed")

        result = vector_store.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_register_vector(self, vector_store, mock_qdrant_client):
        """Test vector registration."""
        vector_store._initialized = True

        result = await vector_store.register_vector(
            vector_id="doc-123",
            vector=[0.1, 0.2, 0.3],
            metadata={"material": "steel"},
        )

        assert result is True
        mock_qdrant_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_vector_adds_timestamp(self, vector_store, mock_qdrant_client):
        """Test that registration adds timestamp if not present."""
        vector_store._initialized = True

        await vector_store.register_vector(
            vector_id="doc-123",
            vector=[0.1, 0.2, 0.3],
            metadata={"material": "steel"},
        )

        call_args = mock_qdrant_client.upsert.call_args
        points = call_args[1]["points"]
        assert "created_at" in points[0].payload

    @pytest.mark.asyncio
    async def test_search_similar(self, vector_store, mock_qdrant_client):
        """Test similarity search."""
        vector_store._initialized = True

        # Mock search results
        mock_hit = MagicMock()
        mock_hit.id = "doc-123"
        mock_hit.score = 0.95
        mock_hit.payload = {"material": "steel"}
        mock_hit.vector = None
        mock_qdrant_client.search.return_value = [mock_hit]

        results = await vector_store.search_similar(
            query_vector=[0.1, 0.2, 0.3],
            top_k=5,
        )

        assert len(results) == 1
        assert results[0].id == "doc-123"
        assert results[0].score == 0.95
        assert results[0].metadata["material"] == "steel"

    @pytest.mark.asyncio
    async def test_search_with_filter(self, vector_store, mock_qdrant_client):
        """Test similarity search with filter conditions."""
        vector_store._initialized = True
        mock_qdrant_client.search.return_value = []

        await vector_store.search_similar(
            query_vector=[0.1, 0.2, 0.3],
            top_k=5,
            filter_conditions={"material": "steel"},
        )

        # Verify filter was passed
        call_args = mock_qdrant_client.search.call_args
        assert call_args[1]["query_filter"] is not None

    @pytest.mark.asyncio
    async def test_get_vector(self, vector_store, mock_qdrant_client):
        """Test retrieving a specific vector."""
        vector_store._initialized = True

        mock_point = MagicMock()
        mock_point.id = "doc-123"
        mock_point.payload = {"material": "steel"}
        mock_point.vector = [0.1, 0.2, 0.3]
        mock_qdrant_client.retrieve.return_value = [mock_point]

        result = await vector_store.get_vector("doc-123")

        assert result is not None
        assert result.id == "doc-123"
        assert result.vector == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_get_vector_not_found(self, vector_store, mock_qdrant_client):
        """Test retrieving a non-existent vector."""
        vector_store._initialized = True
        mock_qdrant_client.retrieve.return_value = []

        result = await vector_store.get_vector("non-existent")

        assert result is None

    @pytest.mark.asyncio
    async def test_delete_vector(self, vector_store, mock_qdrant_client):
        """Test vector deletion."""
        vector_store._initialized = True

        result = await vector_store.delete_vector("doc-123")

        assert result is True
        mock_qdrant_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_count(self, vector_store, mock_qdrant_client):
        """Test counting vectors."""
        vector_store._initialized = True

        mock_count_result = MagicMock()
        mock_count_result.count = 42
        mock_qdrant_client.count.return_value = mock_count_result

        result = await vector_store.count()

        assert result == 42

    @pytest.mark.asyncio
    async def test_batch_registration(self, vector_store, mock_qdrant_client):
        """Test batch vector registration."""
        vector_store._initialized = True

        vectors = [
            ("doc-1", [0.1, 0.2], {"material": "steel"}),
            ("doc-2", [0.3, 0.4], {"material": "aluminum"}),
            ("doc-3", [0.5, 0.6], {"material": "copper"}),
        ]

        result = await vector_store.register_vectors_batch(vectors, batch_size=2)

        assert result == 3
        # Should have 2 batch calls (2 vectors + 1 vector)
        assert mock_qdrant_client.upsert.call_count == 2

    def test_close(self, vector_store, mock_qdrant_client):
        """Test closing the client connection."""
        vector_store._initialized = True

        vector_store.close()

        mock_qdrant_client.close.assert_called_once()
        assert vector_store._client is None
        assert vector_store._initialized is False

    def test_context_manager(self, mock_qdrant_client):
        """Test context manager protocol."""
        from src.core.vector_stores.qdrant_store import QdrantVectorStore

        with QdrantVectorStore() as store:
            store._client = mock_qdrant_client
            assert store is not None

        mock_qdrant_client.close.assert_called_once()


class TestVectorStoreFactory:
    """Test the vector store factory function."""

    def test_factory_exists(self):
        """Test that factory function is accessible."""
        from src.core.vector_stores import get_vector_store

        assert callable(get_vector_store)

    def test_factory_with_faiss_backend(self):
        """Test factory with FAISS backend (graceful handling if unavailable)."""
        from src.core.vector_stores import get_vector_store, VectorStoreError

        try:
            store = get_vector_store(backend="faiss")
            assert store is not None
        except VectorStoreError:
            # FAISS may not be available in test environment, which is acceptable
            pytest.skip("FAISS backend not available in test environment")

    @patch.dict("os.environ", {"VECTOR_STORE_BACKEND": "faiss"})
    def test_factory_respects_env_var(self):
        """Test factory respects environment variable."""
        from src.core.vector_stores import get_vector_store, VectorStoreError

        try:
            store = get_vector_store()
            # Should use FAISS based on env var
            assert store is not None
        except VectorStoreError:
            # Backend may not be available in test environment
            pytest.skip("Vector store backend not available in test environment")
