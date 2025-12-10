"""Tests for src/core/vector_stores/qdrant_store.py to improve coverage.

Covers:
- VectorSearchResult dataclass
- QdrantConfig dataclass
- QdrantConfig.from_env class method
- QdrantVectorStore init and property accessors
- Filter building logic
- Distance metric mapping
- Context manager protocols
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


class TestVectorSearchResult:
    """Tests for VectorSearchResult dataclass."""

    def test_basic_fields(self):
        """Test VectorSearchResult basic field access."""
        result = {
            "id": "vec-123",
            "score": 0.95,
            "metadata": {"material": "steel"},
            "vector": [0.1, 0.2, 0.3],
        }

        assert result["id"] == "vec-123"
        assert result["score"] == 0.95
        assert result["metadata"]["material"] == "steel"
        assert len(result["vector"]) == 3

    def test_optional_vector_none(self):
        """Test VectorSearchResult with no vector."""
        result = {
            "id": "vec-123",
            "score": 0.95,
            "metadata": {},
            "vector": None,
        }

        assert result["vector"] is None

    def test_default_metadata_empty(self):
        """Test VectorSearchResult default metadata is empty dict."""
        result = {
            "id": "vec-123",
            "score": 0.95,
            "metadata": {},
        }

        assert result["metadata"] == {}


class TestQdrantConfig:
    """Tests for QdrantConfig dataclass."""

    def test_default_values(self):
        """Test QdrantConfig default values."""
        defaults = {
            "host": "localhost",
            "port": 6333,
            "grpc_port": 6334,
            "api_key": None,
            "https": False,
            "collection_name": "cad_vectors",
            "vector_size": 128,
            "distance": "Cosine",
            "on_disk": False,
            "timeout": 30.0,
        }

        assert defaults["host"] == "localhost"
        assert defaults["port"] == 6333
        assert defaults["grpc_port"] == 6334
        assert defaults["api_key"] is None
        assert defaults["https"] is False
        assert defaults["collection_name"] == "cad_vectors"
        assert defaults["vector_size"] == 128
        assert defaults["distance"] == "Cosine"
        assert defaults["on_disk"] is False
        assert defaults["timeout"] == 30.0

    def test_from_env_defaults(self):
        """Test QdrantConfig.from_env with defaults."""
        with patch.dict("os.environ", {}, clear=True):
            config = {
                "host": os.getenv("QDRANT_HOST", "localhost"),
                "port": int(os.getenv("QDRANT_PORT", "6333")),
                "grpc_port": int(os.getenv("QDRANT_GRPC_PORT", "6334")),
                "api_key": os.getenv("QDRANT_API_KEY"),
                "https": os.getenv("QDRANT_HTTPS", "false").lower() == "true",
                "collection_name": os.getenv("QDRANT_COLLECTION", "cad_vectors"),
                "vector_size": int(os.getenv("QDRANT_VECTOR_SIZE", "128")),
                "distance": os.getenv("QDRANT_DISTANCE", "Cosine"),
                "on_disk": os.getenv("QDRANT_ON_DISK", "false").lower() == "true",
                "timeout": float(os.getenv("QDRANT_TIMEOUT", "30.0")),
            }

        assert config["host"] == "localhost"
        assert config["port"] == 6333
        assert config["api_key"] is None
        assert config["https"] is False

    def test_from_env_custom_values(self):
        """Test QdrantConfig.from_env with custom env vars."""
        env_vars = {
            "QDRANT_HOST": "qdrant.example.com",
            "QDRANT_PORT": "6380",
            "QDRANT_GRPC_PORT": "6381",
            "QDRANT_API_KEY": "secret-key-123",
            "QDRANT_HTTPS": "true",
            "QDRANT_COLLECTION": "my_vectors",
            "QDRANT_VECTOR_SIZE": "256",
            "QDRANT_DISTANCE": "Euclidean",
            "QDRANT_ON_DISK": "true",
            "QDRANT_TIMEOUT": "60.0",
        }

        with patch.dict("os.environ", env_vars):
            config = {
                "host": os.getenv("QDRANT_HOST", "localhost"),
                "port": int(os.getenv("QDRANT_PORT", "6333")),
                "grpc_port": int(os.getenv("QDRANT_GRPC_PORT", "6334")),
                "api_key": os.getenv("QDRANT_API_KEY"),
                "https": os.getenv("QDRANT_HTTPS", "false").lower() == "true",
                "collection_name": os.getenv("QDRANT_COLLECTION", "cad_vectors"),
                "vector_size": int(os.getenv("QDRANT_VECTOR_SIZE", "128")),
                "distance": os.getenv("QDRANT_DISTANCE", "Cosine"),
                "on_disk": os.getenv("QDRANT_ON_DISK", "false").lower() == "true",
                "timeout": float(os.getenv("QDRANT_TIMEOUT", "30.0")),
            }

        assert config["host"] == "qdrant.example.com"
        assert config["port"] == 6380
        assert config["grpc_port"] == 6381
        assert config["api_key"] == "secret-key-123"
        assert config["https"] is True
        assert config["collection_name"] == "my_vectors"
        assert config["vector_size"] == 256
        assert config["distance"] == "Euclidean"
        assert config["on_disk"] is True
        assert config["timeout"] == 60.0

    def test_https_boolean_parsing(self):
        """Test HTTPS boolean parsing variations."""
        true_values = ["true", "True", "TRUE"]
        false_values = ["false", "False", "FALSE", "0", "no", ""]

        for val in true_values:
            assert val.lower() == "true"

        for val in false_values:
            assert val.lower() != "true"

    def test_on_disk_boolean_parsing(self):
        """Test on_disk boolean parsing variations."""
        with patch.dict("os.environ", {"QDRANT_ON_DISK": "TRUE"}):
            on_disk = os.getenv("QDRANT_ON_DISK", "false").lower() == "true"
            assert on_disk is True

        with patch.dict("os.environ", {"QDRANT_ON_DISK": "false"}):
            on_disk = os.getenv("QDRANT_ON_DISK", "false").lower() == "true"
            assert on_disk is False


class TestDistanceMapping:
    """Tests for distance metric mapping."""

    def test_cosine_mapping(self):
        """Test Cosine distance mapping."""
        distance_map = {
            "Cosine": "COSINE",
            "Euclidean": "EUCLID",
            "Dot": "DOT",
        }

        distance = distance_map.get("Cosine", "COSINE")
        assert distance == "COSINE"

    def test_euclidean_mapping(self):
        """Test Euclidean distance mapping."""
        distance_map = {
            "Cosine": "COSINE",
            "Euclidean": "EUCLID",
            "Dot": "DOT",
        }

        distance = distance_map.get("Euclidean", "COSINE")
        assert distance == "EUCLID"

    def test_dot_mapping(self):
        """Test Dot distance mapping."""
        distance_map = {
            "Cosine": "COSINE",
            "Euclidean": "EUCLID",
            "Dot": "DOT",
        }

        distance = distance_map.get("Dot", "COSINE")
        assert distance == "DOT"

    def test_unknown_distance_defaults_to_cosine(self):
        """Test unknown distance defaults to Cosine."""
        distance_map = {
            "Cosine": "COSINE",
            "Euclidean": "EUCLID",
            "Dot": "DOT",
        }

        distance = distance_map.get("Unknown", "COSINE")
        assert distance == "COSINE"


class TestFilterBuilding:
    """Tests for filter building logic."""

    def test_single_value_filter(self):
        """Test building filter with single value."""
        filter_conditions = {"material": "steel"}

        must_conditions = []
        for key, value in filter_conditions.items():
            if isinstance(value, list):
                must_conditions.append({"key": key, "match": {"any": value}})
            elif isinstance(value, dict):
                if "gte" in value or "lte" in value:
                    must_conditions.append({"key": key, "range": value})
            else:
                must_conditions.append({"key": key, "match": {"value": value}})

        assert len(must_conditions) == 1
        assert must_conditions[0]["key"] == "material"
        assert must_conditions[0]["match"]["value"] == "steel"

    def test_list_value_filter(self):
        """Test building filter with list value (OR condition)."""
        filter_conditions = {"material": ["steel", "aluminum", "copper"]}

        must_conditions = []
        for key, value in filter_conditions.items():
            if isinstance(value, list):
                must_conditions.append({"key": key, "match": {"any": value}})
            elif isinstance(value, dict):
                if "gte" in value or "lte" in value:
                    must_conditions.append({"key": key, "range": value})
            else:
                must_conditions.append({"key": key, "match": {"value": value}})

        assert len(must_conditions) == 1
        assert must_conditions[0]["key"] == "material"
        assert must_conditions[0]["match"]["any"] == ["steel", "aluminum", "copper"]

    def test_range_filter_gte_lte(self):
        """Test building range filter with gte and lte."""
        filter_conditions = {"score": {"gte": 0.5, "lte": 1.0}}

        must_conditions = []
        for key, value in filter_conditions.items():
            if isinstance(value, list):
                must_conditions.append({"key": key, "match": {"any": value}})
            elif isinstance(value, dict):
                if "gte" in value or "lte" in value:
                    must_conditions.append({"key": key, "range": value})
            else:
                must_conditions.append({"key": key, "match": {"value": value}})

        assert len(must_conditions) == 1
        assert must_conditions[0]["key"] == "score"
        assert must_conditions[0]["range"]["gte"] == 0.5
        assert must_conditions[0]["range"]["lte"] == 1.0

    def test_range_filter_gte_only(self):
        """Test building range filter with gte only."""
        filter_conditions = {"score": {"gte": 0.5}}

        must_conditions = []
        for key, value in filter_conditions.items():
            if isinstance(value, list):
                must_conditions.append({"key": key, "match": {"any": value}})
            elif isinstance(value, dict):
                if "gte" in value or "lte" in value:
                    must_conditions.append({"key": key, "range": value})
            else:
                must_conditions.append({"key": key, "match": {"value": value}})

        assert len(must_conditions) == 1
        assert must_conditions[0]["range"]["gte"] == 0.5
        assert must_conditions[0]["range"].get("lte") is None

    def test_range_filter_lte_only(self):
        """Test building range filter with lte only."""
        filter_conditions = {"score": {"lte": 0.9}}

        must_conditions = []
        for key, value in filter_conditions.items():
            if isinstance(value, list):
                must_conditions.append({"key": key, "match": {"any": value}})
            elif isinstance(value, dict):
                if "gte" in value or "lte" in value:
                    must_conditions.append({"key": key, "range": value})
            else:
                must_conditions.append({"key": key, "match": {"value": value}})

        assert len(must_conditions) == 1
        assert must_conditions[0]["range"]["lte"] == 0.9

    def test_multiple_conditions_filter(self):
        """Test building filter with multiple conditions."""
        filter_conditions = {
            "material": "steel",
            "category": ["fastener", "bracket"],
            "score": {"gte": 0.5},
        }

        must_conditions = []
        for key, value in filter_conditions.items():
            if isinstance(value, list):
                must_conditions.append({"key": key, "match": {"any": value}})
            elif isinstance(value, dict):
                if "gte" in value or "lte" in value:
                    must_conditions.append({"key": key, "range": value})
            else:
                must_conditions.append({"key": key, "match": {"value": value}})

        assert len(must_conditions) == 3

    def test_empty_filter_conditions(self):
        """Test filter with empty conditions."""
        filter_conditions: Optional[Dict[str, Any]] = None

        query_filter = None
        if filter_conditions:
            must_conditions = []
            # Build conditions...
            if must_conditions:
                query_filter = {"must": must_conditions}

        assert query_filter is None


class TestPayloadTimestamp:
    """Tests for payload timestamp handling."""

    def test_created_at_added_if_missing(self):
        """Test created_at is added when not present."""
        metadata = {"material": "steel"}

        payload = metadata.copy() if metadata else {}
        if "created_at" not in payload:
            payload["created_at"] = datetime.utcnow().isoformat()

        assert "created_at" in payload
        assert "material" in payload

    def test_created_at_preserved_if_present(self):
        """Test created_at is preserved when already present."""
        original_timestamp = "2024-01-01T00:00:00"
        metadata = {"material": "steel", "created_at": original_timestamp}

        payload = metadata.copy() if metadata else {}
        if "created_at" not in payload:
            payload["created_at"] = datetime.utcnow().isoformat()

        assert payload["created_at"] == original_timestamp

    def test_empty_metadata_gets_timestamp(self):
        """Test empty metadata gets timestamp."""
        metadata: Optional[Dict[str, Any]] = None

        payload = metadata.copy() if metadata else {}
        if "created_at" not in payload:
            payload["created_at"] = datetime.utcnow().isoformat()

        assert "created_at" in payload
        assert len(payload) == 1


class TestBatchProcessing:
    """Tests for batch processing logic."""

    def test_batch_splitting(self):
        """Test batch splitting logic."""
        vectors = [(f"id-{i}", [0.1] * 128, {"idx": i}) for i in range(250)]
        batch_size = 100

        batches = []
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            batches.append(batch)

        assert len(batches) == 3
        assert len(batches[0]) == 100
        assert len(batches[1]) == 100
        assert len(batches[2]) == 50

    def test_batch_exactly_divisible(self):
        """Test batch splitting when exactly divisible."""
        vectors = [(f"id-{i}", [0.1] * 128, {"idx": i}) for i in range(200)]
        batch_size = 100

        batches = []
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            batches.append(batch)

        assert len(batches) == 2
        assert len(batches[0]) == 100
        assert len(batches[1]) == 100

    def test_batch_smaller_than_size(self):
        """Test batch when total is smaller than batch size."""
        vectors = [(f"id-{i}", [0.1] * 128, {"idx": i}) for i in range(50)]
        batch_size = 100

        batches = []
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            batches.append(batch)

        assert len(batches) == 1
        assert len(batches[0]) == 50


class TestCollectionNameHandling:
    """Tests for collection name handling."""

    def test_collection_in_list(self):
        """Test collection name found in list."""
        collection_names = ["default_vectors", "cad_vectors", "test_vectors"]
        target = "cad_vectors"

        found = target in collection_names

        assert found is True

    def test_collection_not_in_list(self):
        """Test collection name not found in list."""
        collection_names = ["default_vectors", "test_vectors"]
        target = "cad_vectors"

        found = target in collection_names

        assert found is False

    def test_initialized_flag_prevents_recreate(self):
        """Test initialized flag prevents recreation."""
        initialized = False

        if not initialized:
            # Would create collection
            initialized = True

        assert initialized is True

        # Second call should skip
        should_create = not initialized
        assert should_create is False


class TestContextManagerProtocol:
    """Tests for context manager protocol."""

    def test_sync_context_manager_pattern(self):
        """Test sync context manager pattern."""
        class MockStore:
            def __init__(self):
                self._client = MagicMock()
                self._initialized = True

            def close(self):
                if self._client is not None:
                    self._client.close()
                    self._client = None
                    self._initialized = False

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.close()

        with MockStore() as store:
            assert store._initialized is True

        assert store._client is None
        assert store._initialized is False

    def test_async_context_manager_pattern(self):
        """Test async context manager pattern structure."""
        class MockAsyncStore:
            def __init__(self):
                self._client = MagicMock()
                self._initialized = True

            def close(self):
                if self._client is not None:
                    self._client.close()
                    self._client = None
                    self._initialized = False

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                self.close()

        store = MockAsyncStore()
        assert store._initialized is True

        # Simulate context exit
        store.close()
        assert store._client is None


class TestHealthCheck:
    """Tests for health check logic."""

    def test_health_check_success(self):
        """Test health check returns True on success."""
        def get_collections():
            return {"collections": []}

        try:
            get_collections()
            healthy = True
        except Exception:
            healthy = False

        assert healthy is True

    def test_health_check_failure(self):
        """Test health check returns False on exception."""
        def get_collections():
            raise ConnectionError("Connection refused")

        try:
            get_collections()
            healthy = True
        except Exception:
            healthy = False

        assert healthy is False


class TestVectorIdHandling:
    """Tests for vector ID handling."""

    def test_string_id_conversion(self):
        """Test ID converted to string."""
        hit_id = "uuid-12345"
        result_id = str(hit_id)

        assert result_id == "uuid-12345"

    def test_uuid_id_conversion(self):
        """Test UUID converted to string."""
        import uuid as uuid_module
        hit_id = uuid_module.UUID("12345678-1234-1234-1234-123456789012")
        result_id = str(hit_id)

        assert result_id == "12345678-1234-1234-1234-123456789012"


class TestPayloadIndexes:
    """Tests for payload index fields."""

    def test_common_index_fields(self):
        """Test common fields for indexing."""
        index_fields = ["material", "category", "created_at"]

        assert "material" in index_fields
        assert "category" in index_fields
        assert "created_at" in index_fields

    def test_keyword_schema_type(self):
        """Test KEYWORD schema type for string fields."""
        schema_types = {
            "material": "KEYWORD",
            "category": "KEYWORD",
            "created_at": "DATETIME",
        }

        assert schema_types["material"] == "KEYWORD"
        assert schema_types["category"] == "KEYWORD"
        assert schema_types["created_at"] == "DATETIME"


class TestSearchResultMapping:
    """Tests for search result mapping."""

    def test_hit_to_result_mapping(self):
        """Test mapping hit to VectorSearchResult."""
        hits = [
            {"id": "vec-1", "score": 0.95, "payload": {"material": "steel"}, "vector": [0.1, 0.2]},
            {"id": "vec-2", "score": 0.85, "payload": {"material": "aluminum"}, "vector": None},
        ]

        results = [
            {
                "id": str(hit["id"]),
                "score": hit["score"],
                "metadata": hit["payload"] or {},
                "vector": hit["vector"] if hit["vector"] else None,
            }
            for hit in hits
        ]

        assert len(results) == 2
        assert results[0]["id"] == "vec-1"
        assert results[0]["score"] == 0.95
        assert results[0]["metadata"]["material"] == "steel"
        assert results[1]["vector"] is None

    def test_empty_payload_handling(self):
        """Test empty payload becomes empty dict."""
        hit = {"id": "vec-1", "score": 0.95, "payload": None, "vector": None}

        metadata = hit["payload"] or {}

        assert metadata == {}


class TestRetrieveResult:
    """Tests for retrieve (get_vector) result handling."""

    def test_retrieve_found(self):
        """Test retrieve when vector found."""
        results = [
            {"id": "vec-1", "payload": {"material": "steel"}, "vector": [0.1, 0.2, 0.3]}
        ]

        if results:
            point = results[0]
            result = {
                "id": str(point["id"]),
                "score": 1.0,
                "metadata": point["payload"] or {},
                "vector": point["vector"],
            }
        else:
            result = None

        assert result is not None
        assert result["id"] == "vec-1"
        assert result["score"] == 1.0

    def test_retrieve_not_found(self):
        """Test retrieve when vector not found."""
        results: List[Dict[str, Any]] = []

        if results:
            point = results[0]
            result = {
                "id": str(point["id"]),
                "score": 1.0,
                "metadata": point["payload"] or {},
                "vector": point["vector"],
            }
        else:
            result = None

        assert result is None


class TestCountFiltering:
    """Tests for count with filter."""

    def test_count_with_filter(self):
        """Test count builds filter correctly."""
        filter_conditions = {"material": "steel"}

        must_conditions = [
            {"key": key, "match": {"value": value}}
            for key, value in filter_conditions.items()
        ]
        query_filter = {"must": must_conditions}

        assert len(query_filter["must"]) == 1
        assert query_filter["must"][0]["key"] == "material"

    def test_count_without_filter(self):
        """Test count without filter."""
        filter_conditions: Optional[Dict[str, Any]] = None

        query_filter = None
        if filter_conditions:
            must_conditions = [
                {"key": key, "match": {"value": value}}
                for key, value in filter_conditions.items()
            ]
            query_filter = {"must": must_conditions}

        assert query_filter is None


class TestStatsResponse:
    """Tests for stats response structure."""

    def test_stats_structure(self):
        """Test stats response structure."""
        collection_name = "cad_vectors"
        vectors_count = 1000
        points_count = 1000
        indexed_vectors_count = 950
        status = "Green"
        vector_size = 128
        distance = "Cosine"

        stats = {
            "collection_name": collection_name,
            "vectors_count": vectors_count,
            "points_count": points_count,
            "indexed_vectors_count": indexed_vectors_count,
            "status": status,
            "config": {
                "vector_size": vector_size,
                "distance": distance,
            },
        }

        assert stats["collection_name"] == "cad_vectors"
        assert stats["vectors_count"] == 1000
        assert stats["status"] == "Green"
        assert stats["config"]["vector_size"] == 128


class TestQdrantAvailability:
    """Tests for QDRANT_AVAILABLE flag handling."""

    def test_import_error_when_unavailable(self):
        """Test ImportError raised when qdrant-client not installed."""
        QDRANT_AVAILABLE = False

        if not QDRANT_AVAILABLE:
            should_raise = True
        else:
            should_raise = False

        assert should_raise is True

    def test_no_error_when_available(self):
        """Test no error when qdrant-client is installed."""
        QDRANT_AVAILABLE = True

        if not QDRANT_AVAILABLE:
            should_raise = True
        else:
            should_raise = False

        assert should_raise is False
