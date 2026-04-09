"""Qdrant vector store implementation.

This module provides a production-ready vector store using Qdrant,
replacing the custom FAISS + Redis implementation in similarity.py.

Benefits over custom implementation:
- Native persistence and replication
- Built-in metadata filtering
- Automatic index management
- No manual lock handling
- ~90% less code to maintain

Example:
    >>> store = QdrantVectorStore()
    >>> await store.register_vector("doc-1", [0.1, 0.2, ...], {"material": "steel"})
    >>> results = await store.search_similar([0.1, 0.2, ...], top_k=5)
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from src.core.similarity import extract_vector_label_contract

logger = logging.getLogger(__name__)

# Conditional import for Qdrant
try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.http.exceptions import UnexpectedResponse

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None
    models = None
    UnexpectedResponse = Exception


@dataclass
class VectorSearchResult:
    """Result from vector similarity search."""

    id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[List[float]] = None


@dataclass
class QdrantConfig:
    """Configuration for Qdrant vector store."""

    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    api_key: Optional[str] = None
    https: bool = False
    collection_name: str = "cad_vectors"
    vector_size: int = 128
    distance: str = "Cosine"
    on_disk: bool = False
    timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "QdrantConfig":
        """Create config from environment variables."""
        return cls(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
            api_key=os.getenv("QDRANT_API_KEY"),
            https=os.getenv("QDRANT_HTTPS", "false").lower() == "true",
            collection_name=os.getenv("QDRANT_COLLECTION", "cad_vectors"),
            vector_size=int(os.getenv("QDRANT_VECTOR_SIZE", "128")),
            distance=os.getenv("QDRANT_DISTANCE", "Cosine"),
            on_disk=os.getenv("QDRANT_ON_DISK", "false").lower() == "true",
            timeout=float(os.getenv("QDRANT_TIMEOUT", "30.0")),
        )


class QdrantVectorStore:
    """Qdrant-based vector store for CAD document similarity.

    This implementation replaces the ~1000 lines of custom code in similarity.py
    with a production-ready solution that handles:
    - Persistence (no manual Redis sync)
    - Metadata filtering (native support)
    - Scalability (horizontal scaling, sharding)
    - High availability (replication)

    Attributes:
        config: Qdrant configuration
        client: Qdrant client instance
    """

    def __init__(self, config: Optional[QdrantConfig] = None):
        """Initialize Qdrant vector store.

        Args:
            config: Qdrant configuration. If None, loads from environment.

        Raises:
            ImportError: If qdrant-client is not installed.
            ConnectionError: If cannot connect to Qdrant server.
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is not installed. " "Install with: pip install qdrant-client"
            )

        self.config = config or QdrantConfig.from_env()
        self._client: Optional[QdrantClient] = None
        self._initialized = False

    @property
    def client(self) -> QdrantClient:
        """Get or create Qdrant client."""
        if self._client is None:
            self._client = QdrantClient(
                host=self.config.host,
                port=self.config.port,
                grpc_port=self.config.grpc_port,
                api_key=self.config.api_key,
                https=self.config.https,
                timeout=self.config.timeout,
            )
        return self._client

    def _ensure_collection(self) -> None:
        """Ensure collection exists with correct configuration."""
        if self._initialized:
            return

        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.config.collection_name not in collection_names:
            logger.info(f"Creating Qdrant collection: {self.config.collection_name}")

            # Map distance metric
            distance_map = {
                "Cosine": models.Distance.COSINE,
                "Euclidean": models.Distance.EUCLID,
                "Dot": models.Distance.DOT,
            }
            distance = distance_map.get(self.config.distance, models.Distance.COSINE)

            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=models.VectorParams(
                    size=self.config.vector_size,
                    distance=distance,
                    on_disk=self.config.on_disk,
                ),
            )

            # Create payload indexes for common filters
            for keyword_field in (
                "material",
                "category",
                "part_type",
                "fine_part_type",
                "coarse_part_type",
                "decision_source",
            ):
                self.client.create_payload_index(
                    collection_name=self.config.collection_name,
                    field_name=keyword_field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            self.client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="created_at",
                field_schema=models.PayloadSchemaType.DATETIME,
            )

        self._initialized = True

    @staticmethod
    def _normalize_payload_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        payload = metadata.copy() if metadata else {}
        contract = extract_vector_label_contract(payload)
        for key, value in contract.items():
            if value is None:
                continue
            payload[key] = value
        if contract.get("decision_source") and "final_decision_source" not in payload:
            payload["final_decision_source"] = contract["decision_source"]
        return payload

    @staticmethod
    def _build_query_filter(filter_conditions: Optional[Dict[str, Any]]):
        if not filter_conditions:
            return None

        must_conditions = []
        for key, value in filter_conditions.items():
            if value is None:
                continue
            if isinstance(value, list):
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value),
                    )
                )
            elif isinstance(value, dict):
                if "gte" in value or "lte" in value:
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            range=models.Range(
                                gte=value.get("gte"),
                                lte=value.get("lte"),
                            ),
                        )
                    )
            else:
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )

        if not must_conditions:
            return None
        return models.Filter(must=must_conditions)

    @staticmethod
    def _to_result(hit: Any, with_vectors: bool = False) -> VectorSearchResult:
        return VectorSearchResult(
            id=str(hit.id),
            score=getattr(hit, "score", 1.0),
            metadata=getattr(hit, "payload", None) or {},
            vector=getattr(hit, "vector", None) if with_vectors else None,
        )

    def health_check(self) -> bool:
        """Check if Qdrant is healthy and accessible.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            # Try to get collections as a health check
            self.client.get_collections()
            return True
        except Exception as e:
            logger.warning(f"Qdrant health check failed: {e}")
            return False

    async def register_vector(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register a vector with metadata.

        This replaces ~100 lines of lock management, FAISS indexing,
        and Redis sync code in similarity.py.

        Args:
            vector_id: Unique identifier for the vector.
            vector: Feature vector (list of floats).
            metadata: Optional metadata (material, category, etc.).

        Returns:
            True if successful.

        Example:
            >>> await store.register_vector(
            ...     "doc-123",
            ...     [0.1, 0.2, 0.3, ...],
            ...     {"material": "steel", "category": "fastener"}
            ... )
        """
        self._ensure_collection()

        # Add timestamp if not present
        payload = self._normalize_payload_metadata(metadata)
        if "created_at" not in payload:
            payload["created_at"] = datetime.utcnow().isoformat()

        try:
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=[
                    models.PointStruct(
                        id=vector_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
            logger.debug(f"Registered vector: {vector_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register vector {vector_id}: {e}")
            raise

    async def register_vectors_batch(
        self,
        vectors: List[tuple],
        batch_size: int = 100,
    ) -> int:
        """Register multiple vectors in batches.

        Args:
            vectors: List of (id, vector, metadata) tuples.
            batch_size: Number of vectors per batch.

        Returns:
            Number of vectors registered.
        """
        self._ensure_collection()

        total = 0
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            points = []

            for vector_id, vector, metadata in batch:
                payload = self._normalize_payload_metadata(metadata)
                if "created_at" not in payload:
                    payload["created_at"] = datetime.utcnow().isoformat()

                points.append(
                    models.PointStruct(
                        id=vector_id,
                        vector=vector,
                        payload=payload,
                    )
                )

            self.client.upsert(
                collection_name=self.config.collection_name,
                points=points,
            )
            total += len(points)

        logger.info(f"Registered {total} vectors in batches")
        return total

    async def search_similar(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        with_vectors: bool = False,
    ) -> List[VectorSearchResult]:
        """Search for similar vectors.

        This replaces ~150 lines of FAISS search, metadata filtering,
        and result merging code in similarity.py.

        Args:
            query_vector: Query feature vector.
            top_k: Number of results to return.
            filter_conditions: Optional metadata filters.
            score_threshold: Minimum similarity score (0-1 for cosine).
            with_vectors: Include vectors in results.

        Returns:
            List of VectorSearchResult sorted by similarity.

        Example:
            >>> results = await store.search_similar(
            ...     [0.1, 0.2, ...],
            ...     top_k=5,
            ...     filter_conditions={"material": "steel"}
            ... )
            >>> for r in results:
            ...     print(f"{r.id}: {r.score:.3f}")
        """
        self._ensure_collection()

        query_filter = self._build_query_filter(filter_conditions)

        try:
            results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold,
                with_vectors=with_vectors,
            )
            return [self._to_result(hit, with_vectors=with_vectors) for hit in results]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    async def list_vectors(
        self,
        offset: int = 0,
        limit: int = 50,
        filter_conditions: Optional[Dict[str, Any]] = None,
        with_vectors: bool = False,
    ) -> tuple[List[VectorSearchResult], int]:
        """List vectors with optional server-side filtering."""
        self._ensure_collection()
        query_filter = self._build_query_filter(filter_conditions)

        try:
            total = self.client.count(
                collection_name=self.config.collection_name,
                count_filter=query_filter,
            ).count
            scroll_limit = max(offset + limit, limit)
            points, _ = self.client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=query_filter,
                limit=scroll_limit,
                with_payload=True,
                with_vectors=with_vectors,
            )
            items = points[offset : offset + limit]
            return [self._to_result(point, with_vectors=with_vectors) for point in items], total
        except Exception as e:
            logger.error(f"List failed: {e}")
            raise

    async def get_vector(self, vector_id: str) -> Optional[VectorSearchResult]:
        """Get a specific vector by ID.

        Args:
            vector_id: Vector identifier.

        Returns:
            VectorSearchResult or None if not found.
        """
        self._ensure_collection()

        try:
            results = self.client.retrieve(
                collection_name=self.config.collection_name,
                ids=[vector_id],
                with_vectors=True,
            )

            if results:
                point = results[0]
                return VectorSearchResult(
                    id=str(point.id),
                    score=1.0,
                    metadata=point.payload or {},
                    vector=point.vector,
                )
            return None
        except Exception as e:
            logger.error(f"Failed to get vector {vector_id}: {e}")
            return None

    async def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector by ID.

        Args:
            vector_id: Vector identifier.

        Returns:
            True if successful.
        """
        self._ensure_collection()

        try:
            self.client.delete(
                collection_name=self.config.collection_name,
                points_selector=models.PointIdsList(points=[vector_id]),
            )
            logger.debug(f"Deleted vector: {vector_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete vector {vector_id}: {e}")
            return False

    async def delete_vectors_by_filter(self, filter_conditions: Dict[str, Any]) -> int:
        """Delete vectors matching filter conditions.

        Args:
            filter_conditions: Metadata filter conditions.

        Returns:
            Number of vectors deleted (approximate).
        """
        self._ensure_collection()

        query_filter = self._build_query_filter(filter_conditions)

        try:
            # Get count before delete
            count_before = self.client.count(
                collection_name=self.config.collection_name,
                count_filter=query_filter,
            ).count

            self.client.delete(
                collection_name=self.config.collection_name,
                points_selector=models.FilterSelector(filter=query_filter),
            )

            logger.info(f"Deleted ~{count_before} vectors by filter")
            return count_before
        except Exception as e:
            logger.error(f"Failed to delete vectors by filter: {e}")
            raise

    async def update_metadata(self, vector_id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for a vector.

        Args:
            vector_id: Vector identifier.
            metadata: New metadata to merge.

        Returns:
            True if successful.
        """
        self._ensure_collection()

        try:
            metadata = self._normalize_payload_metadata(metadata)
            self.client.set_payload(
                collection_name=self.config.collection_name,
                payload=metadata,
                points=[vector_id],
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update metadata for {vector_id}: {e}")
            return False

    async def count(self, filter_conditions: Optional[Dict[str, Any]] = None) -> int:
        """Count vectors, optionally with filter.

        Args:
            filter_conditions: Optional metadata filter.

        Returns:
            Number of vectors.
        """
        self._ensure_collection()

        query_filter = self._build_query_filter(filter_conditions)

        result = self.client.count(
            collection_name=self.config.collection_name,
            count_filter=query_filter,
        )
        return result.count

    async def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics.

        Returns:
            Dictionary with collection stats.
        """
        self._ensure_collection()

        info = self.client.get_collection(self.config.collection_name)

        return {
            "collection_name": self.config.collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status.name,
            "config": {
                "vector_size": self.config.vector_size,
                "distance": self.config.distance,
            },
        }

    async def inspect_collection(self) -> Dict[str, Any]:
        """Inspect collection state without creating or mutating it."""
        requested_config = {
            "vector_size": self.config.vector_size,
            "distance": self.config.distance,
            "on_disk": self.config.on_disk,
            "timeout_seconds": self.config.timeout,
        }
        snapshot: Dict[str, Any] = {
            "enabled": True,
            "sdk_available": True,
            "reachable": False,
            "collection_name": self.config.collection_name,
            "collection_exists": False,
            "collection_status": None,
            "points_count": 0,
            "vectors_count": 0,
            "indexed_vectors_count": 0,
            "unindexed_vectors_count": 0,
            "indexing_progress": 0.0,
            "requested_config": requested_config,
            "error": None,
        }
        try:
            collections = self.client.get_collections().collections
            snapshot["reachable"] = True
            collection_names = {item.name for item in collections}
            if self.config.collection_name not in collection_names:
                return snapshot

            snapshot["collection_exists"] = True
            info = self.client.get_collection(self.config.collection_name)
            points_count = int(info.points_count or 0)
            vectors_count = int(info.vectors_count or points_count)
            indexed_vectors_count = int(info.indexed_vectors_count or 0)
            snapshot.update(
                {
                    "collection_status": str(info.status.name).lower(),
                    "points_count": points_count,
                    "vectors_count": vectors_count,
                    "indexed_vectors_count": indexed_vectors_count,
                    "unindexed_vectors_count": max(points_count - indexed_vectors_count, 0),
                    "indexing_progress": (
                        round(indexed_vectors_count / points_count, 4) if points_count else 0.0
                    ),
                }
            )
            return snapshot
        except Exception as exc:
            snapshot["error"] = str(exc)
            return snapshot

    def close(self) -> None:
        """Close the client connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._initialized = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()
