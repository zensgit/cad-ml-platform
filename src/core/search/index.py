"""Search Index Management.

Provides index creation, mapping, and lifecycle management.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.core.search.client import SearchClient, get_search_client

logger = logging.getLogger(__name__)


class FieldType(str, Enum):
    """Elasticsearch field types."""
    TEXT = "text"
    KEYWORD = "keyword"
    LONG = "long"
    INTEGER = "integer"
    FLOAT = "float"
    DOUBLE = "double"
    BOOLEAN = "boolean"
    DATE = "date"
    OBJECT = "object"
    NESTED = "nested"
    GEO_POINT = "geo_point"
    DENSE_VECTOR = "dense_vector"
    COMPLETION = "completion"


@dataclass
class FieldMapping:
    """Field mapping configuration."""
    name: str
    field_type: FieldType
    analyzer: Optional[str] = None
    search_analyzer: Optional[str] = None
    index: bool = True
    store: bool = False
    copy_to: Optional[List[str]] = None
    fields: Optional[Dict[str, "FieldMapping"]] = None
    properties: Optional[Dict[str, "FieldMapping"]] = None
    dims: Optional[int] = None  # For dense_vector

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Elasticsearch mapping dict."""
        mapping: Dict[str, Any] = {"type": self.field_type.value}

        if self.analyzer:
            mapping["analyzer"] = self.analyzer

        if self.search_analyzer:
            mapping["search_analyzer"] = self.search_analyzer

        if not self.index:
            mapping["index"] = False

        if self.store:
            mapping["store"] = True

        if self.copy_to:
            mapping["copy_to"] = self.copy_to

        if self.dims and self.field_type == FieldType.DENSE_VECTOR:
            mapping["dims"] = self.dims

        if self.fields:
            mapping["fields"] = {
                name: f.to_dict() for name, f in self.fields.items()
            }

        if self.properties:
            mapping["properties"] = {
                name: f.to_dict() for name, f in self.properties.items()
            }

        return mapping


@dataclass
class IndexSettings:
    """Index settings configuration."""
    number_of_shards: int = 1
    number_of_replicas: int = 1
    refresh_interval: str = "1s"
    max_result_window: int = 10000

    # Analysis settings
    analyzers: Optional[Dict[str, Any]] = None
    tokenizers: Optional[Dict[str, Any]] = None
    filters: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Elasticsearch settings dict."""
        settings: Dict[str, Any] = {
            "index": {
                "number_of_shards": self.number_of_shards,
                "number_of_replicas": self.number_of_replicas,
                "refresh_interval": self.refresh_interval,
                "max_result_window": self.max_result_window,
            }
        }

        if self.analyzers or self.tokenizers or self.filters:
            analysis: Dict[str, Any] = {}
            if self.analyzers:
                analysis["analyzer"] = self.analyzers
            if self.tokenizers:
                analysis["tokenizer"] = self.tokenizers
            if self.filters:
                analysis["filter"] = self.filters
            settings["analysis"] = analysis

        return settings


@dataclass
class IndexMapping:
    """Complete index mapping."""
    fields: Dict[str, FieldMapping] = field(default_factory=dict)
    settings: IndexSettings = field(default_factory=IndexSettings)
    dynamic: str = "strict"  # strict, true, false

    def add_field(
        self,
        name: str,
        field_type: FieldType,
        **kwargs: Any,
    ) -> "IndexMapping":
        """Add a field to the mapping."""
        self.fields[name] = FieldMapping(name=name, field_type=field_type, **kwargs)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Elasticsearch index body."""
        return {
            "settings": self.settings.to_dict(),
            "mappings": {
                "dynamic": self.dynamic,
                "properties": {
                    name: f.to_dict() for name, f in self.fields.items()
                },
            },
        }


class IndexManager:
    """Manager for Elasticsearch indices."""

    def __init__(self, client: Optional[SearchClient] = None, prefix: str = ""):
        self._client = client
        self._prefix = prefix
        self._index_cache: Dict[str, bool] = {}

    @property
    def client(self) -> SearchClient:
        if self._client is None:
            self._client = get_search_client()
        return self._client

    def _full_index_name(self, name: str) -> str:
        """Get full index name with prefix."""
        return f"{self._prefix}{name}" if self._prefix else name

    async def create_index(
        self,
        name: str,
        mapping: IndexMapping,
        exist_ok: bool = True,
    ) -> bool:
        """Create an index.

        Args:
            name: Index name
            mapping: Index mapping
            exist_ok: Don't error if index exists

        Returns:
            True if created
        """
        full_name = self._full_index_name(name)

        try:
            # Check if client is Elasticsearch
            if hasattr(self.client, "_get_client"):
                es = await self.client._get_client()

                if await es.indices.exists(index=full_name):
                    if exist_ok:
                        logger.debug(f"Index {full_name} already exists")
                        return True
                    return False

                await es.indices.create(index=full_name, body=mapping.to_dict())
                logger.info(f"Created index: {full_name}")
                self._index_cache[full_name] = True
                return True

            else:
                # In-memory client
                logger.info(f"Created index (in-memory): {full_name}")
                return True

        except Exception as e:
            logger.error(f"Create index error: {e}")
            return False

    async def delete_index(self, name: str) -> bool:
        """Delete an index.

        Args:
            name: Index name

        Returns:
            True if deleted
        """
        full_name = self._full_index_name(name)

        try:
            if hasattr(self.client, "_get_client"):
                es = await self.client._get_client()
                await es.indices.delete(index=full_name)
                logger.info(f"Deleted index: {full_name}")

            self._index_cache.pop(full_name, None)
            return True

        except Exception as e:
            logger.error(f"Delete index error: {e}")
            return False

    async def index_exists(self, name: str) -> bool:
        """Check if index exists.

        Args:
            name: Index name

        Returns:
            True if exists
        """
        full_name = self._full_index_name(name)

        if full_name in self._index_cache:
            return self._index_cache[full_name]

        try:
            if hasattr(self.client, "_get_client"):
                es = await self.client._get_client()
                exists = await es.indices.exists(index=full_name)
                self._index_cache[full_name] = exists
                return exists

            return True  # In-memory always exists

        except Exception:
            return False

    async def update_mapping(
        self,
        name: str,
        mapping: Dict[str, FieldMapping],
    ) -> bool:
        """Update index mapping (add new fields).

        Args:
            name: Index name
            mapping: New field mappings

        Returns:
            True if updated
        """
        full_name = self._full_index_name(name)

        try:
            if hasattr(self.client, "_get_client"):
                es = await self.client._get_client()

                properties = {
                    field_name: field_mapping.to_dict()
                    for field_name, field_mapping in mapping.items()
                }

                await es.indices.put_mapping(
                    index=full_name,
                    body={"properties": properties},
                )
                logger.info(f"Updated mapping for index: {full_name}")
                return True

            return True

        except Exception as e:
            logger.error(f"Update mapping error: {e}")
            return False

    async def refresh_index(self, name: str) -> bool:
        """Refresh an index.

        Args:
            name: Index name

        Returns:
            True if refreshed
        """
        full_name = self._full_index_name(name)

        try:
            if hasattr(self.client, "_get_client"):
                es = await self.client._get_client()
                await es.indices.refresh(index=full_name)
                return True
            return True

        except Exception as e:
            logger.error(f"Refresh error: {e}")
            return False

    async def get_index_stats(self, name: str) -> Dict[str, Any]:
        """Get index statistics.

        Args:
            name: Index name

        Returns:
            Index stats
        """
        full_name = self._full_index_name(name)

        try:
            if hasattr(self.client, "_get_client"):
                es = await self.client._get_client()
                stats = await es.indices.stats(index=full_name)
                return stats["indices"][full_name]

            return {}

        except Exception as e:
            logger.error(f"Get stats error: {e}")
            return {}

    async def reindex(
        self,
        source_index: str,
        dest_index: str,
        query: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Reindex documents from one index to another.

        Args:
            source_index: Source index name
            dest_index: Destination index name
            query: Optional filter query

        Returns:
            True if reindex started
        """
        source_full = self._full_index_name(source_index)
        dest_full = self._full_index_name(dest_index)

        try:
            if hasattr(self.client, "_get_client"):
                es = await self.client._get_client()

                body: Dict[str, Any] = {
                    "source": {"index": source_full},
                    "dest": {"index": dest_full},
                }

                if query:
                    body["source"]["query"] = query

                await es.reindex(body=body, wait_for_completion=False)
                logger.info(f"Started reindex from {source_full} to {dest_full}")
                return True

            return True

        except Exception as e:
            logger.error(f"Reindex error: {e}")
            return False


# Global index manager
_index_manager: Optional[IndexManager] = None


def get_index_manager() -> IndexManager:
    """Get global index manager."""
    global _index_manager
    if _index_manager is None:
        _index_manager = IndexManager()
    return _index_manager


# ============================================================================
# Predefined Index Mappings
# ============================================================================

def create_document_index_mapping() -> IndexMapping:
    """Create index mapping for CAD documents."""
    mapping = IndexMapping()

    mapping.add_field("id", FieldType.KEYWORD)
    mapping.add_field("name", FieldType.TEXT, analyzer="standard", fields={
        "keyword": FieldMapping(name="keyword", field_type=FieldType.KEYWORD),
    })
    mapping.add_field("file_path", FieldType.KEYWORD)
    mapping.add_field("file_type", FieldType.KEYWORD)
    mapping.add_field("status", FieldType.KEYWORD)
    mapping.add_field("file_size", FieldType.LONG)
    mapping.add_field("checksum", FieldType.KEYWORD)

    mapping.add_field("content", FieldType.TEXT, analyzer="standard")
    mapping.add_field("tags", FieldType.KEYWORD)

    mapping.add_field("owner_id", FieldType.KEYWORD)
    mapping.add_field("tenant_id", FieldType.KEYWORD)

    mapping.add_field("created_at", FieldType.DATE)
    mapping.add_field("updated_at", FieldType.DATE)
    mapping.add_field("processed_at", FieldType.DATE)

    mapping.add_field("metadata", FieldType.OBJECT)

    return mapping


def create_model_index_mapping() -> IndexMapping:
    """Create index mapping for ML models."""
    mapping = IndexMapping()

    mapping.add_field("id", FieldType.KEYWORD)
    mapping.add_field("name", FieldType.TEXT, fields={
        "keyword": FieldMapping(name="keyword", field_type=FieldType.KEYWORD),
    })
    mapping.add_field("model_type", FieldType.KEYWORD)
    mapping.add_field("version", FieldType.KEYWORD)
    mapping.add_field("status", FieldType.KEYWORD)
    mapping.add_field("framework", FieldType.KEYWORD)

    mapping.add_field("accuracy", FieldType.FLOAT)
    mapping.add_field("precision", FieldType.FLOAT)
    mapping.add_field("recall", FieldType.FLOAT)
    mapping.add_field("f1_score", FieldType.FLOAT)

    mapping.add_field("owner_id", FieldType.KEYWORD)
    mapping.add_field("tenant_id", FieldType.KEYWORD)

    mapping.add_field("created_at", FieldType.DATE)
    mapping.add_field("deployed_at", FieldType.DATE)

    mapping.add_field("tags", FieldType.KEYWORD)

    return mapping
