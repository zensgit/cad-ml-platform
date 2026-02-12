"""Search Client Implementation.

Provides Elasticsearch and in-memory search clients.
"""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    """A search result hit."""
    id: str
    index: str
    score: float
    source: Dict[str, Any]
    highlight: Optional[Dict[str, List[str]]] = None


@dataclass
class SearchResult:
    """Search result container."""
    hits: List[SearchHit]
    total: int
    max_score: Optional[float] = None
    took_ms: int = 0
    aggregations: Optional[Dict[str, Any]] = None
    scroll_id: Optional[str] = None


class SearchClient(ABC):
    """Abstract base class for search clients."""

    @abstractmethod
    async def index(
        self,
        index: str,
        doc_id: str,
        document: Dict[str, Any],
        refresh: bool = False,
    ) -> bool:
        """Index a document.

        Args:
            index: Index name
            doc_id: Document ID
            document: Document body
            refresh: Whether to refresh immediately

        Returns:
            True if indexed
        """
        pass

    @abstractmethod
    async def bulk_index(
        self,
        index: str,
        documents: List[Tuple[str, Dict[str, Any]]],
        refresh: bool = False,
    ) -> Tuple[int, int]:
        """Bulk index documents.

        Args:
            index: Index name
            documents: List of (doc_id, document) tuples
            refresh: Whether to refresh

        Returns:
            Tuple of (success_count, error_count)
        """
        pass

    @abstractmethod
    async def get(self, index: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID.

        Args:
            index: Index name
            doc_id: Document ID

        Returns:
            Document source or None
        """
        pass

    @abstractmethod
    async def delete(self, index: str, doc_id: str) -> bool:
        """Delete a document.

        Args:
            index: Index name
            doc_id: Document ID

        Returns:
            True if deleted
        """
        pass

    @abstractmethod
    async def search(
        self,
        index: str,
        query: Dict[str, Any],
        from_: int = 0,
        size: int = 10,
        sort: Optional[List[Dict[str, Any]]] = None,
        highlight: Optional[Dict[str, Any]] = None,
        aggregations: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """Search for documents.

        Args:
            index: Index name
            query: Query DSL
            from_: Start offset
            size: Number of results
            sort: Sort specification
            highlight: Highlight configuration
            aggregations: Aggregations

        Returns:
            SearchResult
        """
        pass

    @abstractmethod
    async def count(self, index: str, query: Optional[Dict[str, Any]] = None) -> int:
        """Count documents.

        Args:
            index: Index name
            query: Optional query filter

        Returns:
            Document count
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the client connection."""
        pass


class ElasticsearchClient(SearchClient):
    """Elasticsearch client implementation."""

    def __init__(
        self,
        hosts: Optional[List[str]] = None,
        cloud_id: Optional[str] = None,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.hosts = hosts or [os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")]
        self.cloud_id = cloud_id or os.getenv("ELASTICSEARCH_CLOUD_ID")
        self.api_key = api_key or os.getenv("ELASTICSEARCH_API_KEY")
        self.username = username or os.getenv("ELASTICSEARCH_USERNAME")
        self.password = password or os.getenv("ELASTICSEARCH_PASSWORD")

        self._client: Optional[Any] = None

    async def _get_client(self) -> Any:
        """Get or create Elasticsearch client."""
        if self._client is None:
            try:
                from elasticsearch import AsyncElasticsearch

                kwargs: Dict[str, Any] = {}

                if self.cloud_id:
                    kwargs["cloud_id"] = self.cloud_id
                else:
                    kwargs["hosts"] = self.hosts

                if self.api_key:
                    kwargs["api_key"] = self.api_key
                elif self.username and self.password:
                    kwargs["basic_auth"] = (self.username, self.password)

                self._client = AsyncElasticsearch(**kwargs)
                logger.info(f"Connected to Elasticsearch at {self.hosts}")

            except ImportError:
                raise RuntimeError("elasticsearch package not installed")

        return self._client

    async def index(
        self,
        index: str,
        doc_id: str,
        document: Dict[str, Any],
        refresh: bool = False,
    ) -> bool:
        try:
            client = await self._get_client()
            await client.index(
                index=index,
                id=doc_id,
                document=document,
                refresh="wait_for" if refresh else False,
            )
            return True

        except Exception as e:
            logger.error(f"Index error: {e}")
            return False

    async def bulk_index(
        self,
        index: str,
        documents: List[Tuple[str, Dict[str, Any]]],
        refresh: bool = False,
    ) -> Tuple[int, int]:
        try:
            from elasticsearch.helpers import async_bulk

            client = await self._get_client()

            actions = [
                {
                    "_index": index,
                    "_id": doc_id,
                    "_source": document,
                }
                for doc_id, document in documents
            ]

            success, errors = await async_bulk(
                client,
                actions,
                refresh="wait_for" if refresh else False,
                raise_on_error=False,
            )

            return success, len(errors) if isinstance(errors, list) else 0

        except Exception as e:
            logger.error(f"Bulk index error: {e}")
            return 0, len(documents)

    async def get(self, index: str, doc_id: str) -> Optional[Dict[str, Any]]:
        try:
            client = await self._get_client()
            response = await client.get(index=index, id=doc_id)
            return response["_source"]

        except Exception:
            return None

    async def delete(self, index: str, doc_id: str) -> bool:
        try:
            client = await self._get_client()
            await client.delete(index=index, id=doc_id)
            return True

        except Exception as e:
            logger.error(f"Delete error: {e}")
            return False

    async def search(
        self,
        index: str,
        query: Dict[str, Any],
        from_: int = 0,
        size: int = 10,
        sort: Optional[List[Dict[str, Any]]] = None,
        highlight: Optional[Dict[str, Any]] = None,
        aggregations: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        try:
            client = await self._get_client()

            body: Dict[str, Any] = {
                "query": query,
                "from": from_,
                "size": size,
            }

            if sort:
                body["sort"] = sort

            if highlight:
                body["highlight"] = highlight

            if aggregations:
                body["aggs"] = aggregations

            response = await client.search(index=index, body=body)

            hits = [
                SearchHit(
                    id=hit["_id"],
                    index=hit["_index"],
                    score=hit.get("_score", 0.0) or 0.0,
                    source=hit["_source"],
                    highlight=hit.get("highlight"),
                )
                for hit in response["hits"]["hits"]
            ]

            total = response["hits"]["total"]
            if isinstance(total, dict):
                total = total["value"]

            return SearchResult(
                hits=hits,
                total=total,
                max_score=response["hits"].get("max_score"),
                took_ms=response.get("took", 0),
                aggregations=response.get("aggregations"),
            )

        except Exception as e:
            logger.error(f"Search error: {e}")
            return SearchResult(hits=[], total=0)

    async def count(self, index: str, query: Optional[Dict[str, Any]] = None) -> int:
        try:
            client = await self._get_client()
            body = {"query": query} if query else None
            response = await client.count(index=index, body=body)
            return response["count"]

        except Exception as e:
            logger.error(f"Count error: {e}")
            return 0

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None


class InMemorySearchClient(SearchClient):
    """In-memory search client for testing."""

    def __init__(self):
        self._indices: Dict[str, Dict[str, Dict[str, Any]]] = {}

    async def index(
        self,
        index: str,
        doc_id: str,
        document: Dict[str, Any],
        refresh: bool = False,
    ) -> bool:
        if index not in self._indices:
            self._indices[index] = {}
        self._indices[index][doc_id] = document
        return True

    async def bulk_index(
        self,
        index: str,
        documents: List[Tuple[str, Dict[str, Any]]],
        refresh: bool = False,
    ) -> Tuple[int, int]:
        success = 0
        for doc_id, document in documents:
            if await self.index(index, doc_id, document):
                success += 1
        return success, len(documents) - success

    async def get(self, index: str, doc_id: str) -> Optional[Dict[str, Any]]:
        if index in self._indices:
            return self._indices[index].get(doc_id)
        return None

    async def delete(self, index: str, doc_id: str) -> bool:
        if index in self._indices and doc_id in self._indices[index]:
            del self._indices[index][doc_id]
            return True
        return False

    async def search(
        self,
        index: str,
        query: Dict[str, Any],
        from_: int = 0,
        size: int = 10,
        sort: Optional[List[Dict[str, Any]]] = None,
        highlight: Optional[Dict[str, Any]] = None,
        aggregations: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        if index not in self._indices:
            return SearchResult(hits=[], total=0)

        # Simple matching
        hits = []
        for doc_id, doc in self._indices[index].items():
            if self._matches_query(doc, query):
                hits.append(SearchHit(
                    id=doc_id,
                    index=index,
                    score=1.0,
                    source=doc,
                ))

        total = len(hits)
        hits = hits[from_:from_ + size]

        return SearchResult(hits=hits, total=total, max_score=1.0 if hits else None)

    def _matches_query(self, doc: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """Simple query matching."""
        if "match_all" in query:
            return True

        if "match" in query:
            for field_name, match_query in query["match"].items():
                if isinstance(match_query, dict):
                    search_term = match_query.get("query", "").lower()
                else:
                    search_term = str(match_query).lower()

                doc_value = str(doc.get(field_name, "")).lower()
                if search_term not in doc_value:
                    return False
            return True

        if "term" in query:
            for field_name, term_value in query["term"].items():
                if doc.get(field_name) != term_value:
                    return False
            return True

        if "bool" in query:
            bool_query = query["bool"]

            # Must clauses
            if "must" in bool_query:
                for must_clause in bool_query["must"]:
                    if not self._matches_query(doc, must_clause):
                        return False

            # Must not clauses
            if "must_not" in bool_query:
                for must_not_clause in bool_query["must_not"]:
                    if self._matches_query(doc, must_not_clause):
                        return False

            # Should clauses (at least one must match)
            if "should" in bool_query and bool_query["should"]:
                if not any(
                    self._matches_query(doc, should_clause)
                    for should_clause in bool_query["should"]
                ):
                    return False

            return True

        return True

    async def count(self, index: str, query: Optional[Dict[str, Any]] = None) -> int:
        result = await self.search(index, query or {"match_all": {}}, size=0)
        return result.total

    async def close(self) -> None:
        self._indices.clear()


# Global search client
_search_client: Optional[SearchClient] = None


def get_search_client(client_type: str = "elasticsearch") -> SearchClient:
    """Get global search client.

    Args:
        client_type: "elasticsearch" or "memory"

    Returns:
        SearchClient instance
    """
    global _search_client

    if _search_client is None:
        if client_type == "memory":
            _search_client = InMemorySearchClient()
        else:
            _search_client = ElasticsearchClient()

    return _search_client


def set_search_client(client: SearchClient) -> None:
    """Set global search client."""
    global _search_client
    _search_client = client
