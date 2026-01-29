"""
Knowledge Base Management Module.

Provides interfaces for managing, indexing, and querying
the CAD knowledge base with CRUD operations.
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .semantic_retrieval import SemanticRetriever, SemanticSearchResult


class KnowledgeCategory(Enum):
    """Categories for knowledge items."""

    MATERIALS = "materials"
    WELDING = "welding"
    GDT = "gdt"
    MACHINING = "machining"
    SURFACE_TREATMENT = "surface_treatment"
    FASTENERS = "fasteners"
    ASSEMBLY = "assembly"
    GENERAL = "general"


@dataclass
class KnowledgeItem:
    """A single knowledge item."""

    id: str
    content: str
    category: KnowledgeCategory
    source: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category.value,
            "source": self.source,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeItem":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            category=KnowledgeCategory(data["category"]),
            source=data.get("source", ""),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
        )


@dataclass
class KnowledgeStats:
    """Statistics about the knowledge base."""

    total_items: int
    items_by_category: Dict[str, int]
    items_by_source: Dict[str, int]
    top_tags: List[tuple]
    last_updated: float


class KnowledgeBaseManager:
    """
    Manager for the CAD knowledge base.

    Provides CRUD operations, search, and analytics.

    Example:
        >>> manager = KnowledgeBaseManager()
        >>> item_id = manager.add_item(
        ...     content="304不锈钢的抗拉强度约为520MPa",
        ...     category=KnowledgeCategory.MATERIALS,
        ...     tags=["不锈钢", "强度"]
        ... )
        >>> results = manager.search("不锈钢强度")
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        vector_path: Optional[str] = None,
    ):
        """
        Initialize knowledge base manager.

        Args:
            storage_path: Path for knowledge item storage
            vector_path: Path for vector index storage
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self._items: Dict[str, KnowledgeItem] = {}
        self._tags_index: Dict[str, Set[str]] = {}  # tag -> item_ids
        self._category_index: Dict[KnowledgeCategory, Set[str]] = {}

        # Initialize semantic retriever
        self._retriever = SemanticRetriever(storage_path=vector_path)

        # Load existing data
        if self.storage_path and self.storage_path.exists():
            self._load()

    # CRUD Operations

    def add_item(
        self,
        content: str,
        category: KnowledgeCategory,
        source: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        item_id: Optional[str] = None,
    ) -> str:
        """
        Add a knowledge item.

        Args:
            content: Knowledge content
            category: Category classification
            source: Source identifier
            tags: List of tags
            metadata: Additional metadata
            item_id: Optional custom ID

        Returns:
            Item ID
        """
        item_id = item_id or str(uuid.uuid4())[:8]
        tags = tags or []

        item = KnowledgeItem(
            id=item_id,
            content=content,
            category=category,
            source=source,
            tags=tags,
            metadata=metadata or {},
        )

        self._items[item_id] = item
        self._index_item(item)

        # Add to vector store
        self._retriever.index_text(
            content,
            source=source or category.value,
            metadata={"id": item_id, "category": category.value},
        )

        return item_id

    def add_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Add multiple knowledge items.

        Args:
            items: List of item dictionaries

        Returns:
            List of item IDs
        """
        ids = []
        texts = []
        sources = []
        metadata_list = []

        for item_data in items:
            item_id = item_data.get("id") or str(uuid.uuid4())[:8]
            category = KnowledgeCategory(item_data.get("category", "general"))

            item = KnowledgeItem(
                id=item_id,
                content=item_data["content"],
                category=category,
                source=item_data.get("source", ""),
                tags=item_data.get("tags", []),
                metadata=item_data.get("metadata", {}),
            )

            self._items[item_id] = item
            self._index_item(item)
            ids.append(item_id)

            texts.append(item.content)
            sources.append(item.source or category.value)
            metadata_list.append({"id": item_id, "category": category.value})

        # Batch index to vector store
        self._retriever.index_batch(texts, sources, metadata_list)

        return ids

    def get_item(self, item_id: str) -> Optional[KnowledgeItem]:
        """Get a knowledge item by ID."""
        return self._items.get(item_id)

    def update_item(
        self,
        item_id: str,
        content: Optional[str] = None,
        category: Optional[KnowledgeCategory] = None,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update a knowledge item.

        Args:
            item_id: Item ID
            content: New content (optional)
            category: New category (optional)
            source: New source (optional)
            tags: New tags (optional)
            metadata: New metadata (optional)

        Returns:
            True if updated successfully
        """
        if item_id not in self._items:
            return False

        item = self._items[item_id]

        # Remove from old indices
        self._unindex_item(item)

        # Update fields
        if content is not None:
            item.content = content
        if category is not None:
            item.category = category
        if source is not None:
            item.source = source
        if tags is not None:
            item.tags = tags
        if metadata is not None:
            item.metadata.update(metadata)

        item.updated_at = time.time()

        # Re-index
        self._index_item(item)

        return True

    def delete_item(self, item_id: str) -> bool:
        """Delete a knowledge item."""
        if item_id not in self._items:
            return False

        item = self._items[item_id]
        self._unindex_item(item)
        del self._items[item_id]

        return True

    # Search Operations

    def search(
        self,
        query: str,
        top_k: int = 10,
        category_filter: Optional[KnowledgeCategory] = None,
        tag_filter: Optional[List[str]] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge base.

        Args:
            query: Search query
            top_k: Maximum results
            category_filter: Filter by category
            tag_filter: Filter by tags
            min_score: Minimum similarity score

        Returns:
            List of search results with items
        """
        # Get semantic results
        source_filter = category_filter.value if category_filter else None
        results = self._retriever.search(
            query,
            top_k=top_k * 2,  # Get more for filtering
            min_score=min_score,
            source_filter=source_filter,
        )

        # Enrich with full item data
        enriched = []
        for result in results:
            item_id = result.metadata.get("id")
            if item_id and item_id in self._items:
                item = self._items[item_id]

                # Apply tag filter
                if tag_filter and not any(t in item.tags for t in tag_filter):
                    continue

                enriched.append({
                    "item": item.to_dict(),
                    "score": result.score,
                    "source": result.source,
                })

                if len(enriched) >= top_k:
                    break

        return enriched

    def search_by_tags(
        self,
        tags: List[str],
        match_all: bool = False,
    ) -> List[KnowledgeItem]:
        """
        Search by tags.

        Args:
            tags: Tags to search for
            match_all: Require all tags to match

        Returns:
            Matching items
        """
        if not tags:
            return []

        matching_ids: Optional[Set[str]] = None

        for tag in tags:
            tag_items = self._tags_index.get(tag, set())
            if matching_ids is None:
                matching_ids = tag_items.copy()
            elif match_all:
                matching_ids &= tag_items
            else:
                matching_ids |= tag_items

        if not matching_ids:
            return []

        return [self._items[id] for id in matching_ids if id in self._items]

    def get_by_category(
        self,
        category: KnowledgeCategory,
        limit: int = 100,
    ) -> List[KnowledgeItem]:
        """Get items by category."""
        item_ids = self._category_index.get(category, set())
        items = [self._items[id] for id in item_ids if id in self._items]
        return items[:limit]

    # Analytics

    def get_stats(self) -> KnowledgeStats:
        """Get knowledge base statistics."""
        items_by_category: Dict[str, int] = {}
        items_by_source: Dict[str, int] = {}
        tag_counts: Dict[str, int] = {}

        for item in self._items.values():
            # Category counts
            cat = item.category.value
            items_by_category[cat] = items_by_category.get(cat, 0) + 1

            # Source counts
            src = item.source or "unknown"
            items_by_source[src] = items_by_source.get(src, 0) + 1

            # Tag counts
            for tag in item.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Top tags
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Last updated
        last_updated = max(
            (item.updated_at for item in self._items.values()),
            default=0,
        )

        return KnowledgeStats(
            total_items=len(self._items),
            items_by_category=items_by_category,
            items_by_source=items_by_source,
            top_tags=top_tags,
            last_updated=last_updated,
        )

    def list_categories(self) -> List[Dict[str, Any]]:
        """List all categories with counts."""
        result = []
        for category in KnowledgeCategory:
            count = len(self._category_index.get(category, set()))
            result.append({
                "category": category.value,
                "count": count,
            })
        return result

    def list_tags(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List tags with counts."""
        tag_counts = [
            {"tag": tag, "count": len(ids)}
            for tag, ids in self._tags_index.items()
        ]
        tag_counts.sort(key=lambda x: x["count"], reverse=True)
        return tag_counts[:limit]

    # Persistence

    def save(self) -> bool:
        """Save knowledge base to disk."""
        if not self.storage_path:
            return False

        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "items": [item.to_dict() for item in self._items.values()],
                "saved_at": time.time(),
            }
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Save vector index
            self._retriever.save()
            return True
        except IOError:
            return False

    def _load(self) -> bool:
        """Load knowledge base from disk."""
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item_data in data.get("items", []):
                item = KnowledgeItem.from_dict(item_data)
                self._items[item.id] = item
                self._index_item(item)

            return True
        except (IOError, json.JSONDecodeError):
            return False

    def clear(self) -> None:
        """Clear all knowledge items."""
        self._items.clear()
        self._tags_index.clear()
        self._category_index.clear()
        self._retriever.clear()

    # Indexing helpers

    def _index_item(self, item: KnowledgeItem) -> None:
        """Add item to indices."""
        # Tags index
        for tag in item.tags:
            if tag not in self._tags_index:
                self._tags_index[tag] = set()
            self._tags_index[tag].add(item.id)

        # Category index
        if item.category not in self._category_index:
            self._category_index[item.category] = set()
        self._category_index[item.category].add(item.id)

    def _unindex_item(self, item: KnowledgeItem) -> None:
        """Remove item from indices."""
        # Tags index
        for tag in item.tags:
            if tag in self._tags_index:
                self._tags_index[tag].discard(item.id)

        # Category index
        if item.category in self._category_index:
            self._category_index[item.category].discard(item.id)

    @property
    def item_count(self) -> int:
        """Get total item count."""
        return len(self._items)


def create_knowledge_manager(
    storage_path: str = ".cad_assistant/knowledge.json",
    vector_path: str = ".cad_assistant/knowledge_vectors.json",
) -> KnowledgeBaseManager:
    """
    Factory function to create a knowledge base manager.

    Args:
        storage_path: Path for knowledge storage
        vector_path: Path for vector storage

    Returns:
        Configured KnowledgeBaseManager
    """
    return KnowledgeBaseManager(
        storage_path=storage_path,
        vector_path=vector_path,
    )
