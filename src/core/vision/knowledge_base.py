"""Knowledge Base Module.

Provides knowledge graphs, semantic search, and intelligent information retrieval.
"""

import hashlib
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .base import VisionDescription, VisionProvider


class EntityType(Enum):
    """Types of entities in the knowledge graph."""

    CONCEPT = "concept"
    OBJECT = "object"
    ATTRIBUTE = "attribute"
    ACTION = "action"
    MEASUREMENT = "measurement"
    COMPONENT = "component"
    MATERIAL = "material"
    PROCESS = "process"


class RelationType(Enum):
    """Types of relationships between entities."""

    IS_A = "is_a"
    HAS_A = "has_a"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    SIMILAR_TO = "similar_to"
    DERIVED_FROM = "derived_from"
    DEPENDS_ON = "depends_on"
    CAUSES = "causes"
    USED_FOR = "used_for"
    MEASURED_BY = "measured_by"


class SearchStrategy(Enum):
    """Search strategies."""

    EXACT = "exact"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"
    GRAPH_TRAVERSAL = "graph_traversal"
    HYBRID = "hybrid"


@dataclass
class Entity:
    """Entity in the knowledge graph."""

    entity_id: str
    entity_type: EntityType
    name: str
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class Relationship:
    """Relationship between entities."""

    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SearchResult:
    """Result of a knowledge search."""

    entity: Entity
    score: float
    path: Optional[List[str]] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Result of a knowledge query."""

    results: List[SearchResult]
    total_count: int
    query_time_ms: float
    strategy_used: SearchStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeEntry:
    """Entry in the knowledge base."""

    entry_id: str
    title: str
    content: str
    tags: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)  # Entity IDs
    embeddings: Optional[List[float]] = None
    source: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class KnowledgeGraph:
    """Graph structure for knowledge representation."""

    def __init__(self):
        self._entities: Dict[str, Entity] = {}
        self._relationships: Dict[str, List[Relationship]] = defaultdict(list)
        self._reverse_relationships: Dict[str, List[Relationship]] = defaultdict(list)
        self._type_index: Dict[EntityType, Set[str]] = defaultdict(set)
        self._name_index: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.Lock()

    def add_entity(self, entity: Entity) -> bool:
        """Add an entity to the graph."""
        with self._lock:
            if entity.entity_id in self._entities:
                return False

            self._entities[entity.entity_id] = entity
            self._type_index[entity.entity_type].add(entity.entity_id)

            # Index by name words
            for word in entity.name.lower().split():
                self._name_index[word].add(entity.entity_id)

            return True

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        with self._lock:
            return self._entities.get(entity_id)

    def update_entity(self, entity: Entity) -> bool:
        """Update an existing entity."""
        with self._lock:
            if entity.entity_id not in self._entities:
                return False

            old_entity = self._entities[entity.entity_id]

            # Update type index if changed
            if old_entity.entity_type != entity.entity_type:
                self._type_index[old_entity.entity_type].discard(entity.entity_id)
                self._type_index[entity.entity_type].add(entity.entity_id)

            # Update name index
            for word in old_entity.name.lower().split():
                self._name_index[word].discard(entity.entity_id)
            for word in entity.name.lower().split():
                self._name_index[word].add(entity.entity_id)

            entity.updated_at = datetime.now()
            self._entities[entity.entity_id] = entity
            return True

    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity and its relationships."""
        with self._lock:
            if entity_id not in self._entities:
                return False

            entity = self._entities.pop(entity_id)
            self._type_index[entity.entity_type].discard(entity_id)

            for word in entity.name.lower().split():
                self._name_index[word].discard(entity_id)

            # Remove relationships
            self._relationships.pop(entity_id, None)
            self._reverse_relationships.pop(entity_id, None)

            # Clean up references in other relationships
            for source_id in list(self._relationships.keys()):
                self._relationships[source_id] = [
                    r for r in self._relationships[source_id] if r.target_id != entity_id
                ]

            return True

    def add_relationship(self, relationship: Relationship) -> bool:
        """Add a relationship between entities."""
        with self._lock:
            if (
                relationship.source_id not in self._entities
                or relationship.target_id not in self._entities
            ):
                return False

            self._relationships[relationship.source_id].append(relationship)
            self._reverse_relationships[relationship.target_id].append(relationship)
            return True

    def get_relationships(self, entity_id: str, direction: str = "outgoing") -> List[Relationship]:
        """Get relationships for an entity."""
        with self._lock:
            if direction == "incoming":
                return list(self._reverse_relationships.get(entity_id, []))
            elif direction == "both":
                outgoing = list(self._relationships.get(entity_id, []))
                incoming = list(self._reverse_relationships.get(entity_id, []))
                return outgoing + incoming
            return list(self._relationships.get(entity_id, []))

    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get all entities of a type."""
        with self._lock:
            entity_ids = self._type_index.get(entity_type, set())
            return [self._entities[eid] for eid in entity_ids if eid in self._entities]

    def search_by_name(self, query: str) -> List[Entity]:
        """Search entities by name."""
        with self._lock:
            matches: Set[str] = set()
            for word in query.lower().split():
                matches.update(self._name_index.get(word, set()))
            return [self._entities[eid] for eid in matches if eid in self._entities]

    def traverse(
        self,
        start_id: str,
        max_depth: int = 3,
        relation_types: Optional[List[RelationType]] = None,
    ) -> List[Tuple[Entity, int, List[str]]]:
        """Traverse the graph from a starting entity."""
        with self._lock:
            if start_id not in self._entities:
                return []

            visited: Set[str] = set()
            results: List[Tuple[Entity, int, List[str]]] = []
            queue: List[Tuple[str, int, List[str]]] = [(start_id, 0, [start_id])]

            while queue:
                current_id, depth, path = queue.pop(0)

                if current_id in visited or depth > max_depth:
                    continue

                visited.add(current_id)
                entity = self._entities.get(current_id)
                if entity:
                    results.append((entity, depth, path))

                if depth < max_depth:
                    for rel in self._relationships.get(current_id, []):
                        if relation_types and rel.relation_type not in relation_types:
                            continue
                        if rel.target_id not in visited:
                            queue.append((rel.target_id, depth + 1, path + [rel.target_id]))

            return results

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        with self._lock:
            total_relationships = sum(len(rels) for rels in self._relationships.values())
            return {
                "total_entities": len(self._entities),
                "total_relationships": total_relationships,
                "entities_by_type": {t.value: len(ids) for t, ids in self._type_index.items()},
            }


class SemanticIndex:
    """Semantic search index using embeddings."""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self._embeddings: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def add_embedding(self, entity_id: str, embedding: List[float]) -> bool:
        """Add an embedding for an entity."""
        if len(embedding) != self.dimension:
            return False

        with self._lock:
            self._embeddings[entity_id] = embedding
            return True

    def remove_embedding(self, entity_id: str) -> bool:
        """Remove an embedding."""
        with self._lock:
            return self._embeddings.pop(entity_id, None) is not None

    def search(self, query_embedding: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar entities."""
        if len(query_embedding) != self.dimension:
            return []

        with self._lock:
            scores = []
            for entity_id, embedding in self._embeddings.items():
                score = self._cosine_similarity(query_embedding, embedding)
                scores.append((entity_id, score))

            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)


class KnowledgeStore:
    """Store for knowledge entries."""

    def __init__(self):
        self._entries: Dict[str, KnowledgeEntry] = {}
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.Lock()

    def add_entry(self, entry: KnowledgeEntry) -> bool:
        """Add a knowledge entry."""
        with self._lock:
            if entry.entry_id in self._entries:
                return False

            self._entries[entry.entry_id] = entry
            for tag in entry.tags:
                self._tag_index[tag.lower()].add(entry.entry_id)
            return True

    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Get an entry by ID."""
        with self._lock:
            return self._entries.get(entry_id)

    def update_entry(self, entry: KnowledgeEntry) -> bool:
        """Update an entry."""
        with self._lock:
            if entry.entry_id not in self._entries:
                return False

            old_entry = self._entries[entry.entry_id]
            for tag in old_entry.tags:
                self._tag_index[tag.lower()].discard(entry.entry_id)

            entry.updated_at = datetime.now()
            self._entries[entry.entry_id] = entry
            for tag in entry.tags:
                self._tag_index[tag.lower()].add(entry.entry_id)

            return True

    def search_by_tags(self, tags: List[str]) -> List[KnowledgeEntry]:
        """Search entries by tags."""
        with self._lock:
            matching_ids: Optional[Set[str]] = None
            for tag in tags:
                tag_ids = self._tag_index.get(tag.lower(), set())
                if matching_ids is None:
                    matching_ids = tag_ids.copy()
                else:
                    matching_ids &= tag_ids

            if not matching_ids:
                return []

            return [self._entries[eid] for eid in matching_ids if eid in self._entries]

    def search_by_content(self, query: str) -> List[KnowledgeEntry]:
        """Simple content search."""
        query_lower = query.lower()
        with self._lock:
            results = []
            for entry in self._entries.values():
                if query_lower in entry.title.lower() or query_lower in entry.content.lower():
                    results.append(entry)
            return results

    def get_all_tags(self) -> List[str]:
        """Get all unique tags."""
        with self._lock:
            return list(self._tag_index.keys())


class KnowledgeBase:
    """Main knowledge base engine."""

    def __init__(
        self,
        embedding_dimension: int = 128,
    ):
        self._graph = KnowledgeGraph()
        self._semantic_index = SemanticIndex(dimension=embedding_dimension)
        self._store = KnowledgeStore()
        self._embedding_function: Optional[Callable[[str], List[float]]] = None
        self._lock = threading.Lock()

    def set_embedding_function(self, func: Callable[[str], List[float]]) -> None:
        """Set the function for generating embeddings."""
        self._embedding_function = func

    def add_entity(
        self,
        entity_id: str,
        entity_type: EntityType,
        name: str,
        description: str = "",
        properties: Optional[Dict[str, Any]] = None,
    ) -> Entity:
        """Add an entity to the knowledge base."""
        entity = Entity(
            entity_id=entity_id,
            entity_type=entity_type,
            name=name,
            description=description,
            properties=properties or {},
        )

        # Generate embeddings if function available
        if self._embedding_function and description:
            try:
                embedding = self._embedding_function(f"{name} {description}")
                entity.embeddings = embedding
                self._semantic_index.add_embedding(entity_id, embedding)
            except Exception:
                pass

        self._graph.add_entity(entity)
        return entity

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        weight: float = 1.0,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add a relationship between entities."""
        relationship = Relationship(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            properties=properties or {},
        )
        return self._graph.add_relationship(relationship)

    def add_knowledge(
        self,
        entry_id: str,
        title: str,
        content: str,
        tags: Optional[List[str]] = None,
        source: str = "",
    ) -> KnowledgeEntry:
        """Add a knowledge entry."""
        entry = KnowledgeEntry(
            entry_id=entry_id,
            title=title,
            content=content,
            tags=tags or [],
            source=source,
        )

        if self._embedding_function:
            try:
                embedding = self._embedding_function(f"{title} {content}")
                entry.embeddings = embedding
            except Exception:
                pass

        self._store.add_entry(entry)
        return entry

    def search(
        self,
        query: str,
        strategy: SearchStrategy = SearchStrategy.HYBRID,
        limit: int = 10,
    ) -> QueryResult:
        """Search the knowledge base."""
        start_time = time.time()
        results: List[SearchResult] = []

        if strategy in (SearchStrategy.EXACT, SearchStrategy.HYBRID):
            # Exact name matching
            entities = self._graph.search_by_name(query)
            for entity in entities[:limit]:
                results.append(SearchResult(entity=entity, score=1.0))

        if strategy in (SearchStrategy.FUZZY, SearchStrategy.HYBRID):
            # Fuzzy search on knowledge entries
            entries = self._store.search_by_content(query)
            for entry in entries[:limit]:
                # Convert entry to pseudo-entity
                entity = Entity(
                    entity_id=entry.entry_id,
                    entity_type=EntityType.CONCEPT,
                    name=entry.title,
                    description=entry.content,
                )
                results.append(SearchResult(entity=entity, score=0.8))

        if strategy in (SearchStrategy.SEMANTIC, SearchStrategy.HYBRID):
            # Semantic search
            if self._embedding_function:
                try:
                    query_embedding = self._embedding_function(query)
                    semantic_results = self._semantic_index.search(query_embedding, top_k=limit)
                    for entity_id, score in semantic_results:
                        entity = self._graph.get_entity(entity_id)
                        if entity and score > 0.5:
                            results.append(SearchResult(entity=entity, score=score))
                except Exception:
                    pass

        if strategy == SearchStrategy.GRAPH_TRAVERSAL:
            # Find starting entity and traverse
            name_matches = self._graph.search_by_name(query)
            if name_matches:
                traversal = self._graph.traverse(name_matches[0].entity_id, max_depth=2)
                for entity, depth, path in traversal:
                    score = 1.0 / (1.0 + depth)
                    results.append(SearchResult(entity=entity, score=score, path=path))

        # Deduplicate and sort
        seen: Set[str] = set()
        unique_results = []
        for result in results:
            if result.entity.entity_id not in seen:
                seen.add(result.entity.entity_id)
                unique_results.append(result)

        unique_results.sort(key=lambda r: r.score, reverse=True)
        unique_results = unique_results[:limit]

        query_time = (time.time() - start_time) * 1000

        return QueryResult(
            results=unique_results,
            total_count=len(unique_results),
            query_time_ms=query_time,
            strategy_used=strategy,
        )

    def get_related(
        self, entity_id: str, relation_types: Optional[List[RelationType]] = None
    ) -> List[Entity]:
        """Get related entities."""
        relationships = self._graph.get_relationships(entity_id, direction="both")
        if relation_types:
            relationships = [r for r in relationships if r.relation_type in relation_types]

        related_ids = set()
        for rel in relationships:
            if rel.source_id == entity_id:
                related_ids.add(rel.target_id)
            else:
                related_ids.add(rel.source_id)

        return [self._graph.get_entity(eid) for eid in related_ids if self._graph.get_entity(eid)]

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        return self._graph.get_entity(entity_id)

    def get_knowledge(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Get a knowledge entry by ID."""
        return self._store.get_entry(entry_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        graph_stats = self._graph.get_stats()
        return {
            **graph_stats,
            "total_knowledge_entries": len(self._store._entries),
            "total_tags": len(self._store.get_all_tags()),
        }


class KnowledgeEnhancedVisionProvider(VisionProvider):
    """Vision provider enhanced with knowledge base."""

    def __init__(
        self,
        provider: VisionProvider,
        knowledge_base: Optional[KnowledgeBase] = None,
    ):
        self._provider = provider
        self.knowledge_base = knowledge_base or KnowledgeBase()

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"knowledge_enhanced_{self._provider.provider_name}"

    async def analyze_image(
        self, image_data: bytes, include_description: bool = True, **kwargs: Any
    ) -> VisionDescription:
        """Analyze image with knowledge enhancement."""
        # Get base analysis
        result = await self._provider.analyze_image(image_data, include_description, **kwargs)

        # Enhance with knowledge
        if include_description and result.summary:
            enhanced_details = list(result.details)

            # Search knowledge base for related information
            query_result = self.knowledge_base.search(
                result.summary, strategy=SearchStrategy.HYBRID, limit=3
            )

            for search_result in query_result.results:
                if search_result.score > 0.6:
                    enhanced_details.append(f"Related: {search_result.entity.name}")

            return VisionDescription(
                summary=result.summary,
                details=enhanced_details,
                confidence=result.confidence,
            )

        return result

    def add_knowledge(
        self,
        title: str,
        content: str,
        tags: Optional[List[str]] = None,
    ) -> KnowledgeEntry:
        """Add knowledge to the base."""
        entry_id = hashlib.md5(f"{title}{content}".encode()).hexdigest()[:16]
        return self.knowledge_base.add_knowledge(entry_id, title, content, tags)

    def search_knowledge(self, query: str, limit: int = 10) -> QueryResult:
        """Search the knowledge base."""
        return self.knowledge_base.search(query, limit=limit)


# Factory functions
def create_knowledge_graph() -> KnowledgeGraph:
    """Create a knowledge graph."""
    return KnowledgeGraph()


def create_knowledge_base(
    embedding_dimension: int = 128,
) -> KnowledgeBase:
    """Create a knowledge base."""
    return KnowledgeBase(embedding_dimension=embedding_dimension)


def create_knowledge_provider(
    provider: VisionProvider,
    knowledge_base: Optional[KnowledgeBase] = None,
) -> KnowledgeEnhancedVisionProvider:
    """Create a knowledge-enhanced vision provider."""
    return KnowledgeEnhancedVisionProvider(provider, knowledge_base)


def create_entity(
    entity_id: str,
    entity_type: EntityType,
    name: str,
    description: str = "",
) -> Entity:
    """Create an entity."""
    return Entity(
        entity_id=entity_id,
        entity_type=entity_type,
        name=name,
        description=description,
    )


def create_relationship(
    source_id: str,
    target_id: str,
    relation_type: RelationType,
    weight: float = 1.0,
) -> Relationship:
    """Create a relationship."""
    return Relationship(
        source_id=source_id,
        target_id=target_id,
        relation_type=relation_type,
        weight=weight,
    )
