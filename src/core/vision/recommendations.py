"""
Recommendations Module - Phase 12.

Provides recommendation engine capabilities including collaborative filtering,
content-based recommendations, and similarity calculations.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar, Union

from .base import VisionDescription, VisionProvider

# ============================================================================
# Enums
# ============================================================================


class RecommendationType(Enum):
    """Types of recommendations."""

    COLLABORATIVE = "collaborative"
    CONTENT_BASED = "content_based"
    HYBRID = "hybrid"
    POPULARITY = "popularity"
    TRENDING = "trending"


class SimilarityMetric(Enum):
    """Similarity metrics."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    PEARSON = "pearson"
    JACCARD = "jaccard"
    MANHATTAN = "manhattan"


class InteractionType(Enum):
    """Types of user interactions."""

    VIEW = "view"
    CLICK = "click"
    LIKE = "like"
    PURCHASE = "purchase"
    RATING = "rating"
    SHARE = "share"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class Item:
    """An item that can be recommended."""

    item_id: str
    name: str
    category: str = ""
    features: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class User:
    """A user who receives recommendations."""

    user_id: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    history: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Interaction:
    """A user interaction with an item."""

    interaction_id: str
    user_id: str
    item_id: str
    interaction_type: InteractionType
    value: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Recommendation:
    """A single recommendation."""

    item_id: str
    score: float
    reason: str = ""
    recommendation_type: RecommendationType = RecommendationType.CONTENT_BASED
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecommendationRequest:
    """Request for recommendations."""

    user_id: str
    count: int = 10
    recommendation_type: Optional[RecommendationType] = None
    exclude_items: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecommendationResult:
    """Result containing recommendations."""

    user_id: str
    recommendations: List[Recommendation]
    generated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimilarityResult:
    """Result of similarity calculation."""

    item_a: str
    item_b: str
    similarity: float
    metric: SimilarityMetric


# ============================================================================
# Similarity Calculator
# ============================================================================


class SimilarityCalculator:
    """Calculates similarity between items or users."""

    @staticmethod
    def cosine(vec_a: List[float], vec_b: List[float]) -> float:
        """Calculate cosine similarity."""
        if len(vec_a) != len(vec_b) or len(vec_a) == 0:
            return 0.0

        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = math.sqrt(sum(a * a for a in vec_a))
        mag_b = math.sqrt(sum(b * b for b in vec_b))

        if mag_a == 0 or mag_b == 0:
            return 0.0

        return dot / (mag_a * mag_b)

    @staticmethod
    def euclidean(vec_a: List[float], vec_b: List[float]) -> float:
        """Calculate euclidean similarity (1 / (1 + distance))."""
        if len(vec_a) != len(vec_b):
            return 0.0

        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)))
        return 1.0 / (1.0 + distance)

    @staticmethod
    def pearson(vec_a: List[float], vec_b: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(vec_a) != len(vec_b) or len(vec_a) < 2:
            return 0.0

        n = len(vec_a)
        mean_a = sum(vec_a) / n
        mean_b = sum(vec_b) / n

        num = sum((a - mean_a) * (b - mean_b) for a, b in zip(vec_a, vec_b))
        den_a = math.sqrt(sum((a - mean_a) ** 2 for a in vec_a))
        den_b = math.sqrt(sum((b - mean_b) ** 2 for b in vec_b))

        if den_a == 0 or den_b == 0:
            return 0.0

        return num / (den_a * den_b)

    @staticmethod
    def jaccard(set_a: Set[Any], set_b: Set[Any]) -> float:
        """Calculate Jaccard similarity."""
        if not set_a and not set_b:
            return 0.0

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def manhattan(vec_a: List[float], vec_b: List[float]) -> float:
        """Calculate Manhattan similarity (1 / (1 + distance))."""
        if len(vec_a) != len(vec_b):
            return 0.0

        distance = sum(abs(a - b) for a, b in zip(vec_a, vec_b))
        return 1.0 / (1.0 + distance)

    def calculate(
        self,
        vec_a: Union[List[float], Set[Any]],
        vec_b: Union[List[float], Set[Any]],
        metric: SimilarityMetric = SimilarityMetric.COSINE,
    ) -> float:
        """Calculate similarity using specified metric."""
        if metric == SimilarityMetric.COSINE:
            return self.cosine(list(vec_a), list(vec_b))
        elif metric == SimilarityMetric.EUCLIDEAN:
            return self.euclidean(list(vec_a), list(vec_b))
        elif metric == SimilarityMetric.PEARSON:
            return self.pearson(list(vec_a), list(vec_b))
        elif metric == SimilarityMetric.JACCARD:
            return self.jaccard(set(vec_a), set(vec_b))
        elif metric == SimilarityMetric.MANHATTAN:
            return self.manhattan(list(vec_a), list(vec_b))
        return 0.0


# ============================================================================
# Item Store
# ============================================================================


class ItemStore(ABC):
    """Abstract item storage."""

    @abstractmethod
    def add_item(self, item: Item) -> None:
        """Add an item."""
        pass

    @abstractmethod
    def get_item(self, item_id: str) -> Optional[Item]:
        """Get an item."""
        pass

    @abstractmethod
    def get_items(self, item_ids: List[str]) -> List[Item]:
        """Get multiple items."""
        pass

    @abstractmethod
    def list_items(self, category: Optional[str] = None) -> List[Item]:
        """List all items."""
        pass


class InMemoryItemStore(ItemStore):
    """In-memory item storage."""

    def __init__(self) -> None:
        self._items: Dict[str, Item] = {}

    def add_item(self, item: Item) -> None:
        """Add an item."""
        self._items[item.item_id] = item

    def get_item(self, item_id: str) -> Optional[Item]:
        """Get an item."""
        return self._items.get(item_id)

    def get_items(self, item_ids: List[str]) -> List[Item]:
        """Get multiple items."""
        return [self._items[id_] for id_ in item_ids if id_ in self._items]

    def list_items(self, category: Optional[str] = None) -> List[Item]:
        """List all items."""
        items = list(self._items.values())
        if category:
            items = [i for i in items if i.category == category]
        return items


# ============================================================================
# Interaction Store
# ============================================================================


class InteractionStore(ABC):
    """Abstract interaction storage."""

    @abstractmethod
    def add_interaction(self, interaction: Interaction) -> None:
        """Add an interaction."""
        pass

    @abstractmethod
    def get_user_interactions(self, user_id: str) -> List[Interaction]:
        """Get interactions for a user."""
        pass

    @abstractmethod
    def get_item_interactions(self, item_id: str) -> List[Interaction]:
        """Get interactions for an item."""
        pass


class InMemoryInteractionStore(InteractionStore):
    """In-memory interaction storage."""

    def __init__(self) -> None:
        self._interactions: List[Interaction] = []
        self._by_user: Dict[str, List[Interaction]] = {}
        self._by_item: Dict[str, List[Interaction]] = {}

    def add_interaction(self, interaction: Interaction) -> None:
        """Add an interaction."""
        self._interactions.append(interaction)

        if interaction.user_id not in self._by_user:
            self._by_user[interaction.user_id] = []
        self._by_user[interaction.user_id].append(interaction)

        if interaction.item_id not in self._by_item:
            self._by_item[interaction.item_id] = []
        self._by_item[interaction.item_id].append(interaction)

    def get_user_interactions(self, user_id: str) -> List[Interaction]:
        """Get interactions for a user."""
        return self._by_user.get(user_id, [])

    def get_item_interactions(self, item_id: str) -> List[Interaction]:
        """Get interactions for an item."""
        return self._by_item.get(item_id, [])


# ============================================================================
# Recommendation Strategies
# ============================================================================


class RecommendationStrategy(ABC):
    """Abstract recommendation strategy."""

    @abstractmethod
    def recommend(
        self,
        user_id: str,
        count: int,
        exclude_items: List[str],
    ) -> List[Recommendation]:
        """Generate recommendations."""
        pass


class PopularityStrategy(RecommendationStrategy):
    """Recommends popular items."""

    def __init__(self, interaction_store: InteractionStore, item_store: ItemStore) -> None:
        self._interaction_store = interaction_store
        self._item_store = item_store

    def recommend(
        self,
        user_id: str,
        count: int,
        exclude_items: List[str],
    ) -> List[Recommendation]:
        """Generate popularity-based recommendations."""
        # Count interactions per item
        item_counts: Dict[str, int] = {}
        for item in self._item_store.list_items():
            interactions = self._interaction_store.get_item_interactions(item.item_id)
            item_counts[item.item_id] = len(interactions)

        # Sort by popularity
        sorted_items = sorted(
            item_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Filter and create recommendations
        recommendations = []
        for item_id, pop_count in sorted_items:
            if item_id in exclude_items:
                continue
            recommendations.append(
                Recommendation(
                    item_id=item_id,
                    score=float(pop_count),
                    reason="Popular item",
                    recommendation_type=RecommendationType.POPULARITY,
                )
            )
            if len(recommendations) >= count:
                break

        return recommendations


class ContentBasedStrategy(RecommendationStrategy):
    """Content-based recommendation strategy."""

    def __init__(
        self,
        item_store: ItemStore,
        interaction_store: InteractionStore,
        similarity_metric: SimilarityMetric = SimilarityMetric.JACCARD,
    ) -> None:
        self._item_store = item_store
        self._interaction_store = interaction_store
        self._similarity_metric = similarity_metric
        self._calculator = SimilarityCalculator()

    def recommend(
        self,
        user_id: str,
        count: int,
        exclude_items: List[str],
    ) -> List[Recommendation]:
        """Generate content-based recommendations."""
        # Get user's interacted items
        user_interactions = self._interaction_store.get_user_interactions(user_id)
        interacted_ids = {i.item_id for i in user_interactions}

        if not interacted_ids:
            return []

        # Get user's liked items' tags
        interacted_items = self._item_store.get_items(list(interacted_ids))
        user_tags: Set[str] = set()
        for item in interacted_items:
            user_tags.update(item.tags)

        # Score all other items by tag similarity
        all_items = self._item_store.list_items()
        scored: List[tuple[str, float]] = []

        for item in all_items:
            if item.item_id in interacted_ids or item.item_id in exclude_items:
                continue

            item_tags = set(item.tags)
            similarity = self._calculator.jaccard(user_tags, item_tags)
            if similarity > 0:
                scored.append((item.item_id, similarity))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Create recommendations
        recommendations = []
        for item_id, score in scored[:count]:
            recommendations.append(
                Recommendation(
                    item_id=item_id,
                    score=score,
                    reason="Similar to items you liked",
                    recommendation_type=RecommendationType.CONTENT_BASED,
                )
            )

        return recommendations


class CollaborativeStrategy(RecommendationStrategy):
    """Collaborative filtering strategy."""

    def __init__(
        self,
        item_store: ItemStore,
        interaction_store: InteractionStore,
    ) -> None:
        self._item_store = item_store
        self._interaction_store = interaction_store
        self._calculator = SimilarityCalculator()

    def recommend(
        self,
        user_id: str,
        count: int,
        exclude_items: List[str],
    ) -> List[Recommendation]:
        """Generate collaborative recommendations."""
        # Get user's interactions
        user_interactions = self._interaction_store.get_user_interactions(user_id)
        user_items = {i.item_id for i in user_interactions}

        if not user_items:
            return []

        # Find similar users (users who interacted with same items)
        similar_users: Dict[str, int] = {}
        for item_id in user_items:
            item_interactions = self._interaction_store.get_item_interactions(item_id)
            for interaction in item_interactions:
                if interaction.user_id != user_id:
                    similar_users[interaction.user_id] = (
                        similar_users.get(interaction.user_id, 0) + 1
                    )

        # Get items from similar users
        item_scores: Dict[str, float] = {}
        for similar_user, overlap in similar_users.items():
            their_interactions = self._interaction_store.get_user_interactions(similar_user)
            for interaction in their_interactions:
                if (
                    interaction.item_id not in user_items
                    and interaction.item_id not in exclude_items
                ):
                    item_scores[interaction.item_id] = (
                        item_scores.get(interaction.item_id, 0) + overlap * interaction.value
                    )

        # Sort and create recommendations
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)

        recommendations = []
        for item_id, score in sorted_items[:count]:
            recommendations.append(
                Recommendation(
                    item_id=item_id,
                    score=score,
                    reason="Users like you also liked this",
                    recommendation_type=RecommendationType.COLLABORATIVE,
                )
            )

        return recommendations


# ============================================================================
# Recommendation Engine
# ============================================================================


class RecommendationEngine:
    """Main recommendation engine."""

    def __init__(
        self,
        item_store: Optional[ItemStore] = None,
        interaction_store: Optional[InteractionStore] = None,
    ) -> None:
        self._item_store = item_store or InMemoryItemStore()
        self._interaction_store = interaction_store or InMemoryInteractionStore()
        self._strategies: Dict[RecommendationType, RecommendationStrategy] = {}

        # Register default strategies
        self._strategies[RecommendationType.POPULARITY] = PopularityStrategy(
            self._interaction_store, self._item_store
        )
        self._strategies[RecommendationType.CONTENT_BASED] = ContentBasedStrategy(
            self._item_store, self._interaction_store
        )
        self._strategies[RecommendationType.COLLABORATIVE] = CollaborativeStrategy(
            self._item_store, self._interaction_store
        )

    def add_item(self, item: Item) -> None:
        """Add an item."""
        self._item_store.add_item(item)

    def record_interaction(self, interaction: Interaction) -> None:
        """Record a user interaction."""
        self._interaction_store.add_interaction(interaction)

    def register_strategy(
        self,
        rec_type: RecommendationType,
        strategy: RecommendationStrategy,
    ) -> None:
        """Register a recommendation strategy."""
        self._strategies[rec_type] = strategy

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        """Generate recommendations."""
        rec_type = request.recommendation_type or RecommendationType.HYBRID

        if rec_type == RecommendationType.HYBRID:
            # Combine multiple strategies
            all_recs: List[Recommendation] = []
            for strategy in self._strategies.values():
                recs = strategy.recommend(
                    request.user_id,
                    request.count,
                    request.exclude_items,
                )
                all_recs.extend(recs)

            # Deduplicate and sort
            seen: Set[str] = set()
            unique_recs: List[Recommendation] = []
            for rec in sorted(all_recs, key=lambda r: r.score, reverse=True):
                if rec.item_id not in seen:
                    seen.add(rec.item_id)
                    unique_recs.append(rec)
                if len(unique_recs) >= request.count:
                    break

            recommendations = unique_recs
        else:
            strategy = self._strategies.get(rec_type)
            if not strategy:
                recommendations = []
            else:
                recommendations = strategy.recommend(
                    request.user_id,
                    request.count,
                    request.exclude_items,
                )

        return RecommendationResult(
            user_id=request.user_id,
            recommendations=recommendations,
        )

    def get_similar_items(
        self,
        item_id: str,
        count: int = 5,
        metric: SimilarityMetric = SimilarityMetric.JACCARD,
    ) -> List[SimilarityResult]:
        """Get similar items."""
        target = self._item_store.get_item(item_id)
        if not target:
            return []

        calculator = SimilarityCalculator()
        target_tags = set(target.tags)

        results: List[SimilarityResult] = []
        for item in self._item_store.list_items():
            if item.item_id == item_id:
                continue

            similarity = calculator.jaccard(target_tags, set(item.tags))
            if similarity > 0:
                results.append(
                    SimilarityResult(
                        item_a=item_id,
                        item_b=item.item_id,
                        similarity=similarity,
                        metric=metric,
                    )
                )

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:count]


# ============================================================================
# Recommendation Vision Provider
# ============================================================================


class RecommendationVisionProvider(VisionProvider):
    """Vision provider with recommendation capabilities."""

    def __init__(
        self,
        provider: VisionProvider,
        engine: RecommendationEngine,
        user_id: str,
    ) -> None:
        self._provider = provider
        self._engine = engine
        self._user_id = user_id

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"recommendation_{self._provider.provider_name}"

    async def analyze_image(
        self,
        image_data: bytes,
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> VisionDescription:
        """Analyze image and record interaction."""
        result = await self._provider.analyze_image(image_data, prompt, **kwargs)

        # Record the analysis as an interaction
        import hashlib

        item_id = hashlib.sha256(image_data).hexdigest()[:12]

        interaction = Interaction(
            interaction_id=f"int_{item_id}",
            user_id=self._user_id,
            item_id=item_id,
            interaction_type=InteractionType.VIEW,
        )
        self._engine.record_interaction(interaction)

        return result

    def get_recommendations(self, count: int = 5) -> RecommendationResult:
        """Get recommendations for current user."""
        request = RecommendationRequest(
            user_id=self._user_id,
            count=count,
        )
        return self._engine.recommend(request)

    def get_engine(self) -> RecommendationEngine:
        """Get the recommendation engine."""
        return self._engine


# ============================================================================
# Factory Functions
# ============================================================================


def create_recommendation_engine(
    item_store: Optional[ItemStore] = None,
    interaction_store: Optional[InteractionStore] = None,
) -> RecommendationEngine:
    """Create a recommendation engine."""
    return RecommendationEngine(item_store, interaction_store)


def create_recommendation_provider(
    provider: VisionProvider,
    engine: RecommendationEngine,
    user_id: str,
) -> RecommendationVisionProvider:
    """Create a recommendation vision provider."""
    return RecommendationVisionProvider(provider, engine, user_id)


def create_similarity_calculator() -> SimilarityCalculator:
    """Create a similarity calculator."""
    return SimilarityCalculator()


def create_item(
    item_id: str,
    name: str,
    category: str = "",
    tags: Optional[List[str]] = None,
) -> Item:
    """Create an item."""
    return Item(
        item_id=item_id,
        name=name,
        category=category,
        tags=tags or [],
    )


def create_interaction(
    user_id: str,
    item_id: str,
    interaction_type: InteractionType = InteractionType.VIEW,
    value: float = 1.0,
) -> Interaction:
    """Create an interaction."""
    import uuid

    return Interaction(
        interaction_id=str(uuid.uuid4()),
        user_id=user_id,
        item_id=item_id,
        interaction_type=interaction_type,
        value=value,
    )
