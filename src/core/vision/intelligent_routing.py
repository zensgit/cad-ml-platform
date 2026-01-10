"""Intelligent Routing Module.

Provides ML-based request routing with adaptive decision making.
"""

import asyncio
import hashlib
import random
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .base import VisionDescription, VisionProvider


class RoutingStrategy(Enum):
    """Routing strategies."""

    ROUND_ROBIN = "round_robin"
    LEAST_LATENCY = "least_latency"
    LEAST_LOAD = "least_load"
    WEIGHTED = "weighted"
    ADAPTIVE = "adaptive"
    ML_BASED = "ml_based"
    CONTENT_BASED = "content_based"
    COST_OPTIMIZED = "cost_optimized"


class ProviderCapability(Enum):
    """Provider capabilities."""

    HIGH_ACCURACY = "high_accuracy"
    LOW_LATENCY = "low_latency"
    COST_EFFECTIVE = "cost_effective"
    HIGH_THROUGHPUT = "high_throughput"
    SPECIALIZED_OCR = "specialized_ocr"
    MULTI_LANGUAGE = "multi_language"
    LARGE_IMAGES = "large_images"


class RequestPriority(Enum):
    """Request priority levels."""

    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class ProviderStats:
    """Statistics for a provider."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency: float = 0.0
    current_load: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def average_latency(self) -> float:
        """Calculate average latency."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency / self.successful_requests


@dataclass
class RoutingDecision:
    """Result of a routing decision."""

    selected_provider: str
    strategy_used: RoutingStrategy
    confidence: float
    alternatives: List[str] = field(default_factory=list)
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingContext:
    """Context for routing decisions."""

    image_size: int = 0
    content_type: str = ""
    priority: RequestPriority = RequestPriority.NORMAL
    required_capabilities: List[ProviderCapability] = field(default_factory=list)
    max_latency_ms: Optional[float] = None
    max_cost: Optional[float] = None
    preferred_providers: List[str] = field(default_factory=list)
    excluded_providers: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderConfig:
    """Configuration for a provider."""

    name: str
    weight: float = 1.0
    max_concurrent: int = 100
    capabilities: List[ProviderCapability] = field(default_factory=list)
    cost_per_request: float = 0.0
    typical_latency_ms: float = 100.0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class RoutingRule(ABC):
    """Abstract base class for routing rules."""

    @abstractmethod
    def evaluate(
        self,
        providers: Dict[str, ProviderConfig],
        stats: Dict[str, ProviderStats],
        context: RoutingContext,
    ) -> Optional[str]:
        """Evaluate rule and return provider or None."""
        pass

    @abstractmethod
    def get_priority(self) -> int:
        """Get rule priority (lower = higher priority)."""
        pass


class CapabilityMatchRule(RoutingRule):
    """Route based on required capabilities."""

    def evaluate(
        self,
        providers: Dict[str, ProviderConfig],
        stats: Dict[str, ProviderStats],
        context: RoutingContext,
    ) -> Optional[str]:
        """Find provider with required capabilities."""
        if not context.required_capabilities:
            return None

        matching = []
        for name, config in providers.items():
            if not config.enabled:
                continue
            if name in context.excluded_providers:
                continue

            has_all = all(cap in config.capabilities for cap in context.required_capabilities)
            if has_all:
                matching.append(name)

        return matching[0] if matching else None

    def get_priority(self) -> int:
        return 1


class LatencyConstraintRule(RoutingRule):
    """Route based on latency constraints."""

    def evaluate(
        self,
        providers: Dict[str, ProviderConfig],
        stats: Dict[str, ProviderStats],
        context: RoutingContext,
    ) -> Optional[str]:
        """Find provider meeting latency constraint."""
        if not context.max_latency_ms:
            return None

        candidates = []
        for name, config in providers.items():
            if not config.enabled:
                continue
            if name in context.excluded_providers:
                continue

            provider_stats = stats.get(name, ProviderStats())
            estimated_latency = (
                provider_stats.average_latency
                if provider_stats.total_requests > 10
                else config.typical_latency_ms
            )

            if estimated_latency <= context.max_latency_ms:
                candidates.append((name, estimated_latency))

        if candidates:
            # Return fastest
            candidates.sort(key=lambda x: x[1])
            return candidates[0][0]

        return None

    def get_priority(self) -> int:
        return 2


class CostConstraintRule(RoutingRule):
    """Route based on cost constraints."""

    def evaluate(
        self,
        providers: Dict[str, ProviderConfig],
        stats: Dict[str, ProviderStats],
        context: RoutingContext,
    ) -> Optional[str]:
        """Find provider meeting cost constraint."""
        if not context.max_cost:
            return None

        candidates = []
        for name, config in providers.items():
            if not config.enabled:
                continue
            if name in context.excluded_providers:
                continue

            if config.cost_per_request <= context.max_cost:
                candidates.append((name, config.cost_per_request))

        if candidates:
            # Return cheapest
            candidates.sort(key=lambda x: x[1])
            return candidates[0][0]

        return None

    def get_priority(self) -> int:
        return 3


class LoadBalancer:
    """Load balancer for providers."""

    def __init__(self):
        self._stats: Dict[str, ProviderStats] = defaultdict(ProviderStats)
        self._lock = threading.Lock()

    def record_request_start(self, provider: str) -> None:
        """Record request start."""
        with self._lock:
            self._stats[provider].current_load += 1
            self._stats[provider].total_requests += 1
            self._stats[provider].last_updated = datetime.now()

    def record_request_end(self, provider: str, latency: float, success: bool) -> None:
        """Record request completion."""
        with self._lock:
            stats = self._stats[provider]
            stats.current_load = max(0, stats.current_load - 1)
            if success:
                stats.successful_requests += 1
                stats.total_latency += latency
            else:
                stats.failed_requests += 1
            stats.last_updated = datetime.now()

    def get_stats(self, provider: str) -> ProviderStats:
        """Get stats for a provider."""
        with self._lock:
            return self._stats.get(provider, ProviderStats())

    def get_all_stats(self) -> Dict[str, ProviderStats]:
        """Get all provider stats."""
        with self._lock:
            return dict(self._stats)

    def get_least_loaded(self, providers: List[str]) -> Optional[str]:
        """Get provider with least load."""
        with self._lock:
            if not providers:
                return None

            min_load = float("inf")
            selected = None

            for provider in providers:
                stats = self._stats.get(provider, ProviderStats())
                if stats.current_load < min_load:
                    min_load = stats.current_load
                    selected = provider

            return selected


class AdaptiveRouter:
    """Adaptive router that learns from historical data."""

    def __init__(self, learning_rate: float = 0.1, decay_factor: float = 0.99):
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self._scores: Dict[str, float] = defaultdict(lambda: 1.0)
        self._lock = threading.Lock()

    def update_score(self, provider: str, success: bool, latency: float) -> None:
        """Update provider score based on outcome."""
        with self._lock:
            current = self._scores[provider]

            # Calculate performance score
            if success:
                # Lower latency = higher score (normalized)
                latency_score = max(0, 1 - (latency / 5.0))  # 5s as baseline
                delta = self.learning_rate * (latency_score - current)
            else:
                # Penalize failures
                delta = -self.learning_rate * 0.5

            self._scores[provider] = max(0.1, min(2.0, current + delta))

    def decay_scores(self) -> None:
        """Apply decay to scores (call periodically)."""
        with self._lock:
            for provider in self._scores:
                self._scores[provider] = 1.0 + (self._scores[provider] - 1.0) * self.decay_factor

    def get_best_provider(self, available: List[str]) -> Optional[str]:
        """Get provider with best adaptive score."""
        with self._lock:
            if not available:
                return None

            best_score = -1
            best_provider = None

            for provider in available:
                score = self._scores.get(provider, 1.0)
                if score > best_score:
                    best_score = score
                    best_provider = provider

            return best_provider

    def get_scores(self) -> Dict[str, float]:
        """Get all provider scores."""
        with self._lock:
            return dict(self._scores)


class ContentAnalyzer:
    """Analyzes content for routing decisions."""

    def __init__(self):
        self._patterns: Dict[str, str] = {}  # Pattern -> preferred provider

    def analyze(self, image_data: bytes, context: RoutingContext) -> Dict[str, Any]:
        """Analyze content characteristics."""
        analysis = {
            "size": len(image_data),
            "size_category": self._categorize_size(len(image_data)),
            "hash": hashlib.sha256(image_data[:1024]).hexdigest()[:8],
        }

        # Detect image type from magic bytes
        if image_data[:8] == b"\x89PNG\r\n\x1a\n":
            analysis["format"] = "png"
        elif image_data[:2] == b"\xff\xd8":
            analysis["format"] = "jpeg"
        elif image_data[:6] in (b"GIF87a", b"GIF89a"):
            analysis["format"] = "gif"
        elif image_data[:4] == b"RIFF" and image_data[8:12] == b"WEBP":
            analysis["format"] = "webp"
        elif image_data[:4] == b"%PDF":
            analysis["format"] = "pdf"
        else:
            analysis["format"] = "unknown"

        return analysis

    def _categorize_size(self, size: int) -> str:
        """Categorize image size."""
        if size < 100_000:
            return "small"
        elif size < 1_000_000:
            return "medium"
        elif size < 10_000_000:
            return "large"
        return "very_large"

    def get_recommended_provider(
        self, analysis: Dict[str, Any], providers: List[str]
    ) -> Optional[str]:
        """Get recommended provider based on content analysis."""
        size_cat = analysis.get("size_category", "medium")
        format_type = analysis.get("format", "unknown")

        # Simple heuristics
        if size_cat == "very_large":
            # Prefer providers good with large images
            for p in providers:
                if "large" in p.lower() or "batch" in p.lower():
                    return p

        if format_type == "pdf":
            # Prefer OCR-specialized providers
            for p in providers:
                if "ocr" in p.lower() or "document" in p.lower():
                    return p

        return None


class IntelligentRouter:
    """Main intelligent routing engine."""

    def __init__(
        self,
        default_strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE,
    ):
        self.default_strategy = default_strategy
        self._providers: Dict[str, ProviderConfig] = {}
        self._load_balancer = LoadBalancer()
        self._adaptive_router = AdaptiveRouter()
        self._content_analyzer = ContentAnalyzer()
        self._rules: List[RoutingRule] = []
        self._round_robin_index = 0
        self._lock = threading.Lock()

        # Add default rules
        self._rules.append(CapabilityMatchRule())
        self._rules.append(LatencyConstraintRule())
        self._rules.append(CostConstraintRule())

    def register_provider(self, config: ProviderConfig) -> None:
        """Register a provider."""
        with self._lock:
            self._providers[config.name] = config

    def unregister_provider(self, name: str) -> None:
        """Unregister a provider."""
        with self._lock:
            self._providers.pop(name, None)

    def add_rule(self, rule: RoutingRule) -> None:
        """Add a routing rule."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.get_priority())

    def route(
        self,
        context: Optional[RoutingContext] = None,
        image_data: Optional[bytes] = None,
        strategy: Optional[RoutingStrategy] = None,
    ) -> RoutingDecision:
        """Make a routing decision."""
        context = context or RoutingContext()
        strategy = strategy or self.default_strategy

        with self._lock:
            available = [
                name
                for name, config in self._providers.items()
                if config.enabled and name not in context.excluded_providers
            ]

        if not available:
            return RoutingDecision(
                selected_provider="",
                strategy_used=strategy,
                confidence=0.0,
                reasoning="No available providers",
            )

        # Apply preference filter
        if context.preferred_providers:
            preferred_available = [p for p in available if p in context.preferred_providers]
            if preferred_available:
                available = preferred_available

        # Check rules first
        stats = self._load_balancer.get_all_stats()
        for rule in self._rules:
            with self._lock:
                result = rule.evaluate(self._providers, stats, context)
            if result and result in available:
                return RoutingDecision(
                    selected_provider=result,
                    strategy_used=strategy,
                    confidence=0.9,
                    alternatives=[p for p in available if p != result][:3],
                    reasoning=f"Selected by {rule.__class__.__name__}",
                )

        # Apply strategy
        selected = self._apply_strategy(strategy, available, context, image_data)

        return RoutingDecision(
            selected_provider=selected,
            strategy_used=strategy,
            confidence=0.8 if selected else 0.0,
            alternatives=[p for p in available if p != selected][:3],
            reasoning=f"Selected by {strategy.value} strategy",
        )

    def _apply_strategy(
        self,
        strategy: RoutingStrategy,
        available: List[str],
        context: RoutingContext,
        image_data: Optional[bytes],
    ) -> str:
        """Apply routing strategy."""
        if not available:
            return ""

        if strategy == RoutingStrategy.ROUND_ROBIN:
            with self._lock:
                self._round_robin_index = (self._round_robin_index + 1) % len(available)
                return available[self._round_robin_index]

        elif strategy == RoutingStrategy.LEAST_LATENCY:
            stats = self._load_balancer.get_all_stats()
            best = min(
                available,
                key=lambda p: stats.get(p, ProviderStats()).average_latency or float("inf"),
            )
            return best

        elif strategy == RoutingStrategy.LEAST_LOAD:
            return self._load_balancer.get_least_loaded(available) or available[0]

        elif strategy == RoutingStrategy.WEIGHTED:
            with self._lock:
                weights = [self._providers.get(p, ProviderConfig(name=p)).weight for p in available]
            total = sum(weights)
            if total == 0:
                return random.choice(available)
            r = random.random() * total
            cumulative = 0
            for provider, weight in zip(available, weights):
                cumulative += weight
                if r <= cumulative:
                    return provider
            return available[-1]

        elif strategy == RoutingStrategy.ADAPTIVE:
            return self._adaptive_router.get_best_provider(available) or available[0]

        elif strategy == RoutingStrategy.CONTENT_BASED:
            if image_data:
                analysis = self._content_analyzer.analyze(image_data, context)
                recommended = self._content_analyzer.get_recommended_provider(analysis, available)
                if recommended:
                    return recommended
            return self._adaptive_router.get_best_provider(available) or available[0]

        elif strategy == RoutingStrategy.COST_OPTIMIZED:
            with self._lock:
                best = min(
                    available,
                    key=lambda p: self._providers.get(p, ProviderConfig(name=p)).cost_per_request,
                )
            return best

        # Default: ML-based combines adaptive + content
        return self._adaptive_router.get_best_provider(available) or available[0]

    def record_result(self, provider: str, success: bool, latency: float) -> None:
        """Record request result for learning."""
        self._load_balancer.record_request_end(provider, latency, success)
        self._adaptive_router.update_score(provider, success, latency)

    def get_provider_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all providers."""
        stats = self._load_balancer.get_all_stats()
        scores = self._adaptive_router.get_scores()

        result = {}
        for name in self._providers:
            provider_stats = stats.get(name, ProviderStats())
            result[name] = {
                "total_requests": provider_stats.total_requests,
                "success_rate": provider_stats.success_rate,
                "average_latency": provider_stats.average_latency,
                "current_load": provider_stats.current_load,
                "adaptive_score": scores.get(name, 1.0),
            }
        return result


class RoutedVisionProvider(VisionProvider):
    """Vision provider with intelligent routing."""

    def __init__(
        self,
        providers: Dict[str, VisionProvider],
        router: Optional[IntelligentRouter] = None,
    ):
        self._providers = providers
        self.router = router or IntelligentRouter()

        # Register providers with router
        for name in providers:
            self.router.register_provider(ProviderConfig(name=name))

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return "routed"

    async def analyze_image(
        self, image_data: bytes, include_description: bool = True, **kwargs: Any
    ) -> VisionDescription:
        """Analyze image with intelligent routing."""
        context = kwargs.pop("routing_context", None) or RoutingContext(image_size=len(image_data))
        strategy = kwargs.pop("routing_strategy", None)

        decision = self.router.route(context, image_data, strategy)

        if not decision.selected_provider:
            raise RuntimeError("No provider available for request")

        provider = self._providers.get(decision.selected_provider)
        if not provider:
            raise RuntimeError(f"Provider {decision.selected_provider} not found")

        start_time = time.time()
        success = True

        try:
            self.router._load_balancer.record_request_start(decision.selected_provider)
            result = await provider.analyze_image(image_data, include_description, **kwargs)
            return result
        except Exception as e:
            success = False
            raise
        finally:
            latency = time.time() - start_time
            self.router.record_result(decision.selected_provider, success, latency)

    def get_routing_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get routing statistics."""
        return self.router.get_provider_stats()


# Factory functions
def create_intelligent_router(
    default_strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE,
) -> IntelligentRouter:
    """Create an intelligent router."""
    return IntelligentRouter(default_strategy=default_strategy)


def create_routed_provider(
    providers: Dict[str, VisionProvider],
    router: Optional[IntelligentRouter] = None,
) -> RoutedVisionProvider:
    """Create a routed vision provider."""
    return RoutedVisionProvider(providers, router)


def create_provider_config(
    name: str,
    weight: float = 1.0,
    capabilities: Optional[List[ProviderCapability]] = None,
    cost_per_request: float = 0.0,
    typical_latency_ms: float = 100.0,
) -> ProviderConfig:
    """Create a provider configuration."""
    return ProviderConfig(
        name=name,
        weight=weight,
        capabilities=capabilities or [],
        cost_per_request=cost_per_request,
        typical_latency_ms=typical_latency_ms,
    )


def create_routing_context(
    priority: RequestPriority = RequestPriority.NORMAL,
    max_latency_ms: Optional[float] = None,
    max_cost: Optional[float] = None,
    required_capabilities: Optional[List[ProviderCapability]] = None,
) -> RoutingContext:
    """Create a routing context."""
    return RoutingContext(
        priority=priority,
        max_latency_ms=max_latency_ms,
        max_cost=max_cost,
        required_capabilities=required_capabilities or [],
    )
