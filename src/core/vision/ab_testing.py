"""A/B Testing framework for Vision Provider system.

This module provides experimentation capabilities for comparing provider
performance including:
- Experiment configuration and management
- Traffic splitting and variant assignment
- Statistical analysis of results
- Winner determination
"""

import hashlib
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .base import VisionDescription, VisionProvider


class ExperimentStatus(Enum):
    """Experiment lifecycle status."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class VariantType(Enum):
    """Type of experiment variant."""

    CONTROL = "control"
    TREATMENT = "treatment"


class AllocationStrategy(Enum):
    """Traffic allocation strategies."""

    RANDOM = "random"  # Pure random assignment
    HASH_BASED = "hash_based"  # Deterministic based on request hash
    WEIGHTED = "weighted"  # Weighted random based on variant weights
    ROUND_ROBIN = "round_robin"  # Sequential assignment


class StatisticalMethod(Enum):
    """Statistical methods for analysis."""

    T_TEST = "t_test"
    CHI_SQUARE = "chi_square"
    BAYESIAN = "bayesian"
    BOOTSTRAP = "bootstrap"


@dataclass
class Variant:
    """Experiment variant configuration."""

    name: str
    provider: VisionProvider
    weight: float = 0.5  # Traffic weight (0.0 to 1.0)
    variant_type: VariantType = VariantType.TREATMENT
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Result of a single experiment observation."""

    variant_name: str
    request_id: str
    latency_ms: float
    success: bool
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VariantStats:
    """Aggregated statistics for a variant."""

    variant_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0
    total_confidence: float = 0.0
    latencies: List[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    @property
    def average_confidence(self) -> float:
        """Calculate average confidence."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_confidence / self.successful_requests

    @property
    def latency_variance(self) -> float:
        """Calculate latency variance."""
        if len(self.latencies) < 2:
            return 0.0
        avg = self.average_latency_ms
        return sum((x - avg) ** 2 for x in self.latencies) / (len(self.latencies) - 1)

    @property
    def latency_std_dev(self) -> float:
        """Calculate latency standard deviation."""
        return self.latency_variance**0.5


@dataclass
class ExperimentAnalysis:
    """Statistical analysis of experiment results."""

    experiment_id: str
    control_stats: VariantStats
    treatment_stats: Dict[str, VariantStats]
    winner: Optional[str] = None
    confidence_level: float = 0.0
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    is_significant: bool = False
    recommendation: str = ""
    analysis_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExperimentConfig:
    """Experiment configuration."""

    experiment_id: str
    name: str
    description: str = ""
    variants: List[Variant] = field(default_factory=list)
    allocation_strategy: AllocationStrategy = AllocationStrategy.HASH_BASED
    min_sample_size: int = 100
    max_duration_hours: int = 168  # 1 week default
    confidence_threshold: float = 0.95
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    status: ExperimentStatus = ExperimentStatus.DRAFT
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExperimentManager:
    """Manages A/B testing experiments."""

    def __init__(self) -> None:
        """Initialize the experiment manager."""
        self._experiments: Dict[str, ExperimentConfig] = {}
        self._results: Dict[str, List[ExperimentResult]] = {}
        self._stats: Dict[str, Dict[str, VariantStats]] = {}
        self._round_robin_counters: Dict[str, int] = {}

    def create_experiment(
        self,
        experiment_id: str,
        name: str,
        control: VisionProvider,
        treatments: List[Tuple[str, VisionProvider, float]],
        description: str = "",
        allocation_strategy: AllocationStrategy = AllocationStrategy.HASH_BASED,
        min_sample_size: int = 100,
        max_duration_hours: int = 168,
        confidence_threshold: float = 0.95,
    ) -> ExperimentConfig:
        """Create a new experiment.

        Args:
            experiment_id: Unique identifier for the experiment
            name: Human-readable experiment name
            control: Control variant provider
            treatments: List of (name, provider, weight) tuples for treatment variants
            description: Experiment description
            allocation_strategy: How to allocate traffic
            min_sample_size: Minimum samples before analysis
            max_duration_hours: Maximum experiment duration
            confidence_threshold: Required confidence for significance

        Returns:
            The created experiment configuration
        """
        # Create control variant
        control_variant = Variant(
            name="control",
            provider=control,
            weight=1.0 - sum(w for _, _, w in treatments),
            variant_type=VariantType.CONTROL,
        )

        # Create treatment variants
        treatment_variants = [
            Variant(
                name=name,
                provider=provider,
                weight=weight,
                variant_type=VariantType.TREATMENT,
            )
            for name, provider, weight in treatments
        ]

        config = ExperimentConfig(
            experiment_id=experiment_id,
            name=name,
            description=description,
            variants=[control_variant] + treatment_variants,
            allocation_strategy=allocation_strategy,
            min_sample_size=min_sample_size,
            max_duration_hours=max_duration_hours,
            confidence_threshold=confidence_threshold,
        )

        self._experiments[experiment_id] = config
        self._results[experiment_id] = []
        self._stats[experiment_id] = {
            v.name: VariantStats(variant_name=v.name) for v in config.variants
        }
        self._round_robin_counters[experiment_id] = 0

        return config

    def start_experiment(self, experiment_id: str) -> None:
        """Start an experiment."""
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self._experiments[experiment_id]
        if experiment.status != ExperimentStatus.DRAFT:
            raise ValueError(f"Cannot start experiment in {experiment.status.value} status")

        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.now()

    def pause_experiment(self, experiment_id: str) -> None:
        """Pause a running experiment."""
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self._experiments[experiment_id]
        if experiment.status != ExperimentStatus.RUNNING:
            raise ValueError("Can only pause running experiments")

        experiment.status = ExperimentStatus.PAUSED

    def resume_experiment(self, experiment_id: str) -> None:
        """Resume a paused experiment."""
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self._experiments[experiment_id]
        if experiment.status != ExperimentStatus.PAUSED:
            raise ValueError("Can only resume paused experiments")

        experiment.status = ExperimentStatus.RUNNING

    def complete_experiment(self, experiment_id: str) -> None:
        """Mark an experiment as completed."""
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self._experiments[experiment_id]
        experiment.status = ExperimentStatus.COMPLETED
        experiment.ended_at = datetime.now()

    def cancel_experiment(self, experiment_id: str) -> None:
        """Cancel an experiment."""
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self._experiments[experiment_id]
        experiment.status = ExperimentStatus.CANCELLED
        experiment.ended_at = datetime.now()

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentConfig]:
        """Get experiment configuration."""
        return self._experiments.get(experiment_id)

    def list_experiments(self, status: Optional[ExperimentStatus] = None) -> List[ExperimentConfig]:
        """List all experiments, optionally filtered by status."""
        experiments = list(self._experiments.values())
        if status:
            experiments = [e for e in experiments if e.status == status]
        return experiments

    def allocate_variant(self, experiment_id: str, request_id: str) -> Optional[Variant]:
        """Allocate a variant for a request.

        Args:
            experiment_id: The experiment to allocate for
            request_id: Unique request identifier for deterministic allocation

        Returns:
            The allocated variant, or None if experiment not running
        """
        if experiment_id not in self._experiments:
            return None

        experiment = self._experiments[experiment_id]
        if experiment.status != ExperimentStatus.RUNNING:
            return None

        # Check if experiment has exceeded max duration
        if experiment.started_at:
            elapsed = datetime.now() - experiment.started_at
            if elapsed > timedelta(hours=experiment.max_duration_hours):
                self.complete_experiment(experiment_id)
                return None

        strategy = experiment.allocation_strategy
        variants = experiment.variants

        if strategy == AllocationStrategy.RANDOM:
            return self._allocate_random(variants)
        elif strategy == AllocationStrategy.HASH_BASED:
            return self._allocate_hash_based(variants, request_id)
        elif strategy == AllocationStrategy.WEIGHTED:
            return self._allocate_weighted(variants)
        elif strategy == AllocationStrategy.ROUND_ROBIN:
            return self._allocate_round_robin(experiment_id, variants)

        return variants[0] if variants else None

    def _allocate_random(self, variants: List[Variant]) -> Variant:
        """Allocate using pure random selection."""
        return random.choice(variants)

    def _allocate_hash_based(self, variants: List[Variant], request_id: str) -> Variant:
        """Allocate using deterministic hash-based selection."""
        hash_value = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        normalized = (hash_value % 10000) / 10000.0

        cumulative = 0.0
        for variant in variants:
            cumulative += variant.weight
            if normalized < cumulative:
                return variant

        return variants[-1]

    def _allocate_weighted(self, variants: List[Variant]) -> Variant:
        """Allocate using weighted random selection."""
        weights = [v.weight for v in variants]
        return random.choices(variants, weights=weights, k=1)[0]

    def _allocate_round_robin(self, experiment_id: str, variants: List[Variant]) -> Variant:
        """Allocate using round-robin selection."""
        counter = self._round_robin_counters.get(experiment_id, 0)
        variant = variants[counter % len(variants)]
        self._round_robin_counters[experiment_id] = counter + 1
        return variant

    def record_result(
        self,
        experiment_id: str,
        variant_name: str,
        request_id: str,
        latency_ms: float,
        success: bool,
        confidence: float = 0.0,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an experiment result.

        Args:
            experiment_id: The experiment ID
            variant_name: Name of the variant used
            request_id: Unique request identifier
            latency_ms: Request latency in milliseconds
            success: Whether the request succeeded
            confidence: Confidence score of the result
            error: Error message if failed
            metadata: Additional metadata
        """
        if experiment_id not in self._experiments:
            return

        result = ExperimentResult(
            variant_name=variant_name,
            request_id=request_id,
            latency_ms=latency_ms,
            success=success,
            confidence=confidence,
            error=error,
            metadata=metadata or {},
        )

        self._results[experiment_id].append(result)

        # Update stats
        stats = self._stats[experiment_id].get(variant_name)
        if stats:
            stats.total_requests += 1
            stats.total_latency_ms += latency_ms
            stats.min_latency_ms = min(stats.min_latency_ms, latency_ms)
            stats.max_latency_ms = max(stats.max_latency_ms, latency_ms)
            stats.latencies.append(latency_ms)

            if success:
                stats.successful_requests += 1
                stats.total_confidence += confidence
            else:
                stats.failed_requests += 1

    def get_stats(self, experiment_id: str) -> Dict[str, VariantStats]:
        """Get current statistics for an experiment."""
        return self._stats.get(experiment_id, {})

    def analyze_experiment(
        self,
        experiment_id: str,
        method: StatisticalMethod = StatisticalMethod.T_TEST,
    ) -> ExperimentAnalysis:
        """Analyze experiment results.

        Args:
            experiment_id: The experiment to analyze
            method: Statistical method to use

        Returns:
            Analysis results including winner determination
        """
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self._experiments[experiment_id]
        stats = self._stats[experiment_id]

        # Find control and treatment stats
        control_stats = None
        treatment_stats = {}

        for variant in experiment.variants:
            variant_stats = stats.get(variant.name)
            if variant_stats:
                if variant.variant_type == VariantType.CONTROL:
                    control_stats = variant_stats
                else:
                    treatment_stats[variant.name] = variant_stats

        if not control_stats:
            control_stats = VariantStats(variant_name="control")

        # Perform statistical analysis
        analysis = self._perform_analysis(
            experiment_id,
            control_stats,
            treatment_stats,
            experiment.confidence_threshold,
            method,
        )

        return analysis

    def _perform_analysis(
        self,
        experiment_id: str,
        control: VariantStats,
        treatments: Dict[str, VariantStats],
        confidence_threshold: float,
        method: StatisticalMethod,
    ) -> ExperimentAnalysis:
        """Perform statistical analysis."""
        winner = None
        best_improvement = 0.0
        p_value = None
        effect_size = None
        is_significant = False

        for name, treatment in treatments.items():
            if treatment.total_requests < 30 or control.total_requests < 30:
                continue

            # Calculate effect size (Cohen's d for latency)
            if control.latency_std_dev > 0 and treatment.latency_std_dev > 0:
                pooled_std = (control.latency_std_dev + treatment.latency_std_dev) / 2
                if pooled_std > 0:
                    d = (control.average_latency_ms - treatment.average_latency_ms) / pooled_std
                    effect_size = abs(d)

            # Simple t-test approximation
            if method == StatisticalMethod.T_TEST:
                p_value = self._simple_t_test(control, treatment)

            # Check for improvement
            latency_improvement = (
                (control.average_latency_ms - treatment.average_latency_ms)
                / control.average_latency_ms
                if control.average_latency_ms > 0
                else 0
            )

            success_improvement = treatment.success_rate - control.success_rate

            # Combined improvement score
            improvement = latency_improvement * 0.5 + success_improvement * 0.5

            if improvement > best_improvement:
                best_improvement = improvement
                if p_value is not None and p_value < (1 - confidence_threshold):
                    winner = name
                    is_significant = True

        # Generate recommendation
        recommendation = self._generate_recommendation(control, treatments, winner, is_significant)

        return ExperimentAnalysis(
            experiment_id=experiment_id,
            control_stats=control,
            treatment_stats=treatments,
            winner=winner,
            confidence_level=confidence_threshold,
            p_value=p_value,
            effect_size=effect_size,
            is_significant=is_significant,
            recommendation=recommendation,
        )

    def _simple_t_test(self, control: VariantStats, treatment: VariantStats) -> float:
        """Simplified t-test calculation."""
        if control.total_requests < 2 or treatment.total_requests < 2:
            return 1.0

        # Welch's t-test approximation
        n1, n2 = control.total_requests, treatment.total_requests
        m1, m2 = control.average_latency_ms, treatment.average_latency_ms
        v1, v2 = control.latency_variance, treatment.latency_variance

        if v1 == 0 and v2 == 0:
            return 1.0 if m1 == m2 else 0.0

        se = ((v1 / n1) + (v2 / n2)) ** 0.5
        if se == 0:
            return 1.0

        t_stat = abs(m1 - m2) / se

        # Approximate p-value (simplified)
        df = min(n1, n2) - 1
        if df < 1:
            df = 1

        # Simple approximation: higher t-stat = lower p-value
        p_value = max(0.001, 1.0 / (1.0 + t_stat * (df**0.5) / 10))

        return p_value

    def _generate_recommendation(
        self,
        control: VariantStats,
        treatments: Dict[str, VariantStats],
        winner: Optional[str],
        is_significant: bool,
    ) -> str:
        """Generate analysis recommendation."""
        if not treatments:
            return "No treatment variants to analyze."

        total_samples = control.total_requests + sum(t.total_requests for t in treatments.values())

        if total_samples < 100:
            return f"Insufficient data. Current samples: {total_samples}. Recommend collecting at least 100 samples."

        if winner and is_significant:
            treatment = treatments[winner]
            latency_diff = control.average_latency_ms - treatment.average_latency_ms
            success_diff = treatment.success_rate - control.success_rate

            return (
                f"Winner: {winner}. "
                f"Latency improvement: {latency_diff:.1f}ms. "
                f"Success rate change: {success_diff:+.1%}. "
                f"Recommend deploying {winner} to production."
            )

        return "No statistically significant winner. Continue experiment or increase sample size."


class ABTestingVisionProvider(VisionProvider):
    """Vision provider wrapper that participates in A/B testing."""

    def __init__(
        self,
        experiment_manager: ExperimentManager,
        experiment_id: str,
        fallback_provider: VisionProvider,
    ) -> None:
        """Initialize the A/B testing provider.

        Args:
            experiment_manager: The experiment manager
            experiment_id: ID of the experiment to participate in
            fallback_provider: Provider to use if experiment not running
        """
        self._manager = experiment_manager
        self._experiment_id = experiment_id
        self._fallback = fallback_provider

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return f"ab_testing_{self._experiment_id}"

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image using A/B tested provider.

        Args:
            image_data: Raw image bytes
            include_description: Whether to include description

        Returns:
            Vision analysis description
        """
        import uuid

        request_id = str(uuid.uuid4())

        # Allocate variant
        variant = self._manager.allocate_variant(self._experiment_id, request_id)

        if not variant:
            # Experiment not running, use fallback
            return await self._fallback.analyze_image(image_data, include_description)

        # Execute with timing
        start_time = time.time()
        error_msg = None
        success = False
        confidence = 0.0

        try:
            result = await variant.provider.analyze_image(image_data, include_description)
            success = True
            confidence = result.confidence
            return result
        except Exception as e:
            error_msg = str(e)
            raise
        finally:
            latency_ms = (time.time() - start_time) * 1000

            # Record result
            self._manager.record_result(
                experiment_id=self._experiment_id,
                variant_name=variant.name,
                request_id=request_id,
                latency_ms=latency_ms,
                success=success,
                confidence=confidence,
                error=error_msg,
            )

    def get_current_stats(self) -> Dict[str, VariantStats]:
        """Get current experiment statistics."""
        return self._manager.get_stats(self._experiment_id)

    def analyze_results(
        self, method: StatisticalMethod = StatisticalMethod.T_TEST
    ) -> ExperimentAnalysis:
        """Analyze current experiment results."""
        return self._manager.analyze_experiment(self._experiment_id, method)


def create_ab_testing_provider(
    experiment_id: str,
    name: str,
    control: VisionProvider,
    treatments: List[Tuple[str, VisionProvider, float]],
    allocation_strategy: AllocationStrategy = AllocationStrategy.HASH_BASED,
    min_sample_size: int = 100,
    experiment_manager: Optional[ExperimentManager] = None,
) -> Tuple[ABTestingVisionProvider, ExperimentManager]:
    """Create an A/B testing provider.

    Args:
        experiment_id: Unique experiment identifier
        name: Experiment name
        control: Control variant provider
        treatments: List of (name, provider, weight) for treatments
        allocation_strategy: Traffic allocation strategy
        min_sample_size: Minimum samples for analysis
        experiment_manager: Optional existing manager

    Returns:
        Tuple of (provider, manager)
    """
    manager = experiment_manager or ExperimentManager()

    manager.create_experiment(
        experiment_id=experiment_id,
        name=name,
        control=control,
        treatments=treatments,
        allocation_strategy=allocation_strategy,
        min_sample_size=min_sample_size,
    )

    manager.start_experiment(experiment_id)

    provider = ABTestingVisionProvider(
        experiment_manager=manager,
        experiment_id=experiment_id,
        fallback_provider=control,
    )

    return provider, manager


# Singleton experiment manager
_experiment_manager: Optional[ExperimentManager] = None


def get_experiment_manager() -> ExperimentManager:
    """Get the global experiment manager singleton."""
    global _experiment_manager
    if _experiment_manager is None:
        _experiment_manager = ExperimentManager()
    return _experiment_manager
