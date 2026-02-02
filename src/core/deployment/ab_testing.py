"""A/B Testing Infrastructure.

Provides experiment management, variant assignment, and metrics tracking
for controlled experiments.
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ExperimentStatus(str, Enum):
    """Experiment lifecycle status."""

    DRAFT = "draft"  # Not yet started
    RUNNING = "running"  # Actively collecting data
    PAUSED = "paused"  # Temporarily stopped
    COMPLETED = "completed"  # Finished, results available
    CANCELLED = "cancelled"  # Stopped without results


@dataclass
class Variant:
    """An experiment variant (control or treatment)."""

    name: str
    weight: float = 1.0  # Relative weight for traffic allocation
    description: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    is_control: bool = False

    # Metrics
    impressions: int = 0
    conversions: int = 0
    total_value: float = 0.0

    @property
    def conversion_rate(self) -> float:
        """Calculate conversion rate."""
        return self.conversions / self.impressions if self.impressions > 0 else 0.0

    @property
    def average_value(self) -> float:
        """Calculate average value per impression."""
        return self.total_value / self.impressions if self.impressions > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "weight": self.weight,
            "description": self.description,
            "config": self.config,
            "is_control": self.is_control,
            "impressions": self.impressions,
            "conversions": self.conversions,
            "total_value": self.total_value,
            "conversion_rate": self.conversion_rate,
            "average_value": self.average_value,
        }


@dataclass
class Experiment:
    """An A/B test experiment."""

    id: str
    name: str
    description: str = ""
    status: ExperimentStatus = ExperimentStatus.DRAFT
    variants: List[Variant] = field(default_factory=list)

    # Targeting
    user_percentage: float = 100.0  # Percentage of users in experiment
    include_users: Set[str] = field(default_factory=set)
    exclude_users: Set[str] = field(default_factory=set)
    segments: Set[str] = field(default_factory=set)  # User segments

    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Metrics configuration
    primary_metric: str = "conversion_rate"
    secondary_metrics: List[str] = field(default_factory=list)
    minimum_sample_size: int = 100

    # Metadata
    owner: str = ""
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Ensure at least one variant exists."""
        if not self.variants:
            self.variants = [
                Variant(name="control", is_control=True),
                Variant(name="treatment"),
            ]

    @property
    def total_weight(self) -> float:
        """Get total weight of all variants."""
        return sum(v.weight for v in self.variants)

    @property
    def total_impressions(self) -> int:
        """Get total impressions across all variants."""
        return sum(v.impressions for v in self.variants)

    @property
    def is_significant(self) -> bool:
        """Check if experiment has enough data for significance."""
        return all(v.impressions >= self.minimum_sample_size for v in self.variants)

    def get_control(self) -> Optional[Variant]:
        """Get control variant."""
        for v in self.variants:
            if v.is_control:
                return v
        return self.variants[0] if self.variants else None

    def get_variant(self, name: str) -> Optional[Variant]:
        """Get variant by name."""
        for v in self.variants:
            if v.name == name:
                return v
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "variants": [v.to_dict() for v in self.variants],
            "user_percentage": self.user_percentage,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "created_at": self.created_at.isoformat(),
            "primary_metric": self.primary_metric,
            "total_impressions": self.total_impressions,
            "is_significant": self.is_significant,
            "owner": self.owner,
            "tags": self.tags,
        }


class ABTestManager:
    """Manages A/B test experiments.

    Features:
    - Experiment lifecycle management
    - Consistent variant assignment
    - Metrics tracking
    - Statistical analysis
    """

    def __init__(
        self,
        persistence_fn: Optional[Callable[[Experiment], None]] = None,
    ):
        """Initialize A/B Test manager.

        Args:
            persistence_fn: Function to persist experiment changes
        """
        self._experiments: Dict[str, Experiment] = {}
        self._user_assignments: Dict[str, Dict[str, str]] = {}  # user_id -> {exp_id: variant}
        self._persistence_fn = persistence_fn

    def create_experiment(
        self,
        id: str,
        name: str,
        variants: Optional[List[Variant]] = None,
        **kwargs,
    ) -> Experiment:
        """Create a new experiment.

        Args:
            id: Unique experiment ID
            name: Human-readable name
            variants: List of variants (defaults to control/treatment)
            **kwargs: Additional experiment parameters

        Returns:
            Created experiment
        """
        if id in self._experiments:
            raise ValueError(f"Experiment {id} already exists")

        experiment = Experiment(
            id=id,
            name=name,
            variants=variants or [],
            **kwargs,
        )
        self._experiments[id] = experiment
        self._persist(experiment)
        logger.info(f"Created experiment: {id}")
        return experiment

    def get_experiment(self, id: str) -> Optional[Experiment]:
        """Get experiment by ID."""
        return self._experiments.get(id)

    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
    ) -> List[Experiment]:
        """List experiments, optionally filtered by status."""
        experiments = list(self._experiments.values())
        if status:
            experiments = [e for e in experiments if e.status == status]
        return experiments

    def start_experiment(self, id: str) -> bool:
        """Start an experiment."""
        experiment = self._experiments.get(id)
        if not experiment:
            return False

        if experiment.status not in (ExperimentStatus.DRAFT, ExperimentStatus.PAUSED):
            logger.warning(f"Cannot start experiment {id} in status {experiment.status}")
            return False

        experiment.status = ExperimentStatus.RUNNING
        experiment.start_time = datetime.utcnow()
        self._persist(experiment)
        logger.info(f"Started experiment: {id}")
        return True

    def pause_experiment(self, id: str) -> bool:
        """Pause a running experiment."""
        experiment = self._experiments.get(id)
        if not experiment:
            return False

        if experiment.status != ExperimentStatus.RUNNING:
            return False

        experiment.status = ExperimentStatus.PAUSED
        self._persist(experiment)
        logger.info(f"Paused experiment: {id}")
        return True

    def complete_experiment(self, id: str) -> bool:
        """Complete an experiment."""
        experiment = self._experiments.get(id)
        if not experiment:
            return False

        experiment.status = ExperimentStatus.COMPLETED
        experiment.end_time = datetime.utcnow()
        self._persist(experiment)
        logger.info(f"Completed experiment: {id}")
        return True

    def get_variant_for_user(
        self,
        experiment_id: str,
        user_id: str,
        user_segments: Optional[Set[str]] = None,
    ) -> Optional[str]:
        """Get the variant assignment for a user.

        Args:
            experiment_id: Experiment ID
            user_id: User identifier
            user_segments: User's segments for targeting

        Returns:
            Variant name or None if user not in experiment
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            return None

        if experiment.status != ExperimentStatus.RUNNING:
            return None

        # Check exclusions
        if user_id in experiment.exclude_users:
            return None

        # Check inclusions (bypass percentage check)
        force_include = user_id in experiment.include_users

        # Check segment targeting
        if experiment.segments and user_segments:
            if not experiment.segments.intersection(user_segments):
                return None

        # Check cached assignment
        if user_id in self._user_assignments:
            if experiment_id in self._user_assignments[user_id]:
                return self._user_assignments[user_id][experiment_id]

        # Check if user should be in experiment (percentage)
        if not force_include:
            user_hash = self._get_user_hash(user_id, experiment_id)
            if (user_hash % 100) >= experiment.user_percentage:
                return None

        # Assign variant based on weights
        variant_name = self._assign_variant(experiment, user_id)

        # Cache assignment
        if user_id not in self._user_assignments:
            self._user_assignments[user_id] = {}
        self._user_assignments[user_id][experiment_id] = variant_name

        return variant_name

    def record_impression(
        self,
        experiment_id: str,
        user_id: str,
        variant_name: Optional[str] = None,
    ) -> bool:
        """Record an impression for a variant.

        Args:
            experiment_id: Experiment ID
            user_id: User identifier
            variant_name: Variant name (auto-detected if not provided)

        Returns:
            True if impression recorded
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment or experiment.status != ExperimentStatus.RUNNING:
            return False

        if not variant_name:
            variant_name = self.get_variant_for_user(experiment_id, user_id)

        if not variant_name:
            return False

        variant = experiment.get_variant(variant_name)
        if variant:
            variant.impressions += 1
            return True
        return False

    def record_conversion(
        self,
        experiment_id: str,
        user_id: str,
        value: float = 1.0,
        variant_name: Optional[str] = None,
    ) -> bool:
        """Record a conversion for a variant.

        Args:
            experiment_id: Experiment ID
            user_id: User identifier
            value: Conversion value
            variant_name: Variant name (auto-detected if not provided)

        Returns:
            True if conversion recorded
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment or experiment.status != ExperimentStatus.RUNNING:
            return False

        if not variant_name:
            variant_name = self.get_variant_for_user(experiment_id, user_id)

        if not variant_name:
            return False

        variant = experiment.get_variant(variant_name)
        if variant:
            variant.conversions += 1
            variant.total_value += value
            return True
        return False

    def get_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment results with statistical analysis.

        Args:
            experiment_id: Experiment ID

        Returns:
            Results dictionary with metrics and analysis
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            return None

        control = experiment.get_control()
        if not control:
            return None

        results = {
            "experiment_id": experiment_id,
            "status": experiment.status.value,
            "total_impressions": experiment.total_impressions,
            "is_significant": experiment.is_significant,
            "variants": {},
            "winner": None,
            "lift": None,
        }

        best_variant = None
        best_rate = 0.0

        for variant in experiment.variants:
            variant_results = {
                "impressions": variant.impressions,
                "conversions": variant.conversions,
                "conversion_rate": variant.conversion_rate,
                "average_value": variant.average_value,
                "is_control": variant.is_control,
            }

            # Calculate lift vs control
            if not variant.is_control and control.conversion_rate > 0:
                lift = (variant.conversion_rate - control.conversion_rate) / control.conversion_rate
                variant_results["lift_vs_control"] = lift

            results["variants"][variant.name] = variant_results

            if variant.conversion_rate > best_rate:
                best_rate = variant.conversion_rate
                best_variant = variant.name

        # Determine winner
        if experiment.is_significant and best_variant:
            results["winner"] = best_variant
            if best_variant != control.name and control.conversion_rate > 0:
                results["lift"] = (best_rate - control.conversion_rate) / control.conversion_rate

        return results

    def _get_user_hash(self, user_id: str, experiment_id: str) -> int:
        """Get consistent hash for user assignment."""
        key = f"{user_id}:{experiment_id}"
        return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)  # nosec B324

    def _assign_variant(self, experiment: Experiment, user_id: str) -> str:
        """Assign user to a variant based on weights."""
        user_hash = self._get_user_hash(user_id, experiment.id)
        bucket = user_hash % 1000

        cumulative_weight = 0.0
        total_weight = experiment.total_weight

        for variant in experiment.variants:
            cumulative_weight += variant.weight
            threshold = (cumulative_weight / total_weight) * 1000
            if bucket < threshold:
                return variant.name

        return experiment.variants[-1].name if experiment.variants else "control"

    def _persist(self, experiment: Experiment) -> None:
        """Persist experiment changes."""
        if self._persistence_fn:
            try:
                self._persistence_fn(experiment)
            except Exception as e:
                logger.error(f"Failed to persist experiment {experiment.id}: {e}")


# Global manager instance
_ab_manager: Optional[ABTestManager] = None


def get_ab_test_manager() -> ABTestManager:
    """Get global A/B test manager."""
    global _ab_manager
    if _ab_manager is None:
        _ab_manager = ABTestManager()
    return _ab_manager
