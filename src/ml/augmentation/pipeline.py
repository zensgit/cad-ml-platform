"""
Augmentation pipeline for composing multiple augmentations.

Provides flexible composition of augmentations.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for augmentation pipeline."""
    enabled: bool = True
    p: float = 1.0  # Global probability
    seed: Optional[int] = None
    # Geometric
    rotation_range: tuple = (-15.0, 15.0)
    scale_range: tuple = (0.9, 1.1)
    translate_range: tuple = (-0.05, 0.05)
    flip_horizontal: bool = True
    flip_vertical: bool = False
    # Graph
    node_dropout: float = 0.05
    edge_dropout: float = 0.1
    feature_noise: float = 0.02
    # Intensity
    intensity: str = "medium"  # low, medium, high


class Compose:
    """
    Compose multiple augmentations.

    Applies augmentations in sequence.
    """

    def __init__(self, augmentations: List[Callable]):
        """
        Initialize composition.

        Args:
            augmentations: List of augmentation callables
        """
        self.augmentations = augmentations

    def __call__(self, data: Any) -> Any:
        """Apply all augmentations in sequence."""
        for aug in self.augmentations:
            data = aug(data)
        return data

    def __repr__(self) -> str:
        aug_names = [aug.__class__.__name__ for aug in self.augmentations]
        return f"Compose([{', '.join(aug_names)}])"


class RandomChoice:
    """
    Randomly choose one augmentation to apply.
    """

    def __init__(
        self,
        augmentations: List[Callable],
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize random choice.

        Args:
            augmentations: List of augmentation callables
            weights: Optional weights for each augmentation
        """
        self.augmentations = augmentations
        self.weights = weights

    def __call__(self, data: Any) -> Any:
        """Apply randomly chosen augmentation."""
        if self.weights:
            aug = random.choices(self.augmentations, weights=self.weights, k=1)[0]
        else:
            aug = random.choice(self.augmentations)
        return aug(data)

    def __repr__(self) -> str:
        aug_names = [aug.__class__.__name__ for aug in self.augmentations]
        return f"RandomChoice([{', '.join(aug_names)}])"


class RandomApply:
    """
    Randomly apply an augmentation with given probability.
    """

    def __init__(self, augmentation: Callable, p: float = 0.5):
        """
        Initialize random apply.

        Args:
            augmentation: Augmentation to apply
            p: Probability of applying
        """
        self.augmentation = augmentation
        self.p = p

    def __call__(self, data: Any) -> Any:
        """Maybe apply augmentation."""
        if random.random() < self.p:
            return self.augmentation(data)
        return data

    def __repr__(self) -> str:
        return f"RandomApply({self.augmentation.__class__.__name__}, p={self.p})"


class RandomOrder:
    """
    Apply augmentations in random order.
    """

    def __init__(self, augmentations: List[Callable]):
        """
        Initialize random order.

        Args:
            augmentations: List of augmentation callables
        """
        self.augmentations = augmentations

    def __call__(self, data: Any) -> Any:
        """Apply augmentations in random order."""
        order = list(range(len(self.augmentations)))
        random.shuffle(order)
        for i in order:
            data = self.augmentations[i](data)
        return data

    def __repr__(self) -> str:
        aug_names = [aug.__class__.__name__ for aug in self.augmentations]
        return f"RandomOrder([{', '.join(aug_names)}])"


class AugmentationPipeline:
    """
    High-level augmentation pipeline.

    Provides easy configuration and creation of augmentation pipelines.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        self._config = config or AugmentationConfig()
        self._augmentations: List[Callable] = []
        self._enabled = self._config.enabled

        if self._config.seed is not None:
            random.seed(self._config.seed)

    @property
    def config(self) -> AugmentationConfig:
        return self._config

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> "AugmentationPipeline":
        """Enable augmentation."""
        self._enabled = True
        return self

    def disable(self) -> "AugmentationPipeline":
        """Disable augmentation."""
        self._enabled = False
        return self

    def add(self, augmentation: Callable) -> "AugmentationPipeline":
        """Add an augmentation to the pipeline."""
        self._augmentations.append(augmentation)
        return self

    def clear(self) -> "AugmentationPipeline":
        """Clear all augmentations."""
        self._augmentations.clear()
        return self

    def __call__(self, data: Any) -> Any:
        """Apply pipeline to data."""
        if not self._enabled:
            return data

        if random.random() > self._config.p:
            return data

        for aug in self._augmentations:
            data = aug(data)

        return data

    def __repr__(self) -> str:
        aug_names = [aug.__class__.__name__ for aug in self._augmentations]
        return f"AugmentationPipeline([{', '.join(aug_names)}], enabled={self._enabled})"

    @classmethod
    def from_config(cls, config: AugmentationConfig) -> "AugmentationPipeline":
        """Create pipeline from configuration."""
        pipeline = cls(config)

        if not config.enabled:
            return pipeline

        # Import augmentations
        from src.ml.augmentation.geometric import (
            RandomRotation, RandomScale, RandomTranslation, RandomFlip
        )
        from src.ml.augmentation.graph import (
            NodeDropout, EdgeDropout, NodeFeatureNoise
        )

        # Set intensity multiplier
        intensity_mult = {"low": 0.5, "medium": 1.0, "high": 1.5}.get(config.intensity, 1.0)

        # Add geometric augmentations
        if config.rotation_range != (0, 0):
            r_min = config.rotation_range[0] * intensity_mult
            r_max = config.rotation_range[1] * intensity_mult
            pipeline.add(RandomRotation(angle_range=(r_min, r_max), p=0.5))

        if config.scale_range != (1, 1):
            s_min = 1 - (1 - config.scale_range[0]) * intensity_mult
            s_max = 1 + (config.scale_range[1] - 1) * intensity_mult
            pipeline.add(RandomScale(scale_range=(s_min, s_max), p=0.5))

        if config.translate_range != (0, 0):
            t_min = config.translate_range[0] * intensity_mult
            t_max = config.translate_range[1] * intensity_mult
            pipeline.add(RandomTranslation(translate_range=(t_min, t_max), p=0.5))

        if config.flip_horizontal or config.flip_vertical:
            pipeline.add(RandomFlip(
                horizontal=config.flip_horizontal,
                vertical=config.flip_vertical,
                p=0.5
            ))

        # Add graph augmentations
        if config.node_dropout > 0:
            dropout = config.node_dropout * intensity_mult
            pipeline.add(NodeDropout(dropout_rate=dropout, p=0.3))

        if config.edge_dropout > 0:
            dropout = config.edge_dropout * intensity_mult
            pipeline.add(EdgeDropout(dropout_rate=dropout, p=0.3))

        if config.feature_noise > 0:
            noise = config.feature_noise * intensity_mult
            pipeline.add(NodeFeatureNoise(noise_scale=noise, p=0.5))

        return pipeline

    @classmethod
    def default_geometric(cls) -> "AugmentationPipeline":
        """Create default geometric augmentation pipeline."""
        from src.ml.augmentation.geometric import (
            RandomRotation, RandomScale, RandomTranslation, RandomFlip
        )

        pipeline = cls()
        pipeline.add(RandomRotation(angle_range=(-15, 15), p=0.5))
        pipeline.add(RandomScale(scale_range=(0.9, 1.1), p=0.5))
        pipeline.add(RandomTranslation(translate_range=(-0.05, 0.05), p=0.5))
        pipeline.add(RandomFlip(horizontal=True, vertical=False, p=0.5))

        return pipeline

    @classmethod
    def default_graph(cls) -> "AugmentationPipeline":
        """Create default graph augmentation pipeline."""
        from src.ml.augmentation.graph import (
            NodeDropout, EdgeDropout, NodeFeatureNoise
        )

        pipeline = cls()
        pipeline.add(NodeDropout(dropout_rate=0.05, p=0.3))
        pipeline.add(EdgeDropout(dropout_rate=0.1, p=0.3))
        pipeline.add(NodeFeatureNoise(noise_scale=0.02, p=0.5))

        return pipeline

    @classmethod
    def default_cad(cls) -> "AugmentationPipeline":
        """Create default CAD augmentation pipeline."""
        from src.ml.augmentation.geometric import (
            RandomRotation, RandomScale, RandomFlip
        )
        from src.ml.augmentation.graph import (
            EdgeDropout, NodeFeatureNoise
        )

        pipeline = cls()
        # CAD-specific: only 90-degree rotations
        pipeline.add(RandomChoice([
            RandomRotation(angle_range=(0, 0), p=1.0),  # No rotation
            RandomRotation(angle_range=(90, 90), p=1.0),
            RandomRotation(angle_range=(180, 180), p=1.0),
            RandomRotation(angle_range=(270, 270), p=1.0),
        ]))
        pipeline.add(RandomScale(scale_range=(0.95, 1.05), p=0.5))
        pipeline.add(RandomFlip(horizontal=True, vertical=True, p=0.3))
        pipeline.add(EdgeDropout(dropout_rate=0.05, p=0.2))
        pipeline.add(NodeFeatureNoise(noise_scale=0.01, p=0.3))

        return pipeline
