"""
Data Augmentation Module (M4).

Provides augmentation for CAD/DXF data:
- Geometric transformations
- Graph structure augmentation
- Feature perturbation
- Composable augmentation pipelines
"""

from src.ml.augmentation.geometric import (
    GeometricAugmentation,
    RandomRotation,
    RandomScale,
    RandomTranslation,
    RandomFlip,
    RandomShear,
    AffineTransform,
)
from src.ml.augmentation.graph import (
    GraphAugmentation,
    NodeDropout,
    EdgeDropout,
    NodeFeatureNoise,
    EdgeFeaturePerturbation,
    SubgraphSampling,
    GraphMixup,
)
from src.ml.augmentation.pipeline import (
    AugmentationPipeline,
    Compose,
    RandomChoice,
    RandomApply,
    AugmentationConfig,
)
from src.ml.augmentation.cad import (
    CADAugmentation,
    LayerShuffle,
    EntityDropout,
    TextPerturbation,
    DimensionNoise,
)

__all__ = [
    # Geometric
    "GeometricAugmentation",
    "RandomRotation",
    "RandomScale",
    "RandomTranslation",
    "RandomFlip",
    "RandomShear",
    "AffineTransform",
    # Graph
    "GraphAugmentation",
    "NodeDropout",
    "EdgeDropout",
    "NodeFeatureNoise",
    "EdgeFeaturePerturbation",
    "SubgraphSampling",
    "GraphMixup",
    # Pipeline
    "AugmentationPipeline",
    "Compose",
    "RandomChoice",
    "RandomApply",
    "AugmentationConfig",
    # CAD
    "CADAugmentation",
    "LayerShuffle",
    "EntityDropout",
    "TextPerturbation",
    "DimensionNoise",
]
