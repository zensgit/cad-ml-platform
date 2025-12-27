"""
ML Integration Module - Phase 12.

Provides machine learning model integration including model management,
feature extraction, inference pipelines, and model versioning.
"""

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

from .base import VisionDescription, VisionProvider

# ============================================================================
# Enums
# ============================================================================


class ModelType(Enum):
    """Types of ML models."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    EMBEDDING = "embedding"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    NLP = "nlp"
    CUSTOM = "custom"


class ModelStatus(Enum):
    """Model deployment status."""

    DRAFT = "draft"
    TRAINING = "training"
    VALIDATING = "validating"
    READY = "ready"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class InferenceMode(Enum):
    """Inference execution modes."""

    SYNC = "sync"
    ASYNC = "async"
    BATCH = "batch"
    STREAMING = "streaming"


class FeatureType(Enum):
    """Types of features."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    IMAGE = "image"
    EMBEDDING = "embedding"
    TIMESTAMP = "timestamp"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class ModelMetadata:
    """Metadata for an ML model."""

    model_id: str
    name: str
    version: str
    model_type: ModelType
    status: ModelStatus = ModelStatus.DRAFT
    description: str = ""
    framework: str = "unknown"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelVersion:
    """A specific version of a model."""

    version_id: str
    model_id: str
    version: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifact_path: str = ""
    is_current: bool = False
    changelog: str = ""


@dataclass
class Feature:
    """A feature definition."""

    name: str
    feature_type: FeatureType
    required: bool = True
    default_value: Optional[Any] = None
    description: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureVector:
    """A vector of features."""

    features: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_list(self, feature_names: List[str]) -> List[Any]:
        """Convert to ordered list."""
        return [self.features.get(name) for name in feature_names]


@dataclass
class InferenceRequest:
    """Request for model inference."""

    request_id: str
    model_id: str
    features: FeatureVector
    mode: InferenceMode = InferenceMode.SYNC
    timeout_ms: int = 5000
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    """Result from model inference."""

    request_id: str
    model_id: str
    predictions: List[Any]
    confidence: float = 0.0
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineStage:
    """A stage in an inference pipeline."""

    stage_id: str
    name: str
    processor: Callable[[Any], Any]
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Configuration for an inference pipeline."""

    pipeline_id: str
    name: str
    stages: List[PipelineStage] = field(default_factory=list)
    timeout_ms: int = 10000
    retry_count: int = 3


# ============================================================================
# Model Store Interface
# ============================================================================


class ModelStore(ABC):
    """Abstract model storage."""

    @abstractmethod
    def save_model(self, metadata: ModelMetadata, model_data: bytes) -> str:
        """Save a model."""
        pass

    @abstractmethod
    def load_model(self, model_id: str) -> tuple[ModelMetadata, bytes]:
        """Load a model."""
        pass

    @abstractmethod
    def get_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata."""
        pass

    @abstractmethod
    def list_models(self, model_type: Optional[ModelType] = None) -> List[ModelMetadata]:
        """List all models."""
        pass

    @abstractmethod
    def delete_model(self, model_id: str) -> bool:
        """Delete a model."""
        pass


class InMemoryModelStore(ModelStore):
    """In-memory model storage."""

    def __init__(self) -> None:
        self._models: Dict[str, tuple[ModelMetadata, bytes]] = {}

    def save_model(self, metadata: ModelMetadata, model_data: bytes) -> str:
        """Save a model."""
        self._models[metadata.model_id] = (metadata, model_data)
        return metadata.model_id

    def load_model(self, model_id: str) -> tuple[ModelMetadata, bytes]:
        """Load a model."""
        if model_id not in self._models:
            raise KeyError(f"Model not found: {model_id}")
        return self._models[model_id]

    def get_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata."""
        if model_id in self._models:
            return self._models[model_id][0]
        return None

    def list_models(self, model_type: Optional[ModelType] = None) -> List[ModelMetadata]:
        """List all models."""
        models = [m[0] for m in self._models.values()]
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        return models

    def delete_model(self, model_id: str) -> bool:
        """Delete a model."""
        if model_id in self._models:
            del self._models[model_id]
            return True
        return False


# ============================================================================
# Feature Extractor
# ============================================================================


class FeatureExtractor(ABC):
    """Abstract feature extractor."""

    @abstractmethod
    def extract(self, data: Any) -> FeatureVector:
        """Extract features from data."""
        pass

    @abstractmethod
    def get_feature_schema(self) -> List[Feature]:
        """Get the feature schema."""
        pass


class ImageFeatureExtractor(FeatureExtractor):
    """Feature extractor for images."""

    def __init__(self, features: Optional[List[str]] = None) -> None:
        self._features = features or ["width", "height", "channels", "size"]

    def extract(self, data: bytes) -> FeatureVector:
        """Extract features from image data."""
        features = {
            "size": len(data),
            "hash": hashlib.md5(data).hexdigest()[:8],
        }

        # Simple image header detection
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            features["format"] = "png"
        elif data[:2] == b"\xff\xd8":
            features["format"] = "jpeg"
        elif data[:4] == b"GIF8":
            features["format"] = "gif"
        else:
            features["format"] = "unknown"

        return FeatureVector(features=features)

    def get_feature_schema(self) -> List[Feature]:
        """Get the feature schema."""
        return [
            Feature(name="size", feature_type=FeatureType.NUMERIC),
            Feature(name="hash", feature_type=FeatureType.TEXT),
            Feature(name="format", feature_type=FeatureType.CATEGORICAL),
        ]


class CompositeFeatureExtractor(FeatureExtractor):
    """Combines multiple feature extractors."""

    def __init__(self, extractors: List[FeatureExtractor]) -> None:
        self._extractors = extractors

    def extract(self, data: Any) -> FeatureVector:
        """Extract features using all extractors."""
        combined: Dict[str, Any] = {}
        for extractor in self._extractors:
            vector = extractor.extract(data)
            combined.update(vector.features)
        return FeatureVector(features=combined)

    def get_feature_schema(self) -> List[Feature]:
        """Get combined feature schema."""
        schema: List[Feature] = []
        for extractor in self._extractors:
            schema.extend(extractor.get_feature_schema())
        return schema


# ============================================================================
# Model Interface
# ============================================================================


class MLModel(ABC):
    """Abstract ML model interface."""

    @abstractmethod
    def predict(self, features: FeatureVector) -> List[Any]:
        """Make predictions."""
        pass

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Get model metadata."""
        pass


class SimpleClassifier(MLModel):
    """Simple rule-based classifier for demonstration."""

    def __init__(
        self,
        model_id: str,
        rules: Dict[str, Callable[[Dict[str, Any]], bool]],
    ) -> None:
        self._model_id = model_id
        self._rules = rules
        self._metadata = ModelMetadata(
            model_id=model_id,
            name="simple_classifier",
            version="1.0.0",
            model_type=ModelType.CLASSIFICATION,
            status=ModelStatus.READY,
        )

    def predict(self, features: FeatureVector) -> List[Any]:
        """Make predictions based on rules."""
        results = []
        for label, rule in self._rules.items():
            if rule(features.features):
                results.append(label)
        return results if results else ["unknown"]

    def get_metadata(self) -> ModelMetadata:
        """Get model metadata."""
        return self._metadata


class EnsembleModel(MLModel):
    """Ensemble of multiple models."""

    def __init__(
        self,
        model_id: str,
        models: List[MLModel],
        aggregation: str = "voting",
    ) -> None:
        self._model_id = model_id
        self._models = models
        self._aggregation = aggregation
        self._metadata = ModelMetadata(
            model_id=model_id,
            name="ensemble_model",
            version="1.0.0",
            model_type=ModelType.CLASSIFICATION,
            status=ModelStatus.READY,
        )

    def predict(self, features: FeatureVector) -> List[Any]:
        """Make ensemble predictions."""
        all_predictions: List[List[Any]] = []
        for model in self._models:
            preds = model.predict(features)
            all_predictions.append(preds)

        if self._aggregation == "voting":
            # Simple voting
            from collections import Counter

            flat = [p for preds in all_predictions for p in preds]
            counter = Counter(flat)
            return [counter.most_common(1)[0][0]] if counter else []

        return all_predictions[0] if all_predictions else []

    def get_metadata(self) -> ModelMetadata:
        """Get model metadata."""
        return self._metadata


# ============================================================================
# Model Registry
# ============================================================================


class ModelRegistry:
    """Registry for managing ML models."""

    def __init__(self, store: Optional[ModelStore] = None) -> None:
        self._store = store or InMemoryModelStore()
        self._active_models: Dict[str, MLModel] = {}
        self._versions: Dict[str, List[ModelVersion]] = {}

    def register_model(
        self,
        model: MLModel,
        model_data: bytes = b"",
    ) -> str:
        """Register a model."""
        metadata = model.get_metadata()
        self._store.save_model(metadata, model_data)
        self._active_models[metadata.model_id] = model

        # Track version
        version = ModelVersion(
            version_id=f"{metadata.model_id}_v{metadata.version}",
            model_id=metadata.model_id,
            version=metadata.version,
            is_current=True,
            metrics=metadata.metrics,
        )

        if metadata.model_id not in self._versions:
            self._versions[metadata.model_id] = []

        # Mark old versions as not current
        for v in self._versions[metadata.model_id]:
            v.is_current = False

        self._versions[metadata.model_id].append(version)
        return metadata.model_id

    def get_model(self, model_id: str) -> Optional[MLModel]:
        """Get an active model."""
        return self._active_models.get(model_id)

    def get_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata."""
        return self._store.get_metadata(model_id)

    def list_models(self, model_type: Optional[ModelType] = None) -> List[ModelMetadata]:
        """List all registered models."""
        return self._store.list_models(model_type)

    def get_versions(self, model_id: str) -> List[ModelVersion]:
        """Get all versions of a model."""
        return self._versions.get(model_id, [])

    def get_current_version(self, model_id: str) -> Optional[ModelVersion]:
        """Get the current version of a model."""
        versions = self._versions.get(model_id, [])
        for v in versions:
            if v.is_current:
                return v
        return None

    def deprecate_model(self, model_id: str) -> bool:
        """Deprecate a model."""
        metadata = self._store.get_metadata(model_id)
        if metadata:
            metadata.status = ModelStatus.DEPRECATED
            return True
        return False

    def delete_model(self, model_id: str) -> bool:
        """Delete a model."""
        if model_id in self._active_models:
            del self._active_models[model_id]
        if model_id in self._versions:
            del self._versions[model_id]
        return self._store.delete_model(model_id)


# ============================================================================
# Inference Pipeline
# ============================================================================


class InferencePipeline:
    """Pipeline for model inference."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._stages: List[PipelineStage] = config.stages.copy()

    def add_stage(self, stage: PipelineStage) -> "InferencePipeline":
        """Add a processing stage."""
        self._stages.append(stage)
        return self

    def execute(self, data: Any) -> Any:
        """Execute the pipeline."""
        result = data
        for stage in self._stages:
            result = stage.processor(result)
        return result

    def get_config(self) -> PipelineConfig:
        """Get pipeline configuration."""
        return self._config


class InferenceEngine:
    """Engine for running model inference."""

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry
        self._pipelines: Dict[str, InferencePipeline] = {}

    def register_pipeline(
        self,
        model_id: str,
        pipeline: InferencePipeline,
    ) -> None:
        """Register a pipeline for a model."""
        self._pipelines[model_id] = pipeline

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Run inference."""
        start_time = datetime.utcnow()

        model = self._registry.get_model(request.model_id)
        if not model:
            raise ValueError(f"Model not found: {request.model_id}")

        # Apply pipeline preprocessing if exists
        features = request.features
        if request.model_id in self._pipelines:
            pipeline = self._pipelines[request.model_id]
            features = pipeline.execute(features)

        # Run prediction
        predictions = model.predict(features)

        # Calculate latency
        latency = (datetime.utcnow() - start_time).total_seconds() * 1000

        return InferenceResult(
            request_id=request.request_id,
            model_id=request.model_id,
            predictions=predictions,
            confidence=0.9,  # Placeholder
            latency_ms=latency,
        )

    def batch_infer(
        self,
        requests: List[InferenceRequest],
    ) -> List[InferenceResult]:
        """Run batch inference."""
        return [self.infer(req) for req in requests]


# ============================================================================
# ML Vision Provider
# ============================================================================


class MLVisionProvider(VisionProvider):
    """Vision provider with ML model integration."""

    def __init__(
        self,
        provider: VisionProvider,
        registry: ModelRegistry,
        model_id: str,
        extractor: Optional[FeatureExtractor] = None,
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._model_id = model_id
        self._extractor = extractor or ImageFeatureExtractor()

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"ml_{self._provider.provider_name}"

    async def analyze_image(
        self,
        image_data: bytes,
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> VisionDescription:
        """Analyze image with ML enhancement."""
        # Get base analysis
        result = await self._provider.analyze_image(image_data, prompt, **kwargs)

        # Extract features and run ML model
        features = self._extractor.extract(image_data)
        model = self._registry.get_model(self._model_id)

        if model:
            predictions = model.predict(features)
            # Enhance details with predictions
            ml_details = [f"ML prediction: {p}" for p in predictions]
            result = VisionDescription(
                summary=result.summary,
                details=result.details + ml_details,
                confidence=result.confidence,
            )

        return result

    def get_registry(self) -> ModelRegistry:
        """Get the model registry."""
        return self._registry


# ============================================================================
# Factory Functions
# ============================================================================


def create_model_registry(store: Optional[ModelStore] = None) -> ModelRegistry:
    """Create a model registry."""
    return ModelRegistry(store)


def create_inference_engine(registry: ModelRegistry) -> InferenceEngine:
    """Create an inference engine."""
    return InferenceEngine(registry)


def create_ml_provider(
    provider: VisionProvider,
    registry: ModelRegistry,
    model_id: str,
    extractor: Optional[FeatureExtractor] = None,
) -> MLVisionProvider:
    """Create an ML vision provider."""
    return MLVisionProvider(provider, registry, model_id, extractor)


def create_feature_extractor(
    extractor_type: str = "image",
) -> FeatureExtractor:
    """Create a feature extractor."""
    if extractor_type == "image":
        return ImageFeatureExtractor()
    return ImageFeatureExtractor()


def create_simple_classifier(
    model_id: str,
    rules: Dict[str, Callable[[Dict[str, Any]], bool]],
) -> SimpleClassifier:
    """Create a simple rule-based classifier."""
    return SimpleClassifier(model_id, rules)


def create_ensemble_model(
    model_id: str,
    models: List[MLModel],
    aggregation: str = "voting",
) -> EnsembleModel:
    """Create an ensemble model."""
    return EnsembleModel(model_id, models, aggregation)
