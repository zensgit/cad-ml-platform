"""
Model registry for experiment tracking.

Provides:
- Model version management
- Model staging (dev -> staging -> production)
- Model metadata tracking
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelStage(str, Enum):
    """Model deployment stage."""
    NONE = "none"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelVersion:
    """A specific version of a model."""
    version: str
    model_path: str
    stage: ModelStage = ModelStage.NONE
    created_at: datetime = field(default_factory=datetime.utcnow)
    metrics: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    run_id: Optional[str] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "model_path": self.model_path,
            "stage": self.stage.value,
            "created_at": self.created_at.isoformat(),
            "metrics": self.metrics,
            "params": self.params,
            "tags": self.tags,
            "run_id": self.run_id,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        """Create from dictionary."""
        return cls(
            version=data["version"],
            model_path=data["model_path"],
            stage=ModelStage(data.get("stage", "none")),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            metrics=data.get("metrics", {}),
            params=data.get("params", {}),
            tags=data.get("tags", {}),
            run_id=data.get("run_id"),
            description=data.get("description", ""),
        )


@dataclass
class ModelInfo:
    """Information about a registered model."""
    name: str
    versions: List[ModelVersion] = field(default_factory=list)
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def latest_version(self) -> Optional[ModelVersion]:
        """Get latest version."""
        if not self.versions:
            return None
        return max(self.versions, key=lambda v: v.created_at)

    @property
    def production_version(self) -> Optional[ModelVersion]:
        """Get production version."""
        for v in self.versions:
            if v.stage == ModelStage.PRODUCTION:
                return v
        return None

    @property
    def staging_version(self) -> Optional[ModelVersion]:
        """Get staging version."""
        for v in self.versions:
            if v.stage == ModelStage.STAGING:
                return v
        return None

    def get_version(self, version: str) -> Optional[ModelVersion]:
        """Get specific version."""
        for v in self.versions:
            if v.version == version:
                return v
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "versions": [v.to_dict() for v in self.versions],
            "description": self.description,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInfo":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            versions=[ModelVersion.from_dict(v) for v in data.get("versions", [])],
            description=data.get("description", ""),
            tags=data.get("tags", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.utcnow(),
        )


class ModelRegistry:
    """
    Registry for managing model versions.

    Supports:
    - Model registration with versions
    - Stage transitions (dev -> staging -> production)
    - Model lookup by name/version/stage
    - Metric-based model comparison
    """

    def __init__(self, base_path: Path):
        """
        Initialize model registry.

        Args:
            base_path: Base directory for model storage
        """
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, ModelInfo] = {}
        self._registry_path = self._base_path / "registry.json"

        if self._registry_path.exists():
            self._load_registry()

    @property
    def model_names(self) -> List[str]:
        """Get all model names."""
        return list(self._models.keys())

    def _generate_version(self, model_name: str) -> str:
        """Generate next version number."""
        model = self._models.get(model_name)
        if model is None or not model.versions:
            return "1"

        # Find max version number
        max_version = 0
        for v in model.versions:
            try:
                num = int(v.version)
                max_version = max(max_version, num)
            except ValueError:
                pass

        return str(max_version + 1)

    def register(
        self,
        name: str,
        model_path: str,
        version: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        run_id: Optional[str] = None,
        description: str = "",
        copy_model: bool = True,
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            name: Model name
            model_path: Path to model file
            version: Version string (auto-generated if not provided)
            metrics: Model metrics
            params: Model parameters
            tags: Model tags
            run_id: Associated run ID
            description: Version description
            copy_model: Whether to copy model to registry storage

        Returns:
            ModelVersion object
        """
        src_path = Path(model_path)
        if not src_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Get or create model info
        if name not in self._models:
            self._models[name] = ModelInfo(name=name)

        model_info = self._models[name]

        # Generate version if not provided
        version = version or self._generate_version(name)

        # Check for duplicate version
        if model_info.get_version(version) is not None:
            raise ValueError(f"Version {version} already exists for model {name}")

        # Copy model to registry
        if copy_model:
            dst_dir = self._base_path / name / version
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_dir / src_path.name
            shutil.copy2(src_path, dst_path)
            stored_path = str(dst_path)
        else:
            stored_path = str(src_path)

        # Create version
        model_version = ModelVersion(
            version=version,
            model_path=stored_path,
            metrics=metrics or {},
            params=params or {},
            tags=tags or {},
            run_id=run_id,
            description=description,
        )

        model_info.versions.append(model_version)
        model_info.updated_at = datetime.utcnow()

        self._save_registry()

        logger.info(f"Registered model: {name} version {version}")
        return model_version

    def get_model(self, name: str) -> Optional[ModelInfo]:
        """
        Get model info.

        Args:
            name: Model name

        Returns:
            ModelInfo or None
        """
        return self._models.get(name)

    def get_version(
        self,
        name: str,
        version: str = "latest",
    ) -> Optional[ModelVersion]:
        """
        Get specific model version.

        Args:
            name: Model name
            version: Version string, "latest", "production", or "staging"

        Returns:
            ModelVersion or None
        """
        model = self._models.get(name)
        if model is None:
            return None

        if version == "latest":
            return model.latest_version
        elif version == "production":
            return model.production_version
        elif version == "staging":
            return model.staging_version
        else:
            return model.get_version(version)

    def get_model_path(
        self,
        name: str,
        version: str = "latest",
    ) -> Optional[str]:
        """
        Get model file path.

        Args:
            name: Model name
            version: Version string

        Returns:
            Model path or None
        """
        model_version = self.get_version(name, version)
        return model_version.model_path if model_version else None

    def transition_stage(
        self,
        name: str,
        version: str,
        stage: ModelStage,
    ) -> bool:
        """
        Transition model to new stage.

        Args:
            name: Model name
            version: Version string
            stage: Target stage

        Returns:
            True if successful
        """
        model = self._models.get(name)
        if model is None:
            return False

        model_version = model.get_version(version)
        if model_version is None:
            return False

        # If promoting to production/staging, demote current version
        if stage in (ModelStage.PRODUCTION, ModelStage.STAGING):
            for v in model.versions:
                if v.stage == stage:
                    v.stage = ModelStage.ARCHIVED

        model_version.stage = stage
        model.updated_at = datetime.utcnow()

        self._save_registry()

        logger.info(f"Transitioned model {name} v{version} to {stage.value}")
        return True

    def promote_to_staging(self, name: str, version: str) -> bool:
        """Promote model version to staging."""
        return self.transition_stage(name, version, ModelStage.STAGING)

    def promote_to_production(self, name: str, version: str) -> bool:
        """Promote model version to production."""
        return self.transition_stage(name, version, ModelStage.PRODUCTION)

    def archive(self, name: str, version: str) -> bool:
        """Archive model version."""
        return self.transition_stage(name, version, ModelStage.ARCHIVED)

    def delete_version(self, name: str, version: str) -> bool:
        """
        Delete a model version.

        Args:
            name: Model name
            version: Version string

        Returns:
            True if deleted
        """
        model = self._models.get(name)
        if model is None:
            return False

        model_version = model.get_version(version)
        if model_version is None:
            return False

        # Don't delete production models
        if model_version.stage == ModelStage.PRODUCTION:
            raise ValueError("Cannot delete production model version")

        # Remove model file
        model_path = Path(model_version.model_path)
        if model_path.exists() and str(self._base_path) in str(model_path):
            model_path.unlink()
            # Remove parent if empty
            if model_path.parent.exists() and not any(model_path.parent.iterdir()):
                model_path.parent.rmdir()

        model.versions = [v for v in model.versions if v.version != version]
        model.updated_at = datetime.utcnow()

        self._save_registry()

        logger.info(f"Deleted model {name} version {version}")
        return True

    def compare_versions(
        self,
        name: str,
        versions: Optional[List[str]] = None,
        metric: str = "accuracy",
    ) -> List[Dict[str, Any]]:
        """
        Compare model versions by metric.

        Args:
            name: Model name
            versions: Versions to compare (all if None)
            metric: Metric to compare

        Returns:
            List of version comparisons
        """
        model = self._models.get(name)
        if model is None:
            return []

        target_versions = model.versions
        if versions:
            target_versions = [v for v in model.versions if v.version in versions]

        comparisons = []
        for v in target_versions:
            metric_value = v.metrics.get(metric)
            comparisons.append({
                "version": v.version,
                "stage": v.stage.value,
                "metric": metric,
                "value": metric_value,
                "created_at": v.created_at.isoformat(),
            })

        # Sort by metric value (descending)
        comparisons.sort(key=lambda x: x["value"] if x["value"] is not None else float("-inf"), reverse=True)
        return comparisons

    def search(
        self,
        name: Optional[str] = None,
        stage: Optional[ModelStage] = None,
        tag_key: Optional[str] = None,
        tag_value: Optional[str] = None,
    ) -> List[ModelVersion]:
        """
        Search for model versions.

        Args:
            name: Model name filter
            stage: Stage filter
            tag_key: Tag key filter
            tag_value: Tag value filter

        Returns:
            List of matching versions
        """
        results = []

        models = [self._models[name]] if name and name in self._models else self._models.values()

        for model in models:
            for version in model.versions:
                if stage and version.stage != stage:
                    continue
                if tag_key and tag_key not in version.tags:
                    continue
                if tag_value and version.tags.get(tag_key) != tag_value:
                    continue
                results.append(version)

        return results

    def list_models(self) -> List[ModelInfo]:
        """List all models."""
        return list(self._models.values())

    def _save_registry(self) -> None:
        """Save registry to file."""
        data = {
            name: model.to_dict()
            for name, model in self._models.items()
        }
        with open(self._registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_registry(self) -> None:
        """Load registry from file."""
        with open(self._registry_path) as f:
            data = json.load(f)

        self._models = {
            name: ModelInfo.from_dict(model_data)
            for name, model_data in data.items()
        }

    def __repr__(self) -> str:
        return f"ModelRegistry(base_path={self._base_path}, models={len(self._models)})"
