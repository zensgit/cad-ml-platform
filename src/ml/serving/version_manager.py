"""
Model version management for production deployment.

Provides:
- Model versioning and registration
- Version lifecycle management
- Rollback capabilities
- Version comparison and promotion
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class ModelStage(str, Enum):
    """Model lifecycle stage."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class VersionStatus(str, Enum):
    """Version status."""
    REGISTERED = "registered"
    VALIDATING = "validating"
    VALIDATED = "validated"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class ModelVersion:
    """Model version metadata."""
    model_name: str
    version: str
    stage: ModelStage = ModelStage.DEVELOPMENT
    status: VersionStatus = VersionStatus.REGISTERED
    artifact_path: Optional[str] = None
    artifact_hash: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    parent_version: Optional[str] = None
    run_id: Optional[str] = None

    @property
    def full_name(self) -> str:
        return f"{self.model_name}:{self.version}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "stage": self.stage.value,
            "status": self.status.value,
            "artifact_path": self.artifact_path,
            "artifact_hash": self.artifact_hash,
            "metrics": self.metrics,
            "parameters": self.parameters,
            "tags": self.tags,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "parent_version": self.parent_version,
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        return cls(
            model_name=data["model_name"],
            version=data["version"],
            stage=ModelStage(data.get("stage", "development")),
            status=VersionStatus(data.get("status", "registered")),
            artifact_path=data.get("artifact_path"),
            artifact_hash=data.get("artifact_hash"),
            metrics=data.get("metrics", {}),
            parameters=data.get("parameters", {}),
            tags=data.get("tags", {}),
            description=data.get("description", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(),
            created_by=data.get("created_by", ""),
            parent_version=data.get("parent_version"),
            run_id=data.get("run_id"),
        )


@dataclass
class RegisteredModel:
    """Registered model with multiple versions."""
    name: str
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    versions: Dict[str, ModelVersion] = field(default_factory=dict)
    latest_version: Optional[str] = None
    production_version: Optional[str] = None
    staging_version: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_version(self, version: ModelVersion) -> None:
        """Add a version to this model."""
        self.versions[version.version] = version
        self.latest_version = version.version
        self.updated_at = datetime.now()

    def get_version(self, version: str) -> Optional[ModelVersion]:
        """Get a specific version."""
        return self.versions.get(version)

    def list_versions(self, stage: Optional[ModelStage] = None) -> List[ModelVersion]:
        """List versions, optionally filtered by stage."""
        versions = list(self.versions.values())
        if stage:
            versions = [v for v in versions if v.stage == stage]
        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "versions": {k: v.to_dict() for k, v in self.versions.items()},
            "latest_version": self.latest_version,
            "production_version": self.production_version,
            "staging_version": self.staging_version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class VersionManagerConfig:
    """Version manager configuration."""
    registry_path: str = "./model_registry"
    artifact_store_path: str = "./model_artifacts"
    max_versions_per_model: int = 100
    auto_archive_old_versions: bool = True
    validation_required: bool = True
    promotion_requires_validation: bool = True


class ModelVersionManager:
    """
    Model version manager for production deployments.

    Handles:
    - Version registration and tracking
    - Stage transitions (dev → staging → production)
    - Artifact management
    - Rollback capabilities
    """

    def __init__(self, config: Optional[VersionManagerConfig] = None):
        """
        Initialize version manager.

        Args:
            config: Manager configuration
        """
        self._config = config or VersionManagerConfig()
        self._registry: Dict[str, RegisteredModel] = {}
        self._lock = threading.RLock()
        self._callbacks: Dict[str, List[Callable]] = {
            "on_register": [],
            "on_promote": [],
            "on_rollback": [],
            "on_archive": [],
        }

        # Setup directories
        self._registry_path = Path(self._config.registry_path)
        self._artifact_path = Path(self._config.artifact_store_path)
        self._registry_path.mkdir(parents=True, exist_ok=True)
        self._artifact_path.mkdir(parents=True, exist_ok=True)

        # Load existing registry
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from disk."""
        registry_file = self._registry_path / "registry.json"
        if registry_file.exists():
            try:
                with open(registry_file) as f:
                    data = json.load(f)

                for model_data in data.get("models", []):
                    model = RegisteredModel(
                        name=model_data["name"],
                        description=model_data.get("description", ""),
                        tags=model_data.get("tags", {}),
                        latest_version=model_data.get("latest_version"),
                        production_version=model_data.get("production_version"),
                        staging_version=model_data.get("staging_version"),
                    )

                    for version_data in model_data.get("versions", {}).values():
                        version = ModelVersion.from_dict(version_data)
                        model.versions[version.version] = version

                    self._registry[model.name] = model

                logger.info(f"Loaded {len(self._registry)} models from registry")
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")

    def _save_registry(self) -> None:
        """Save registry to disk."""
        registry_file = self._registry_path / "registry.json"

        data = {
            "models": [model.to_dict() for model in self._registry.values()],
            "updated_at": datetime.now().isoformat(),
        }

        with open(registry_file, "w") as f:
            json.dump(data, f, indent=2)

    def register_model(
        self,
        name: str,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> RegisteredModel:
        """
        Register a new model.

        Args:
            name: Model name
            description: Model description
            tags: Model tags

        Returns:
            RegisteredModel
        """
        with self._lock:
            if name in self._registry:
                return self._registry[name]

            model = RegisteredModel(
                name=name,
                description=description,
                tags=tags or {},
            )

            self._registry[name] = model
            self._save_registry()

            logger.info(f"Registered model: {name}")
            return model

    def register_version(
        self,
        model_name: str,
        version: str,
        artifact_path: Optional[Union[str, Path]] = None,
        metrics: Optional[Dict[str, float]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        description: str = "",
        created_by: str = "",
        parent_version: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            model_name: Model name
            version: Version string
            artifact_path: Path to model artifacts
            metrics: Model metrics
            parameters: Training parameters
            tags: Version tags
            description: Version description
            created_by: Creator identifier
            parent_version: Parent version (for lineage)
            run_id: Associated experiment run ID

        Returns:
            ModelVersion
        """
        with self._lock:
            # Ensure model exists
            if model_name not in self._registry:
                self.register_model(model_name)

            model = self._registry[model_name]

            # Check if version already exists
            if version in model.versions:
                raise ValueError(f"Version {version} already exists for model {model_name}")

            # Calculate artifact hash if path provided
            artifact_hash = None
            stored_path = None

            if artifact_path:
                artifact_path = Path(artifact_path)
                if artifact_path.exists():
                    artifact_hash = self._calculate_hash(artifact_path)
                    stored_path = self._store_artifact(model_name, version, artifact_path)

            # Create version
            model_version = ModelVersion(
                model_name=model_name,
                version=version,
                artifact_path=str(stored_path) if stored_path else None,
                artifact_hash=artifact_hash,
                metrics=metrics or {},
                parameters=parameters or {},
                tags=tags or {},
                description=description,
                created_by=created_by,
                parent_version=parent_version,
                run_id=run_id,
            )

            model.add_version(model_version)
            self._save_registry()

            # Trigger callbacks
            self._trigger_callbacks("on_register", model_version)

            logger.info(f"Registered version: {model_name}:{version}")
            return model_version

    def _store_artifact(
        self,
        model_name: str,
        version: str,
        source_path: Path,
    ) -> Path:
        """Store model artifact."""
        dest_dir = self._artifact_path / model_name / version
        dest_dir.mkdir(parents=True, exist_ok=True)

        if source_path.is_dir():
            dest_path = dest_dir / source_path.name
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.copytree(source_path, dest_path)
        else:
            dest_path = dest_dir / source_path.name
            shutil.copy2(source_path, dest_path)

        return dest_path

    def _calculate_hash(self, path: Path) -> str:
        """Calculate artifact hash."""
        hasher = hashlib.sha256()

        if path.is_file():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
        else:
            # Hash directory contents
            for file_path in sorted(path.rglob("*")):
                if file_path.is_file():
                    hasher.update(str(file_path.relative_to(path)).encode())
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            hasher.update(chunk)

        return hasher.hexdigest()

    def get_model(self, name: str) -> Optional[RegisteredModel]:
        """Get registered model."""
        return self._registry.get(name)

    def get_version(self, model_name: str, version: str) -> Optional[ModelVersion]:
        """Get specific version."""
        model = self._registry.get(model_name)
        if model:
            return model.get_version(version)
        return None

    def get_latest_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get latest version of a model."""
        model = self._registry.get(model_name)
        if model and model.latest_version:
            return model.get_version(model.latest_version)
        return None

    def get_production_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get production version of a model."""
        model = self._registry.get(model_name)
        if model and model.production_version:
            return model.get_version(model.production_version)
        return None

    def promote_version(
        self,
        model_name: str,
        version: str,
        stage: ModelStage,
        validated: bool = False,
    ) -> ModelVersion:
        """
        Promote version to a new stage.

        Args:
            model_name: Model name
            version: Version to promote
            stage: Target stage
            validated: Whether version has been validated

        Returns:
            Updated ModelVersion
        """
        with self._lock:
            model = self._registry.get(model_name)
            if not model:
                raise ValueError(f"Model not found: {model_name}")

            model_version = model.get_version(version)
            if not model_version:
                raise ValueError(f"Version not found: {version}")

            # Check validation requirement
            if self._config.promotion_requires_validation and stage in (ModelStage.STAGING, ModelStage.PRODUCTION):
                if not validated and model_version.status != VersionStatus.VALIDATED:
                    raise ValueError(f"Version must be validated before promotion to {stage.value}")

            # Update version stage
            old_stage = model_version.stage
            model_version.stage = stage
            model_version.status = VersionStatus.DEPLOYED
            model_version.updated_at = datetime.now()

            # Update model pointers
            if stage == ModelStage.PRODUCTION:
                # Archive previous production version
                if model.production_version and model.production_version != version:
                    prev_version = model.get_version(model.production_version)
                    if prev_version:
                        prev_version.stage = ModelStage.ARCHIVED

                model.production_version = version
            elif stage == ModelStage.STAGING:
                model.staging_version = version

            self._save_registry()

            # Trigger callbacks
            self._trigger_callbacks("on_promote", model_version, old_stage, stage)

            logger.info(f"Promoted {model_name}:{version} to {stage.value}")
            return model_version

    def rollback(
        self,
        model_name: str,
        target_version: Optional[str] = None,
    ) -> ModelVersion:
        """
        Rollback to a previous version.

        Args:
            model_name: Model name
            target_version: Target version (previous production if None)

        Returns:
            Rolled back version
        """
        with self._lock:
            model = self._registry.get(model_name)
            if not model:
                raise ValueError(f"Model not found: {model_name}")

            current_prod = model.production_version

            if target_version:
                target = model.get_version(target_version)
                if not target:
                    raise ValueError(f"Version not found: {target_version}")
            else:
                # Find previous production version
                versions = model.list_versions()
                archived = [v for v in versions if v.stage == ModelStage.ARCHIVED]
                if not archived:
                    raise ValueError("No previous version to rollback to")
                target = archived[0]

            # Mark current as rolled back
            if current_prod:
                current = model.get_version(current_prod)
                if current:
                    current.status = VersionStatus.ROLLED_BACK
                    current.updated_at = datetime.now()

            # Promote target to production
            target.stage = ModelStage.PRODUCTION
            target.status = VersionStatus.DEPLOYED
            target.updated_at = datetime.now()
            model.production_version = target.version

            self._save_registry()

            # Trigger callbacks
            self._trigger_callbacks("on_rollback", target, current_prod)

            logger.info(f"Rolled back {model_name} to version {target.version}")
            return target

    def validate_version(
        self,
        model_name: str,
        version: str,
        validation_metrics: Optional[Dict[str, float]] = None,
    ) -> ModelVersion:
        """
        Mark a version as validated.

        Args:
            model_name: Model name
            version: Version to validate
            validation_metrics: Validation metrics

        Returns:
            Updated ModelVersion
        """
        with self._lock:
            model_version = self.get_version(model_name, version)
            if not model_version:
                raise ValueError(f"Version not found: {model_name}:{version}")

            model_version.status = VersionStatus.VALIDATED
            model_version.updated_at = datetime.now()

            if validation_metrics:
                model_version.metrics.update(validation_metrics)

            self._save_registry()

            logger.info(f"Validated version: {model_name}:{version}")
            return model_version

    def compare_versions(
        self,
        model_name: str,
        version_a: str,
        version_b: str,
    ) -> Dict[str, Any]:
        """
        Compare two versions.

        Args:
            model_name: Model name
            version_a: First version
            version_b: Second version

        Returns:
            Comparison results
        """
        v_a = self.get_version(model_name, version_a)
        v_b = self.get_version(model_name, version_b)

        if not v_a or not v_b:
            raise ValueError("One or both versions not found")

        # Compare metrics
        metric_diff = {}
        all_metrics = set(v_a.metrics.keys()) | set(v_b.metrics.keys())
        for metric in all_metrics:
            val_a = v_a.metrics.get(metric)
            val_b = v_b.metrics.get(metric)
            if val_a is not None and val_b is not None:
                metric_diff[metric] = {
                    "version_a": val_a,
                    "version_b": val_b,
                    "diff": val_b - val_a,
                    "diff_percent": ((val_b - val_a) / val_a * 100) if val_a != 0 else 0,
                }

        # Compare parameters
        param_diff = {}
        all_params = set(v_a.parameters.keys()) | set(v_b.parameters.keys())
        for param in all_params:
            val_a = v_a.parameters.get(param)
            val_b = v_b.parameters.get(param)
            if val_a != val_b:
                param_diff[param] = {"version_a": val_a, "version_b": val_b}

        return {
            "version_a": v_a.to_dict(),
            "version_b": v_b.to_dict(),
            "metric_comparison": metric_diff,
            "parameter_changes": param_diff,
            "same_artifact": v_a.artifact_hash == v_b.artifact_hash,
        }

    def list_models(self) -> List[RegisteredModel]:
        """List all registered models."""
        return list(self._registry.values())

    def list_versions(
        self,
        model_name: str,
        stage: Optional[ModelStage] = None,
    ) -> List[ModelVersion]:
        """List versions for a model."""
        model = self._registry.get(model_name)
        if not model:
            return []
        return model.list_versions(stage)

    def delete_version(self, model_name: str, version: str) -> bool:
        """Delete a version."""
        with self._lock:
            model = self._registry.get(model_name)
            if not model or version not in model.versions:
                return False

            model_version = model.versions[version]

            # Cannot delete production version
            if model_version.stage == ModelStage.PRODUCTION:
                raise ValueError("Cannot delete production version")

            # Delete artifact
            if model_version.artifact_path:
                artifact_path = Path(model_version.artifact_path)
                if artifact_path.exists():
                    if artifact_path.is_dir():
                        shutil.rmtree(artifact_path)
                    else:
                        artifact_path.unlink()

            del model.versions[version]
            self._save_registry()

            logger.info(f"Deleted version: {model_name}:{version}")
            return True

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register event callback."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _trigger_callbacks(self, event: str, *args, **kwargs) -> None:
        """Trigger event callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Callback error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total_versions = sum(len(m.versions) for m in self._registry.values())
        production_models = sum(1 for m in self._registry.values() if m.production_version)

        return {
            "total_models": len(self._registry),
            "total_versions": total_versions,
            "production_models": production_models,
            "registry_path": str(self._registry_path),
            "artifact_path": str(self._artifact_path),
        }


def get_version_manager(config: Optional[VersionManagerConfig] = None) -> ModelVersionManager:
    """Get or create version manager singleton."""
    global _version_manager
    if "_version_manager" not in globals():
        _version_manager = ModelVersionManager(config)
    return _version_manager
