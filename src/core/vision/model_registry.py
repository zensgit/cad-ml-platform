"""Model Registry Module for Vision System.

This module provides model lifecycle management including:
- Model versioning and metadata management
- Model deployment and serving
- A/B testing and traffic routing
- Model lineage and provenance tracking
- Model validation and approval workflows
- Rollback and canary deployments

Phase 18: Advanced ML Pipeline & AutoML
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from .base import VisionDescription, VisionProvider


# ========================
# Enums
# ========================


class ModelStage(str, Enum):
    """Model lifecycle stages."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelStatus(str, Enum):
    """Model status."""

    PENDING = "pending"
    TRAINING = "training"
    VALIDATING = "validating"
    READY = "ready"
    DEPLOYED = "deployed"
    FAILED = "failed"
    RETIRED = "retired"


class DeploymentStrategy(str, Enum):
    """Model deployment strategies."""

    DIRECT = "direct"
    CANARY = "canary"
    BLUE_GREEN = "blue_green"
    SHADOW = "shadow"
    A_B_TEST = "ab_test"


class ServingFormat(str, Enum):
    """Model serving formats."""

    PICKLE = "pickle"
    ONNX = "onnx"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CUSTOM = "custom"


class ApprovalStatus(str, Enum):
    """Model approval status."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class MetricType(str, Enum):
    """Model metric types."""

    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC = "auc"
    MSE = "mse"
    MAE = "mae"
    RMSE = "rmse"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    CUSTOM = "custom"


# ========================
# Dataclasses
# ========================


@dataclass
class ModelVersion:
    """A specific version of a model."""

    model_id: str
    version: str
    name: str
    description: str = ""
    stage: ModelStage = ModelStage.DEVELOPMENT
    status: ModelStatus = ModelStatus.PENDING
    artifact_path: Optional[str] = None
    serving_format: ServingFormat = ServingFormat.PICKLE
    metrics: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"


@dataclass
class ModelMetadata:
    """Metadata for a model."""

    model_id: str
    name: str
    description: str = ""
    owner: str = "system"
    team: str = "default"
    task_type: str = "classification"
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ModelLineage:
    """Lineage information for a model."""

    model_id: str
    version: str
    parent_model: Optional[str] = None
    parent_version: Optional[str] = None
    training_dataset: Optional[str] = None
    training_run_id: Optional[str] = None
    feature_set: List[str] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""

    deployment_id: str
    model_id: str
    version: str
    strategy: DeploymentStrategy
    target_environment: str = "production"
    replicas: int = 1
    resources: Dict[str, Any] = field(default_factory=dict)
    traffic_percentage: float = 100.0
    canary_steps: List[float] = field(default_factory=list)
    rollback_on_failure: bool = True
    health_check_path: str = "/health"
    timeout_seconds: int = 30


@dataclass
class Deployment:
    """A model deployment."""

    deployment_id: str
    model_id: str
    version: str
    environment: str
    status: str = "pending"
    endpoint_url: Optional[str] = None
    traffic_percentage: float = 100.0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ABTestConfig:
    """A/B test configuration."""

    test_id: str
    name: str
    control_model: str
    control_version: str
    treatment_model: str
    treatment_version: str
    traffic_split: float = 0.5  # Percentage to treatment
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    success_metric: str = "conversion_rate"
    min_sample_size: int = 1000


@dataclass
class ABTestResult:
    """Result of an A/B test."""

    test_id: str
    control_metrics: Dict[str, float]
    treatment_metrics: Dict[str, float]
    sample_sizes: Dict[str, int]
    p_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    winner: Optional[str] = None
    is_significant: bool = False
    computed_at: datetime = field(default_factory=datetime.now)


@dataclass
class ApprovalRequest:
    """Model approval request."""

    request_id: str
    model_id: str
    version: str
    target_stage: ModelStage
    requestor: str
    reason: str = ""
    status: ApprovalStatus = ApprovalStatus.PENDING
    approvers: List[str] = field(default_factory=list)
    approved_by: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None


# ========================
# Core Classes
# ========================


class ModelRegistry:
    """Central registry for model management."""

    def __init__(self, storage_path: Optional[str] = None):
        self._storage_path = Path(storage_path) if storage_path else None
        self._models: Dict[str, ModelMetadata] = {}
        self._versions: Dict[str, Dict[str, ModelVersion]] = defaultdict(dict)
        self._lineage: Dict[str, Dict[str, ModelLineage]] = defaultdict(dict)
        self._lock = threading.RLock()

    def register_model(self, metadata: ModelMetadata) -> None:
        """Register a new model."""
        with self._lock:
            self._models[metadata.model_id] = metadata

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata."""
        return self._models.get(model_id)

    def list_models(
        self,
        owner: Optional[str] = None,
        team: Optional[str] = None,
    ) -> List[ModelMetadata]:
        """List registered models."""
        models = list(self._models.values())
        if owner:
            models = [m for m in models if m.owner == owner]
        if team:
            models = [m for m in models if m.team == team]
        return models

    def create_version(self, version: ModelVersion) -> None:
        """Create a new model version."""
        with self._lock:
            self._versions[version.model_id][version.version] = version

    def get_version(
        self,
        model_id: str,
        version: Optional[str] = None,
    ) -> Optional[ModelVersion]:
        """Get a model version."""
        with self._lock:
            versions = self._versions.get(model_id, {})
            if version is None:
                # Get latest version
                if not versions:
                    return None
                return max(versions.values(), key=lambda v: v.created_at)
            return versions.get(version)

    def list_versions(
        self,
        model_id: str,
        stage: Optional[ModelStage] = None,
    ) -> List[ModelVersion]:
        """List model versions."""
        with self._lock:
            versions = list(self._versions.get(model_id, {}).values())
            if stage:
                versions = [v for v in versions if v.stage == stage]
            return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def update_version(self, version: ModelVersion) -> None:
        """Update a model version."""
        with self._lock:
            version.updated_at = datetime.now()
            self._versions[version.model_id][version.version] = version

    def transition_stage(
        self,
        model_id: str,
        version: str,
        stage: ModelStage,
    ) -> bool:
        """Transition a model version to a new stage."""
        with self._lock:
            model_version = self.get_version(model_id, version)
            if model_version is None:
                return False

            # Validate transition
            valid_transitions = {
                ModelStage.DEVELOPMENT: [ModelStage.STAGING, ModelStage.ARCHIVED],
                ModelStage.STAGING: [ModelStage.PRODUCTION, ModelStage.DEVELOPMENT, ModelStage.ARCHIVED],
                ModelStage.PRODUCTION: [ModelStage.STAGING, ModelStage.ARCHIVED],
                ModelStage.ARCHIVED: [ModelStage.DEVELOPMENT],
            }

            if stage not in valid_transitions.get(model_version.stage, []):
                return False

            model_version.stage = stage
            model_version.updated_at = datetime.now()
            self._versions[model_id][version] = model_version
            return True

    def set_lineage(self, lineage: ModelLineage) -> None:
        """Set model lineage."""
        with self._lock:
            self._lineage[lineage.model_id][lineage.version] = lineage

    def get_lineage(
        self,
        model_id: str,
        version: str,
    ) -> Optional[ModelLineage]:
        """Get model lineage."""
        return self._lineage.get(model_id, {}).get(version)

    def delete_version(self, model_id: str, version: str) -> bool:
        """Delete a model version."""
        with self._lock:
            if model_id in self._versions and version in self._versions[model_id]:
                del self._versions[model_id][version]
                return True
            return False


class ModelDeployer:
    """Handle model deployments."""

    def __init__(self, registry: ModelRegistry):
        self._registry = registry
        self._deployments: Dict[str, Deployment] = {}
        self._active_deployments: Dict[str, str] = {}  # environment -> deployment_id
        self._lock = threading.RLock()

    def deploy(self, config: DeploymentConfig) -> Deployment:
        """Deploy a model."""
        version = self._registry.get_version(config.model_id, config.version)
        if version is None:
            raise ValueError(f"Model version not found: {config.model_id}:{config.version}")

        deployment = Deployment(
            deployment_id=config.deployment_id,
            model_id=config.model_id,
            version=config.version,
            environment=config.target_environment,
            status="deploying",
            traffic_percentage=config.traffic_percentage,
            started_at=datetime.now(),
        )

        with self._lock:
            self._deployments[deployment.deployment_id] = deployment

        # Simulate deployment
        self._execute_deployment(deployment, config)

        return deployment

    def _execute_deployment(
        self,
        deployment: Deployment,
        config: DeploymentConfig,
    ) -> None:
        """Execute deployment based on strategy."""
        try:
            if config.strategy == DeploymentStrategy.DIRECT:
                self._deploy_direct(deployment)
            elif config.strategy == DeploymentStrategy.CANARY:
                self._deploy_canary(deployment, config.canary_steps)
            elif config.strategy == DeploymentStrategy.BLUE_GREEN:
                self._deploy_blue_green(deployment)
            else:
                self._deploy_direct(deployment)

            deployment.status = "running"
            deployment.completed_at = datetime.now()
            deployment.endpoint_url = f"http://model-{deployment.deployment_id}.internal/predict"

            with self._lock:
                self._active_deployments[deployment.environment] = deployment.deployment_id
                self._deployments[deployment.deployment_id] = deployment

        except Exception as e:
            deployment.status = "failed"
            deployment.completed_at = datetime.now()
            with self._lock:
                self._deployments[deployment.deployment_id] = deployment
            raise

    def _deploy_direct(self, deployment: Deployment) -> None:
        """Direct deployment - immediate replacement."""
        time.sleep(0.01)  # Simulate deployment time

    def _deploy_canary(
        self,
        deployment: Deployment,
        steps: List[float],
    ) -> None:
        """Canary deployment - gradual rollout."""
        for step in steps or [10, 25, 50, 100]:
            deployment.traffic_percentage = step
            time.sleep(0.001)  # Simulate step time

    def _deploy_blue_green(self, deployment: Deployment) -> None:
        """Blue-green deployment."""
        # Deploy to green environment
        time.sleep(0.01)
        # Switch traffic
        deployment.traffic_percentage = 100.0

    def get_deployment(self, deployment_id: str) -> Optional[Deployment]:
        """Get deployment by ID."""
        return self._deployments.get(deployment_id)

    def get_active_deployment(self, environment: str) -> Optional[Deployment]:
        """Get active deployment for an environment."""
        deployment_id = self._active_deployments.get(environment)
        if deployment_id:
            return self._deployments.get(deployment_id)
        return None

    def list_deployments(
        self,
        model_id: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> List[Deployment]:
        """List deployments."""
        deployments = list(self._deployments.values())
        if model_id:
            deployments = [d for d in deployments if d.model_id == model_id]
        if environment:
            deployments = [d for d in deployments if d.environment == environment]
        return deployments

    def rollback(
        self,
        deployment_id: str,
        target_version: Optional[str] = None,
    ) -> Optional[Deployment]:
        """Rollback a deployment."""
        current = self._deployments.get(deployment_id)
        if current is None:
            return None

        # Find previous version
        versions = self._registry.list_versions(current.model_id)
        current_idx = next(
            (i for i, v in enumerate(versions) if v.version == current.version),
            -1,
        )

        if target_version:
            rollback_version = target_version
        elif current_idx >= 0 and current_idx < len(versions) - 1:
            rollback_version = versions[current_idx + 1].version
        else:
            return None

        # Create rollback deployment
        rollback_config = DeploymentConfig(
            deployment_id=f"{deployment_id}-rollback-{int(time.time())}",
            model_id=current.model_id,
            version=rollback_version,
            strategy=DeploymentStrategy.DIRECT,
            target_environment=current.environment,
        )

        return self.deploy(rollback_config)

    def scale(self, deployment_id: str, replicas: int) -> bool:
        """Scale a deployment."""
        deployment = self._deployments.get(deployment_id)
        if deployment is None:
            return False
        # Scaling logic would go here
        return True


class ABTestManager:
    """Manage A/B tests for models."""

    def __init__(self, registry: ModelRegistry):
        self._registry = registry
        self._tests: Dict[str, ABTestConfig] = {}
        self._results: Dict[str, ABTestResult] = {}
        self._assignments: Dict[str, Dict[str, str]] = defaultdict(dict)  # test_id -> user_id -> variant
        self._lock = threading.RLock()

    def create_test(self, config: ABTestConfig) -> ABTestConfig:
        """Create an A/B test."""
        with self._lock:
            self._tests[config.test_id] = config
        return config

    def get_test(self, test_id: str) -> Optional[ABTestConfig]:
        """Get A/B test configuration."""
        return self._tests.get(test_id)

    def assign_variant(self, test_id: str, user_id: str) -> str:
        """Assign a user to a variant."""
        test = self._tests.get(test_id)
        if test is None:
            return "control"

        with self._lock:
            # Check existing assignment
            if user_id in self._assignments[test_id]:
                return self._assignments[test_id][user_id]

            # Random assignment based on traffic split
            import random
            variant = "treatment" if random.random() < test.traffic_split else "control"
            self._assignments[test_id][user_id] = variant
            return variant

    def get_model_for_user(
        self,
        test_id: str,
        user_id: str,
    ) -> Tuple[str, str]:
        """Get model and version for a user."""
        test = self._tests.get(test_id)
        if test is None:
            raise ValueError(f"Test not found: {test_id}")

        variant = self.assign_variant(test_id, user_id)

        if variant == "control":
            return test.control_model, test.control_version
        else:
            return test.treatment_model, test.treatment_version

    def record_outcome(
        self,
        test_id: str,
        user_id: str,
        metric: str,
        value: float,
    ) -> None:
        """Record an outcome for a user."""
        # In a real implementation, this would store outcomes for analysis
        pass

    def compute_results(self, test_id: str) -> ABTestResult:
        """Compute A/B test results."""
        test = self._tests.get(test_id)
        if test is None:
            raise ValueError(f"Test not found: {test_id}")

        with self._lock:
            control_users = [u for u, v in self._assignments[test_id].items() if v == "control"]
            treatment_users = [u for u, v in self._assignments[test_id].items() if v == "treatment"]

        # Simulated results
        import random
        control_metrics = {test.success_metric: random.uniform(0.1, 0.3)}
        treatment_metrics = {test.success_metric: random.uniform(0.1, 0.35)}

        # Simple statistical test
        control_rate = control_metrics[test.success_metric]
        treatment_rate = treatment_metrics[test.success_metric]
        difference = treatment_rate - control_rate

        result = ABTestResult(
            test_id=test_id,
            control_metrics=control_metrics,
            treatment_metrics=treatment_metrics,
            sample_sizes={
                "control": len(control_users),
                "treatment": len(treatment_users),
            },
            p_value=0.05 if abs(difference) > 0.05 else 0.5,  # Simplified
            confidence_interval=(difference - 0.02, difference + 0.02),
            winner="treatment" if difference > 0.05 else "control" if difference < -0.05 else None,
            is_significant=abs(difference) > 0.05,
        )

        self._results[test_id] = result
        return result

    def stop_test(self, test_id: str) -> Optional[ABTestConfig]:
        """Stop an A/B test."""
        test = self._tests.get(test_id)
        if test:
            test.end_time = datetime.now()
            self._tests[test_id] = test
        return test


class ApprovalWorkflow:
    """Model approval workflow management."""

    def __init__(self, registry: ModelRegistry):
        self._registry = registry
        self._requests: Dict[str, ApprovalRequest] = {}
        self._policies: Dict[ModelStage, List[str]] = {
            ModelStage.STAGING: [],
            ModelStage.PRODUCTION: ["ml-lead", "qa-lead"],
        }
        self._lock = threading.RLock()

    def request_approval(
        self,
        model_id: str,
        version: str,
        target_stage: ModelStage,
        requestor: str,
        reason: str = "",
    ) -> ApprovalRequest:
        """Request approval for stage transition."""
        request_id = hashlib.md5(
            f"{model_id}:{version}:{target_stage}:{time.time()}".encode()
        ).hexdigest()[:8]

        required_approvers = self._policies.get(target_stage, [])

        request = ApprovalRequest(
            request_id=request_id,
            model_id=model_id,
            version=version,
            target_stage=target_stage,
            requestor=requestor,
            reason=reason,
            approvers=required_approvers,
        )

        with self._lock:
            self._requests[request_id] = request

        return request

    def approve(
        self,
        request_id: str,
        approver: str,
    ) -> bool:
        """Approve a request."""
        with self._lock:
            request = self._requests.get(request_id)
            if request is None or request.status != ApprovalStatus.PENDING:
                return False

            if request.approvers and approver not in request.approvers:
                return False

            request.status = ApprovalStatus.APPROVED
            request.approved_by = approver
            request.resolved_at = datetime.now()
            self._requests[request_id] = request

            # Perform stage transition
            self._registry.transition_stage(
                request.model_id,
                request.version,
                request.target_stage,
            )

            return True

    def reject(
        self,
        request_id: str,
        approver: str,
        reason: str = "",
    ) -> bool:
        """Reject a request."""
        with self._lock:
            request = self._requests.get(request_id)
            if request is None or request.status != ApprovalStatus.PENDING:
                return False

            request.status = ApprovalStatus.REJECTED
            request.approved_by = approver
            request.resolved_at = datetime.now()
            if reason:
                request.reason = f"{request.reason} | Rejected: {reason}"
            self._requests[request_id] = request

            return True

    def get_pending_requests(
        self,
        approver: Optional[str] = None,
    ) -> List[ApprovalRequest]:
        """Get pending approval requests."""
        with self._lock:
            requests = [
                r for r in self._requests.values()
                if r.status == ApprovalStatus.PENDING
            ]
            if approver:
                requests = [r for r in requests if approver in r.approvers]
            return requests


class ModelValidator:
    """Validate models before deployment."""

    def __init__(self, registry: ModelRegistry):
        self._registry = registry
        self._validation_rules: List[Callable] = []

    def add_rule(self, rule: Callable[[ModelVersion], Tuple[bool, str]]) -> None:
        """Add a validation rule."""
        self._validation_rules.append(rule)

    def validate(self, model_id: str, version: str) -> Tuple[bool, List[str]]:
        """Validate a model version."""
        model_version = self._registry.get_version(model_id, version)
        if model_version is None:
            return False, ["Model version not found"]

        issues = []

        # Check basic requirements
        if not model_version.artifact_path:
            issues.append("No artifact path specified")

        if not model_version.metrics:
            issues.append("No metrics recorded")

        # Run custom validation rules
        for rule in self._validation_rules:
            try:
                passed, message = rule(model_version)
                if not passed:
                    issues.append(message)
            except Exception as e:
                issues.append(f"Validation rule error: {str(e)}")

        return len(issues) == 0, issues


# ========================
# Vision Provider
# ========================


class ModelRegistryVisionProvider(VisionProvider):
    """Vision provider for model registry capabilities."""

    def __init__(self, storage_path: Optional[str] = None):
        self._storage_path = storage_path
        self._registry: Optional[ModelRegistry] = None
        self._deployer: Optional[ModelDeployer] = None
        self._ab_manager: Optional[ABTestManager] = None

    def get_description(self) -> VisionDescription:
        """Get provider description."""
        return VisionDescription(
            name="Model Registry Vision Provider",
            version="1.0.0",
            description="Model versioning, deployment, and A/B testing",
            capabilities=[
                "model_versioning",
                "model_deployment",
                "ab_testing",
                "approval_workflow",
                "rollback",
            ],
        )

    def initialize(self) -> None:
        """Initialize the provider."""
        self._registry = ModelRegistry(self._storage_path)
        self._deployer = ModelDeployer(self._registry)
        self._ab_manager = ABTestManager(self._registry)

    def shutdown(self) -> None:
        """Shutdown the provider."""
        self._registry = None
        self._deployer = None
        self._ab_manager = None

    def get_registry(self) -> ModelRegistry:
        """Get the model registry."""
        if self._registry is None:
            self.initialize()
        return self._registry

    def get_deployer(self) -> ModelDeployer:
        """Get the model deployer."""
        if self._deployer is None:
            self.initialize()
        return self._deployer

    def get_ab_manager(self) -> ABTestManager:
        """Get the A/B test manager."""
        if self._ab_manager is None:
            self.initialize()
        return self._ab_manager


# ========================
# Factory Functions
# ========================


def create_model_registry(storage_path: Optional[str] = None) -> ModelRegistry:
    """Create a model registry."""
    return ModelRegistry(storage_path=storage_path)


def create_model_version(
    model_id: str,
    version: str,
    name: str,
    description: str = "",
    stage: ModelStage = ModelStage.DEVELOPMENT,
) -> ModelVersion:
    """Create a model version."""
    return ModelVersion(
        model_id=model_id,
        version=version,
        name=name,
        description=description,
        stage=stage,
    )


def create_model_metadata(
    model_id: str,
    name: str,
    description: str = "",
    owner: str = "system",
) -> ModelMetadata:
    """Create model metadata."""
    return ModelMetadata(
        model_id=model_id,
        name=name,
        description=description,
        owner=owner,
    )


def create_deployment_config(
    deployment_id: str,
    model_id: str,
    version: str,
    strategy: DeploymentStrategy = DeploymentStrategy.DIRECT,
    target_environment: str = "production",
) -> DeploymentConfig:
    """Create a deployment configuration."""
    return DeploymentConfig(
        deployment_id=deployment_id,
        model_id=model_id,
        version=version,
        strategy=strategy,
        target_environment=target_environment,
    )


def create_ab_test_config(
    test_id: str,
    name: str,
    control_model: str,
    control_version: str,
    treatment_model: str,
    treatment_version: str,
    traffic_split: float = 0.5,
) -> ABTestConfig:
    """Create an A/B test configuration."""
    return ABTestConfig(
        test_id=test_id,
        name=name,
        control_model=control_model,
        control_version=control_version,
        treatment_model=treatment_model,
        treatment_version=treatment_version,
        traffic_split=traffic_split,
    )


def create_model_deployer(registry: ModelRegistry) -> ModelDeployer:
    """Create a model deployer."""
    return ModelDeployer(registry=registry)


def create_ab_test_manager(registry: ModelRegistry) -> ABTestManager:
    """Create an A/B test manager."""
    return ABTestManager(registry=registry)


def create_approval_workflow(registry: ModelRegistry) -> ApprovalWorkflow:
    """Create an approval workflow."""
    return ApprovalWorkflow(registry=registry)


def create_model_registry_provider(
    storage_path: Optional[str] = None,
) -> ModelRegistryVisionProvider:
    """Create a model registry vision provider."""
    return ModelRegistryVisionProvider(storage_path=storage_path)
