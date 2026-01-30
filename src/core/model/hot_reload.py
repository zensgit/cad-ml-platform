"""Model Hot Reload Mechanism.

Features:
- Zero-downtime model updates
- Version management
- Automatic health checks after reload
- Rollback capability
- A/B testing support
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model status."""

    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    DRAINING = "draining"  # Accepting no new requests
    RETIRED = "retired"


@dataclass
class ModelVersion:
    """Represents a model version."""

    version: str
    path: str
    status: ModelStatus = ModelStatus.LOADING
    loaded_at: Optional[datetime] = None
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    # Performance metrics
    request_count: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "path": self.path,
            "status": self.status.value,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "checksum": self.checksum,
            "metadata": self.metadata,
            "error": self.error,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "avg_latency_ms": self.avg_latency_ms,
        }


@dataclass
class ReloadConfig:
    """Configuration for model reloading."""

    # Health check settings
    health_check_timeout_s: float = 30.0
    health_check_retries: int = 3

    # Traffic settings
    drain_timeout_s: float = 60.0
    warmup_requests: int = 10

    # Rollback settings
    auto_rollback: bool = True
    error_threshold: float = 0.1  # 10% error rate triggers rollback

    # A/B testing
    enable_ab_testing: bool = False
    ab_traffic_split: float = 0.1  # 10% to new model


class ModelLoader:
    """Base class for model loaders."""

    async def load(self, path: str) -> Any:
        """Load a model from path."""
        raise NotImplementedError

    async def unload(self, model: Any) -> None:
        """Unload a model."""
        pass

    async def health_check(self, model: Any) -> bool:
        """Check if model is healthy."""
        return True

    def compute_checksum(self, path: str) -> str:
        """Compute checksum for a model path."""
        if os.path.isfile(path):
            with open(path, "rb") as f:
                return hashlib.md5(f.read(), usedforsecurity=False).hexdigest()
        elif os.path.isdir(path):
            # For directories, hash file names and sizes
            hasher = hashlib.md5(usedforsecurity=False)
            for root, _, files in sorted(os.walk(path)):
                for name in sorted(files):
                    file_path = os.path.join(root, name)
                    hasher.update(name.encode())
                    hasher.update(str(os.path.getsize(file_path)).encode())
            return hasher.hexdigest()
        return ""


class DefaultModelLoader(ModelLoader):
    """Default model loader (mock implementation)."""

    async def load(self, path: str) -> Any:
        """Load a model (mock)."""
        await asyncio.sleep(0.1)  # Simulate loading
        return {"path": path, "loaded": True}

    async def health_check(self, model: Any) -> bool:
        """Check model health (mock)."""
        return model.get("loaded", False)


class HotReloadManager:
    """Manages hot reloading of models."""

    def __init__(
        self,
        config: Optional[ReloadConfig] = None,
        loader: Optional[ModelLoader] = None,
    ):
        self.config = config or ReloadConfig()
        self.loader = loader or DefaultModelLoader()

        self._versions: Dict[str, ModelVersion] = {}
        self._models: Dict[str, Any] = {}
        self._active_version: Optional[str] = None
        self._previous_version: Optional[str] = None

        self._lock: Optional[asyncio.Lock] = None
        self._reload_callbacks: List[Callable] = []

        # Metrics
        self._metrics = {
            "reload_count": 0,
            "rollback_count": 0,
            "failed_reloads": 0,
        }

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def load_model(
        self,
        version: str,
        path: str,
        metadata: Optional[Dict[str, Any]] = None,
        activate: bool = True,
    ) -> bool:
        """Load a new model version."""
        async with self._get_lock():
            # Create version record
            model_version = ModelVersion(
                version=version,
                path=path,
                status=ModelStatus.LOADING,
                metadata=metadata or {},
            )

            try:
                # Compute checksum
                model_version.checksum = self.loader.compute_checksum(path)

                # Load model
                logger.info(f"Loading model version {version} from {path}")
                model = await self.loader.load(path)

                # Health check
                is_healthy = await self._run_health_check(model)
                if not is_healthy:
                    raise ValueError("Model failed health check")

                # Store
                self._models[version] = model
                model_version.status = ModelStatus.READY
                model_version.loaded_at = datetime.now()
                self._versions[version] = model_version

                # Activate if requested
                if activate:
                    await self._activate_version(version)

                logger.info(f"Model version {version} loaded successfully")
                self._metrics["reload_count"] += 1
                return True

            except Exception as e:
                logger.error(f"Failed to load model version {version}: {e}")
                model_version.status = ModelStatus.FAILED
                model_version.error = str(e)
                self._versions[version] = model_version
                self._metrics["failed_reloads"] += 1
                return False

    async def _run_health_check(self, model: Any) -> bool:
        """Run health check with retries."""
        for attempt in range(self.config.health_check_retries):
            try:
                is_healthy = await asyncio.wait_for(
                    self.loader.health_check(model),
                    timeout=self.config.health_check_timeout_s,
                )
                if is_healthy:
                    return True
            except asyncio.TimeoutError:
                logger.warning(f"Health check timed out (attempt {attempt + 1})")
            except Exception as e:
                logger.warning(f"Health check failed (attempt {attempt + 1}): {e}")

            await asyncio.sleep(1.0)

        return False

    async def _activate_version(self, version: str) -> None:
        """Activate a model version."""
        if version not in self._versions:
            raise ValueError(f"Unknown version: {version}")

        version_info = self._versions[version]
        if version_info.status not in (ModelStatus.READY, ModelStatus.DRAINING):
            raise ValueError(f"Version {version} is not ready")

        # Store previous version for rollback
        if self._active_version:
            self._previous_version = self._active_version
            self._versions[self._previous_version].status = ModelStatus.DRAINING

        self._active_version = version
        logger.info(f"Activated model version {version}")

        # Notify callbacks
        for callback in self._reload_callbacks:
            try:
                await callback(version)
            except Exception as e:
                logger.warning(f"Reload callback error: {e}")

    async def rollback(self) -> bool:
        """Rollback to previous version."""
        async with self._get_lock():
            if not self._previous_version:
                logger.warning("No previous version to rollback to")
                return False

            if self._previous_version not in self._versions:
                logger.warning(f"Previous version {self._previous_version} not available")
                return False

            prev_version = self._versions[self._previous_version]
            if prev_version.status not in (ModelStatus.READY, ModelStatus.DRAINING):
                logger.warning(f"Previous version {self._previous_version} not in valid state")
                return False

            # Rollback
            current = self._active_version
            await self._activate_version(self._previous_version)

            # Mark current as failed
            if current and current in self._versions:
                self._versions[current].status = ModelStatus.RETIRED

            self._metrics["rollback_count"] += 1
            logger.info(f"Rolled back from {current} to {self._previous_version}")
            return True

    async def unload_version(self, version: str) -> bool:
        """Unload a model version."""
        async with self._get_lock():
            if version not in self._versions:
                return False

            if version == self._active_version:
                logger.warning("Cannot unload active version")
                return False

            if version in self._models:
                await self.loader.unload(self._models[version])
                del self._models[version]

            self._versions[version].status = ModelStatus.RETIRED
            logger.info(f"Unloaded model version {version}")
            return True

    def get_active_model(self) -> Optional[Any]:
        """Get the active model."""
        if self._active_version and self._active_version in self._models:
            return self._models[self._active_version]
        return None

    def get_active_version(self) -> Optional[str]:
        """Get the active version string."""
        return self._active_version

    def get_version_info(self, version: str) -> Optional[Dict[str, Any]]:
        """Get version information."""
        if version in self._versions:
            return self._versions[version].to_dict()
        return None

    def list_versions(self) -> List[Dict[str, Any]]:
        """List all versions."""
        return [v.to_dict() for v in self._versions.values()]

    def record_request(self, latency_ms: float, error: bool = False) -> None:
        """Record a request for metrics."""
        if self._active_version and self._active_version in self._versions:
            v = self._versions[self._active_version]
            v.request_count += 1
            if error:
                v.error_count += 1

            # Update rolling average latency
            if v.request_count == 1:
                v.avg_latency_ms = latency_ms
            else:
                v.avg_latency_ms = (v.avg_latency_ms * (v.request_count - 1) + latency_ms) / v.request_count

            # Check for auto-rollback
            if self.config.auto_rollback and v.request_count >= 100:
                error_rate = v.error_count / v.request_count
                if error_rate > self.config.error_threshold:
                    logger.warning(f"Error rate {error_rate:.2%} exceeded threshold, triggering rollback")
                    asyncio.create_task(self.rollback())

    def add_reload_callback(self, callback: Callable) -> None:
        """Add a callback to be called after reload."""
        self._reload_callbacks.append(callback)

    def get_metrics(self) -> Dict[str, Any]:
        """Get reload metrics."""
        active_info = None
        if self._active_version:
            active_info = self.get_version_info(self._active_version)

        return {
            **self._metrics,
            "active_version": self._active_version,
            "previous_version": self._previous_version,
            "total_versions": len(self._versions),
            "active_info": active_info,
        }


# Global manager
_hot_reload_manager: Optional[HotReloadManager] = None


def get_hot_reload_manager() -> HotReloadManager:
    """Get the global hot reload manager."""
    global _hot_reload_manager
    if _hot_reload_manager is None:
        _hot_reload_manager = HotReloadManager()
    return _hot_reload_manager
