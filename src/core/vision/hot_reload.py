"""Configuration hot-reloading module for Vision Provider system.

This module provides hot-reloading capabilities including:
- Configuration file watching
- Dynamic provider reconfiguration
- Graceful reload with zero downtime
- Configuration validation
- Reload callbacks and events
"""

import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .base import VisionDescription, VisionProvider


class ReloadTrigger(Enum):
    """Trigger for configuration reload."""

    FILE_CHANGE = "file_change"
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    API_CALL = "api_call"


class ReloadStatus(Enum):
    """Status of a reload operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class ReloadEvent:
    """Event representing a configuration reload."""

    event_id: str
    trigger: ReloadTrigger
    status: ReloadStatus
    timestamp: datetime = field(default_factory=datetime.now)
    old_config: Optional[Dict[str, Any]] = None
    new_config: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class ConfigVersion:
    """Version information for configuration."""

    version: int
    config: Dict[str, Any]
    loaded_at: datetime = field(default_factory=datetime.now)
    source: str = "unknown"


class ConfigSource(ABC):
    """Abstract base class for configuration sources."""

    @abstractmethod
    def load(self) -> Dict[str, Any]:
        """Load configuration from source."""
        pass

    @abstractmethod
    def has_changed(self) -> bool:
        """Check if configuration has changed."""
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return source identifier."""
        pass


class FileConfigSource(ConfigSource):
    """Configuration source from file."""

    def __init__(self, path: Union[str, Path]) -> None:
        """Initialize file config source."""
        self._path = Path(path)
        self._last_modified: Optional[float] = None
        self._last_hash: Optional[str] = None

    @property
    def source_name(self) -> str:
        """Return source name."""
        return f"file:{self._path}"

    def load(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not self._path.exists():
            return {}

        self._last_modified = self._path.stat().st_mtime

        content = self._path.read_text()
        self._last_hash = str(hash(content))

        if self._path.suffix == ".json":
            return json.loads(content)
        elif self._path.suffix in (".yaml", ".yml"):
            # Simple YAML-like parsing for basic cases
            # In production, use PyYAML
            return json.loads(content)
        else:
            return {"raw": content}

    def has_changed(self) -> bool:
        """Check if file has changed."""
        if not self._path.exists():
            return self._last_modified is not None

        current_mtime = self._path.stat().st_mtime
        return current_mtime != self._last_modified


class DictConfigSource(ConfigSource):
    """Configuration source from dictionary."""

    def __init__(self, config: Dict[str, Any], name: str = "dict") -> None:
        """Initialize dict config source."""
        self._config = config
        self._name = name
        self._changed = False

    @property
    def source_name(self) -> str:
        """Return source name."""
        return f"dict:{self._name}"

    def load(self) -> Dict[str, Any]:
        """Load configuration."""
        self._changed = False
        return dict(self._config)

    def has_changed(self) -> bool:
        """Check if config has changed."""
        return self._changed

    def update(self, config: Dict[str, Any]) -> None:
        """Update configuration."""
        self._config = config
        self._changed = True


@dataclass
class HotReloadConfig:
    """Configuration for hot reload behavior."""

    enabled: bool = True
    poll_interval_seconds: float = 5.0
    max_retries: int = 3
    rollback_on_error: bool = True
    validate_before_apply: bool = True
    graceful_shutdown_timeout: float = 30.0


class ConfigValidator:
    """Validates configuration before applying."""

    def __init__(self) -> None:
        """Initialize validator."""
        self._rules: List[Callable[[Dict[str, Any]], Optional[str]]] = []

    def add_rule(
        self, rule: Callable[[Dict[str, Any]], Optional[str]]
    ) -> None:
        """Add validation rule.

        Rule should return None if valid, error message if invalid.
        """
        self._rules.append(rule)

    def validate(self, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate configuration.

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors: List[str] = []

        for rule in self._rules:
            try:
                error = rule(config)
                if error:
                    errors.append(error)
            except Exception as e:
                errors.append(f"Validation error: {e}")

        return len(errors) == 0, errors


class HotReloadManager:
    """Manages hot-reloading of configuration."""

    def __init__(
        self,
        source: ConfigSource,
        config: Optional[HotReloadConfig] = None,
    ) -> None:
        """Initialize hot reload manager.

        Args:
            source: Configuration source
            config: Hot reload configuration
        """
        self._source = source
        self._config = config or HotReloadConfig()
        self._validator = ConfigValidator()
        self._current_config: Optional[Dict[str, Any]] = None
        self._version = 0
        self._history: List[ConfigVersion] = []
        self._callbacks: List[Callable[[ReloadEvent], None]] = []
        self._events: List[ReloadEvent] = []
        self._watcher_thread: Optional[threading.Thread] = None
        self._stop_watcher = threading.Event()
        self._lock = threading.Lock()

    @property
    def current_config(self) -> Optional[Dict[str, Any]]:
        """Get current configuration."""
        with self._lock:
            return dict(self._current_config) if self._current_config else None

    @property
    def version(self) -> int:
        """Get current configuration version."""
        return self._version

    @property
    def history(self) -> List[ConfigVersion]:
        """Get configuration history."""
        return list(self._history)

    @property
    def events(self) -> List[ReloadEvent]:
        """Get reload events."""
        return list(self._events)

    def add_callback(
        self, callback: Callable[[ReloadEvent], None]
    ) -> None:
        """Add reload callback."""
        self._callbacks.append(callback)

    def remove_callback(
        self, callback: Callable[[ReloadEvent], None]
    ) -> None:
        """Remove reload callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def load_initial(self) -> Dict[str, Any]:
        """Load initial configuration."""
        config = self._source.load()

        with self._lock:
            self._current_config = config
            self._version = 1
            self._history.append(
                ConfigVersion(
                    version=self._version,
                    config=dict(config),
                    source=self._source.source_name,
                )
            )

        return config

    def reload(
        self, trigger: ReloadTrigger = ReloadTrigger.MANUAL
    ) -> ReloadEvent:
        """Reload configuration.

        Args:
            trigger: What triggered the reload

        Returns:
            ReloadEvent with result
        """
        import uuid

        event_id = str(uuid.uuid4())
        start_time = time.time()

        event = ReloadEvent(
            event_id=event_id,
            trigger=trigger,
            status=ReloadStatus.IN_PROGRESS,
            old_config=self.current_config,
        )

        try:
            # Load new config
            new_config = self._source.load()
            event.new_config = new_config

            # Validate if enabled
            if self._config.validate_before_apply:
                is_valid, errors = self._validator.validate(new_config)
                if not is_valid:
                    raise ValueError(f"Validation failed: {', '.join(errors)}")

            # Apply new config
            with self._lock:
                old_config = self._current_config
                self._current_config = new_config
                self._version += 1
                self._history.append(
                    ConfigVersion(
                        version=self._version,
                        config=dict(new_config),
                        source=self._source.source_name,
                    )
                )

            event.status = ReloadStatus.SUCCESS
            event.duration_ms = (time.time() - start_time) * 1000

        except Exception as e:
            event.status = ReloadStatus.FAILED
            event.error = str(e)
            event.duration_ms = (time.time() - start_time) * 1000

            # Rollback if configured
            if self._config.rollback_on_error and event.old_config:
                with self._lock:
                    self._current_config = event.old_config
                event.status = ReloadStatus.ROLLED_BACK

        self._events.append(event)

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception:
                pass  # Don't let callback errors affect reload

        return event

    def start_watching(self) -> None:
        """Start watching for configuration changes."""
        if not self._config.enabled:
            return

        if self._watcher_thread and self._watcher_thread.is_alive():
            return

        self._stop_watcher.clear()
        self._watcher_thread = threading.Thread(
            target=self._watch_loop,
            daemon=True,
        )
        self._watcher_thread.start()

    def stop_watching(self) -> None:
        """Stop watching for configuration changes."""
        self._stop_watcher.set()
        if self._watcher_thread:
            self._watcher_thread.join(timeout=5.0)

    def _watch_loop(self) -> None:
        """Background loop for watching configuration changes."""
        while not self._stop_watcher.is_set():
            try:
                if self._source.has_changed():
                    self.reload(ReloadTrigger.FILE_CHANGE)
            except Exception:
                pass  # Log in production

            self._stop_watcher.wait(self._config.poll_interval_seconds)

    def rollback(self, to_version: Optional[int] = None) -> bool:
        """Rollback to a previous configuration version.

        Args:
            to_version: Target version (default: previous)

        Returns:
            True if rollback successful
        """
        if not self._history:
            return False

        if to_version is None:
            # Rollback to previous version
            if len(self._history) < 2:
                return False
            target = self._history[-2]
        else:
            # Find specific version
            target = None
            for cv in self._history:
                if cv.version == to_version:
                    target = cv
                    break
            if not target:
                return False

        with self._lock:
            self._current_config = dict(target.config)
            self._version += 1
            self._history.append(
                ConfigVersion(
                    version=self._version,
                    config=dict(target.config),
                    source=f"rollback_from_v{target.version}",
                )
            )

        return True


class HotReloadingVisionProvider(VisionProvider):
    """Vision provider with hot-reloading configuration."""

    def __init__(
        self,
        provider_factory: Callable[[Dict[str, Any]], VisionProvider],
        reload_manager: HotReloadManager,
    ) -> None:
        """Initialize hot-reloading provider.

        Args:
            provider_factory: Factory to create provider from config
            reload_manager: Hot reload manager
        """
        self._provider_factory = provider_factory
        self._reload_manager = reload_manager
        self._provider: Optional[VisionProvider] = None
        self._lock = threading.Lock()

        # Initialize provider
        config = reload_manager.load_initial()
        self._provider = provider_factory(config)

        # Register for reload events
        reload_manager.add_callback(self._on_reload)

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        with self._lock:
            if self._provider:
                return f"hot_reload_{self._provider.provider_name}"
            return "hot_reload_uninitialized"

    @property
    def reload_manager(self) -> HotReloadManager:
        """Get reload manager."""
        return self._reload_manager

    def _on_reload(self, event: ReloadEvent) -> None:
        """Handle reload event."""
        if event.status == ReloadStatus.SUCCESS and event.new_config:
            try:
                new_provider = self._provider_factory(event.new_config)
                with self._lock:
                    self._provider = new_provider
            except Exception:
                pass  # Log in production

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image using current provider.

        Args:
            image_data: Raw image bytes
            include_description: Whether to include description

        Returns:
            Vision analysis description
        """
        with self._lock:
            if not self._provider:
                raise RuntimeError("Provider not initialized")
            provider = self._provider

        return await provider.analyze_image(image_data, include_description)


def create_hot_reloading_provider(
    provider_factory: Callable[[Dict[str, Any]], VisionProvider],
    config_source: ConfigSource,
    hot_reload_config: Optional[HotReloadConfig] = None,
    start_watching: bool = True,
) -> HotReloadingVisionProvider:
    """Create a hot-reloading vision provider.

    Args:
        provider_factory: Factory to create provider from config
        config_source: Configuration source
        hot_reload_config: Hot reload configuration
        start_watching: Whether to start watching immediately

    Returns:
        HotReloadingVisionProvider instance
    """
    manager = HotReloadManager(
        source=config_source,
        config=hot_reload_config,
    )

    provider = HotReloadingVisionProvider(
        provider_factory=provider_factory,
        reload_manager=manager,
    )

    if start_watching:
        manager.start_watching()

    return provider
