"""Centralized configuration management for Vision Provider system.

This module provides configuration management including:
- Hierarchical configuration sources
- Environment-based configuration
- Configuration validation
- Dynamic configuration updates
- Configuration versioning
"""

import json
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Set, Type, TypeVar, Union

from .base import VisionDescription, VisionProvider


class ConfigSource(Enum):
    """Source of configuration."""

    DEFAULT = "default"
    FILE = "file"
    ENVIRONMENT = "environment"
    REMOTE = "remote"
    OVERRIDE = "override"


class ConfigFormat(Enum):
    """Configuration file format."""

    JSON = "json"
    YAML = "yaml"
    INI = "ini"
    ENV = "env"


class ValidationLevel(Enum):
    """Configuration validation level."""

    NONE = "none"
    WARN = "warn"
    STRICT = "strict"


@dataclass
class ConfigValue:
    """Configuration value with metadata."""

    key: str
    value: Any
    source: ConfigSource = ConfigSource.DEFAULT
    timestamp: datetime = field(default_factory=datetime.now)
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "value": self.value,
            "source": self.source.value,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "metadata": dict(self.metadata),
        }


@dataclass
class ConfigSchema:
    """Schema for configuration validation."""

    key: str
    value_type: Type[Any]
    required: bool = False
    default: Any = None
    description: str = ""
    validators: List[Callable[[Any], bool]] = field(default_factory=list)
    choices: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None

    def validate(self, value: Any) -> List[str]:
        """Validate value against schema.

        Args:
            value: Value to validate

        Returns:
            List of validation errors
        """
        errors = []

        if value is None:
            if self.required:
                errors.append(f"Required config '{self.key}' is missing")
            return errors

        # Type check
        if not isinstance(value, self.value_type):
            errors.append(
                f"Config '{self.key}' expected {self.value_type.__name__}, "
                f"got {type(value).__name__}"
            )
            return errors

        # Choices check
        if self.choices and value not in self.choices:
            errors.append(f"Config '{self.key}' value '{value}' not in choices: {self.choices}")

        # Range check
        if self.min_value is not None and value < self.min_value:
            errors.append(f"Config '{self.key}' value {value} below minimum {self.min_value}")

        if self.max_value is not None and value > self.max_value:
            errors.append(f"Config '{self.key}' value {value} above maximum {self.max_value}")

        # Custom validators
        for validator in self.validators:
            try:
                if not validator(value):
                    errors.append(f"Config '{self.key}' failed custom validation")
            except Exception as e:
                errors.append(f"Config '{self.key}' validation error: {str(e)}")

        return errors


class ConfigProvider(ABC):
    """Abstract base class for configuration providers."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get configuration value.

        Args:
            key: Configuration key

        Returns:
            Configuration value or None
        """
        pass

    @abstractmethod
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values.

        Returns:
            Dictionary of all values
        """
        pass

    @property
    @abstractmethod
    def source(self) -> ConfigSource:
        """Return configuration source."""
        pass


class DictConfigProvider(ConfigProvider):
    """Configuration provider from dictionary."""

    def __init__(
        self,
        config: Dict[str, Any],
        config_source: ConfigSource = ConfigSource.DEFAULT,
    ) -> None:
        """Initialize provider.

        Args:
            config: Configuration dictionary
            config_source: Source type
        """
        self._config = config
        self._source = config_source

    def get(self, key: str) -> Optional[Any]:
        """Get configuration value."""
        parts = key.split(".")
        value = self._config

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None

        return value

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        return dict(self._config)

    @property
    def source(self) -> ConfigSource:
        """Return configuration source."""
        return self._source


class EnvironmentConfigProvider(ConfigProvider):
    """Configuration provider from environment variables."""

    def __init__(
        self,
        prefix: str = "",
        separator: str = "_",
    ) -> None:
        """Initialize provider.

        Args:
            prefix: Environment variable prefix
            separator: Separator for nested keys
        """
        self._prefix = prefix
        self._separator = separator

    def get(self, key: str) -> Optional[Any]:
        """Get configuration from environment."""
        env_key = key.replace(".", self._separator).upper()
        if self._prefix:
            env_key = f"{self._prefix}{self._separator}{env_key}"

        value = os.environ.get(env_key)

        if value is not None:
            # Try to parse as JSON
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value

        return None

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration from environment."""
        result = {}
        prefix = f"{self._prefix}{self._separator}" if self._prefix else ""

        for key, value in os.environ.items():
            if prefix and not key.startswith(prefix):
                continue

            config_key = key
            if prefix:
                config_key = key[len(prefix) :]
            config_key = config_key.lower().replace(self._separator, ".")

            try:
                result[config_key] = json.loads(value)
            except json.JSONDecodeError:
                result[config_key] = value

        return result

    @property
    def source(self) -> ConfigSource:
        """Return configuration source."""
        return ConfigSource.ENVIRONMENT


class FileConfigProvider(ConfigProvider):
    """Configuration provider from file."""

    def __init__(
        self,
        file_path: str,
        format_type: ConfigFormat = ConfigFormat.JSON,
    ) -> None:
        """Initialize provider.

        Args:
            file_path: Path to configuration file
            format_type: File format
        """
        self._file_path = file_path
        self._format = format_type
        self._config: Dict[str, Any] = {}
        self._loaded = False

    def load(self) -> None:
        """Load configuration from file."""
        if not os.path.exists(self._file_path):
            return

        with open(self._file_path, "r") as f:
            content = f.read()

        if self._format == ConfigFormat.JSON:
            self._config = json.loads(content)
        elif self._format == ConfigFormat.ENV:
            self._config = self._parse_env(content)
        else:
            self._config = json.loads(content)

        self._loaded = True

    def _parse_env(self, content: str) -> Dict[str, Any]:
        """Parse .env file format."""
        result = {}
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if "=" in line:
                key, value = line.split("=", 1)
                result[key.strip()] = value.strip().strip("\"'")

        return result

    def get(self, key: str) -> Optional[Any]:
        """Get configuration value."""
        if not self._loaded:
            self.load()

        parts = key.split(".")
        value = self._config

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None

        return value

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        if not self._loaded:
            self.load()
        return dict(self._config)

    @property
    def source(self) -> ConfigSource:
        """Return configuration source."""
        return ConfigSource.FILE


@dataclass
class ConfigChange:
    """Record of configuration change."""

    key: str
    old_value: Any
    new_value: Any
    source: ConfigSource
    timestamp: datetime = field(default_factory=datetime.now)
    change_id: str = field(default_factory=lambda: str(int(time.time() * 1000)))


class ConfigWatcher:
    """Watcher for configuration changes."""

    def __init__(self) -> None:
        """Initialize watcher."""
        self._callbacks: Dict[str, List[Callable[[ConfigChange], None]]] = {}
        self._global_callbacks: List[Callable[[ConfigChange], None]] = []
        self._lock = threading.Lock()

    def watch(
        self,
        key: str,
        callback: Callable[[ConfigChange], None],
    ) -> None:
        """Watch for changes to a key.

        Args:
            key: Configuration key
            callback: Callback function
        """
        with self._lock:
            if key not in self._callbacks:
                self._callbacks[key] = []
            self._callbacks[key].append(callback)

    def watch_all(
        self,
        callback: Callable[[ConfigChange], None],
    ) -> None:
        """Watch for all configuration changes.

        Args:
            callback: Callback function
        """
        with self._lock:
            self._global_callbacks.append(callback)

    def unwatch(
        self,
        key: str,
        callback: Callable[[ConfigChange], None],
    ) -> bool:
        """Remove watcher.

        Args:
            key: Configuration key
            callback: Callback to remove

        Returns:
            True if removed
        """
        with self._lock:
            if key in self._callbacks:
                try:
                    self._callbacks[key].remove(callback)
                    return True
                except ValueError:
                    pass
        return False

    def notify(self, change: ConfigChange) -> None:
        """Notify watchers of change.

        Args:
            change: Configuration change
        """
        callbacks = []

        with self._lock:
            callbacks.extend(self._global_callbacks)
            if change.key in self._callbacks:
                callbacks.extend(self._callbacks[change.key])

        for callback in callbacks:
            try:
                callback(change)
            except Exception:
                pass


class ConfigurationManager:
    """Centralized configuration manager."""

    def __init__(
        self,
        validation_level: ValidationLevel = ValidationLevel.WARN,
    ) -> None:
        """Initialize configuration manager.

        Args:
            validation_level: Validation level
        """
        self._providers: List[ConfigProvider] = []
        self._schemas: Dict[str, ConfigSchema] = {}
        self._overrides: Dict[str, Any] = {}
        self._cache: Dict[str, ConfigValue] = {}
        self._watcher = ConfigWatcher()
        self._validation_level = validation_level
        self._lock = threading.Lock()
        self._history: List[ConfigChange] = []

    @property
    def watcher(self) -> ConfigWatcher:
        """Return configuration watcher."""
        return self._watcher

    def add_provider(self, provider: ConfigProvider) -> None:
        """Add configuration provider.

        Args:
            provider: Configuration provider
        """
        with self._lock:
            self._providers.append(provider)
            self._cache.clear()

    def add_schema(self, schema: ConfigSchema) -> None:
        """Add configuration schema.

        Args:
            schema: Configuration schema
        """
        with self._lock:
            self._schemas[schema.key] = schema

    def get(
        self,
        key: str,
        default: Any = None,
        use_cache: bool = True,
    ) -> Any:
        """Get configuration value.

        Args:
            key: Configuration key
            default: Default value
            use_cache: Whether to use cache

        Returns:
            Configuration value
        """
        with self._lock:
            # Check cache
            if use_cache and key in self._cache:
                return self._cache[key].value

            # Check overrides first
            if key in self._overrides:
                value = self._overrides[key]
                source = ConfigSource.OVERRIDE
            else:
                # Check providers in reverse order (later providers override earlier)
                value = None
                source = ConfigSource.DEFAULT

                for provider in reversed(self._providers):
                    pvalue = provider.get(key)
                    if pvalue is not None:
                        value = pvalue
                        source = provider.source
                        break

            # Use default if no value found
            if value is None:
                schema = self._schemas.get(key)
                if schema and schema.default is not None:
                    value = schema.default
                else:
                    value = default

            # Validate
            self._validate_value(key, value)

            # Cache
            if value is not None:
                self._cache[key] = ConfigValue(
                    key=key,
                    value=value,
                    source=source,
                )

            return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration override.

        Args:
            key: Configuration key
            value: Configuration value
        """
        with self._lock:
            old_value = self._overrides.get(key)
            self._overrides[key] = value

            # Clear cache
            if key in self._cache:
                del self._cache[key]

            # Record change
            change = ConfigChange(
                key=key,
                old_value=old_value,
                new_value=value,
                source=ConfigSource.OVERRIDE,
            )
            self._history.append(change)

        # Notify watchers
        self._watcher.notify(change)

    def delete(self, key: str) -> bool:
        """Delete configuration override.

        Args:
            key: Configuration key

        Returns:
            True if deleted
        """
        with self._lock:
            if key in self._overrides:
                old_value = self._overrides[key]
                del self._overrides[key]

                if key in self._cache:
                    del self._cache[key]

                change = ConfigChange(
                    key=key,
                    old_value=old_value,
                    new_value=None,
                    source=ConfigSource.OVERRIDE,
                )
                self._history.append(change)
                self._watcher.notify(change)
                return True

        return False

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values.

        Returns:
            Dictionary of all values
        """
        result = {}

        with self._lock:
            # Merge all providers
            for provider in self._providers:
                result.update(provider.get_all())

            # Apply overrides
            result.update(self._overrides)

        return result

    def _validate_value(self, key: str, value: Any) -> None:
        """Validate configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        if self._validation_level == ValidationLevel.NONE:
            return

        schema = self._schemas.get(key)
        if not schema:
            return

        errors = schema.validate(value)
        if errors:
            if self._validation_level == ValidationLevel.STRICT:
                raise ValueError(f"Config validation failed: {errors}")

    def validate_all(self) -> Dict[str, List[str]]:
        """Validate all configuration.

        Returns:
            Dictionary of validation errors by key
        """
        errors = {}

        with self._lock:
            for key, schema in self._schemas.items():
                value = self.get(key, use_cache=False)
                key_errors = schema.validate(value)
                if key_errors:
                    errors[key] = key_errors

        return errors

    def get_history(
        self,
        key: Optional[str] = None,
        limit: int = 100,
    ) -> List[ConfigChange]:
        """Get configuration change history.

        Args:
            key: Optional filter by key
            limit: Maximum results

        Returns:
            List of changes
        """
        with self._lock:
            if key:
                changes = [c for c in self._history if c.key == key]
            else:
                changes = list(self._history)

            return changes[-limit:]

    def clear_cache(self) -> None:
        """Clear configuration cache."""
        with self._lock:
            self._cache.clear()

    def refresh(self) -> None:
        """Refresh configuration from providers."""
        self.clear_cache()

        # Reload file providers
        for provider in self._providers:
            if isinstance(provider, FileConfigProvider):
                provider.load()


@dataclass
class ConfigProfile:
    """Configuration profile."""

    name: str
    config: Dict[str, Any]
    description: str = ""
    parent: Optional[str] = None
    active: bool = False


class ProfileManager:
    """Manager for configuration profiles."""

    def __init__(self, config_manager: ConfigurationManager) -> None:
        """Initialize profile manager.

        Args:
            config_manager: Configuration manager
        """
        self._config_manager = config_manager
        self._profiles: Dict[str, ConfigProfile] = {}
        self._active_profile: Optional[str] = None
        self._lock = threading.Lock()

    def register_profile(self, profile: ConfigProfile) -> None:
        """Register a configuration profile.

        Args:
            profile: Configuration profile
        """
        with self._lock:
            self._profiles[profile.name] = profile

    def get_profile(self, name: str) -> Optional[ConfigProfile]:
        """Get profile by name.

        Args:
            name: Profile name

        Returns:
            Profile or None
        """
        with self._lock:
            return self._profiles.get(name)

    def activate_profile(self, name: str) -> bool:
        """Activate a profile.

        Args:
            name: Profile name

        Returns:
            True if activated
        """
        with self._lock:
            profile = self._profiles.get(name)
            if not profile:
                return False

            # Deactivate current profile
            if self._active_profile:
                current = self._profiles.get(self._active_profile)
                if current:
                    current.active = False

            # Apply profile config
            resolved = self._resolve_profile(profile)
            for key, value in resolved.items():
                self._config_manager.set(key, value)

            profile.active = True
            self._active_profile = name
            return True

    def _resolve_profile(self, profile: ConfigProfile) -> Dict[str, Any]:
        """Resolve profile including inheritance.

        Args:
            profile: Profile to resolve

        Returns:
            Resolved configuration
        """
        result = {}

        # Resolve parent first
        if profile.parent:
            parent = self._profiles.get(profile.parent)
            if parent:
                result.update(self._resolve_profile(parent))

        # Apply profile config
        result.update(profile.config)
        return result

    def list_profiles(self) -> List[str]:
        """List all profile names.

        Returns:
            List of profile names
        """
        with self._lock:
            return list(self._profiles.keys())

    @property
    def active_profile(self) -> Optional[str]:
        """Return active profile name."""
        return self._active_profile


@dataclass
class ConfigSnapshot:
    """Snapshot of configuration state."""

    snapshot_id: str
    timestamp: datetime
    config: Dict[str, Any]
    profile: Optional[str] = None


class ConfigSnapshotManager:
    """Manager for configuration snapshots."""

    def __init__(self, config_manager: ConfigurationManager) -> None:
        """Initialize snapshot manager.

        Args:
            config_manager: Configuration manager
        """
        self._config_manager = config_manager
        self._snapshots: Dict[str, ConfigSnapshot] = {}
        self._lock = threading.Lock()

    def create_snapshot(self, name: Optional[str] = None) -> ConfigSnapshot:
        """Create configuration snapshot.

        Args:
            name: Optional snapshot name

        Returns:
            Created snapshot
        """
        snapshot_id = name or f"snapshot_{int(time.time() * 1000)}"

        snapshot = ConfigSnapshot(
            snapshot_id=snapshot_id,
            timestamp=datetime.now(),
            config=self._config_manager.get_all(),
        )

        with self._lock:
            self._snapshots[snapshot_id] = snapshot

        return snapshot

    def restore_snapshot(self, snapshot_id: str) -> bool:
        """Restore configuration from snapshot.

        Args:
            snapshot_id: Snapshot ID

        Returns:
            True if restored
        """
        with self._lock:
            snapshot = self._snapshots.get(snapshot_id)
            if not snapshot:
                return False

            for key, value in snapshot.config.items():
                self._config_manager.set(key, value)

            return True

    def get_snapshot(self, snapshot_id: str) -> Optional[ConfigSnapshot]:
        """Get snapshot by ID.

        Args:
            snapshot_id: Snapshot ID

        Returns:
            Snapshot or None
        """
        with self._lock:
            return self._snapshots.get(snapshot_id)

    def list_snapshots(self) -> List[str]:
        """List all snapshot IDs.

        Returns:
            List of snapshot IDs
        """
        with self._lock:
            return list(self._snapshots.keys())

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a snapshot.

        Args:
            snapshot_id: Snapshot ID

        Returns:
            True if deleted
        """
        with self._lock:
            if snapshot_id in self._snapshots:
                del self._snapshots[snapshot_id]
                return True
        return False


class ConfigurableVisionProvider(VisionProvider):
    """Vision provider with centralized configuration."""

    def __init__(
        self,
        provider: VisionProvider,
        config_manager: ConfigurationManager,
        config_prefix: str = "vision",
    ) -> None:
        """Initialize configurable provider.

        Args:
            provider: Underlying vision provider
            config_manager: Configuration manager
            config_prefix: Configuration key prefix
        """
        self._provider = provider
        self._config_manager = config_manager
        self._config_prefix = config_prefix

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return f"configurable_{self._provider.provider_name}"

    @property
    def config_manager(self) -> ConfigurationManager:
        """Return configuration manager."""
        return self._config_manager

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value with prefix.

        Args:
            key: Configuration key
            default: Default value

        Returns:
            Configuration value
        """
        full_key = f"{self._config_prefix}.{key}"
        return self._config_manager.get(full_key, default)

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image with configuration.

        Args:
            image_data: Raw image bytes
            include_description: Whether to include description

        Returns:
            Vision analysis description
        """
        # Check if provider is enabled
        enabled = self.get_config("enabled", True)
        if not enabled:
            raise RuntimeError("Vision provider is disabled by configuration")

        # Get configuration
        timeout = self.get_config("timeout_seconds", 30.0)
        max_image_size = self.get_config("max_image_size", 10 * 1024 * 1024)

        # Validate image size
        if len(image_data) > max_image_size:
            raise ValueError(f"Image size {len(image_data)} exceeds maximum {max_image_size}")

        return await self._provider.analyze_image(image_data, include_description)


def create_configurable_provider(
    provider: VisionProvider,
    config: Optional[Dict[str, Any]] = None,
    config_prefix: str = "vision",
) -> ConfigurableVisionProvider:
    """Create a configurable vision provider.

    Args:
        provider: Underlying vision provider
        config: Optional initial configuration
        config_prefix: Configuration key prefix

    Returns:
        ConfigurableVisionProvider instance
    """
    config_manager = ConfigurationManager()

    if config:
        dict_provider = DictConfigProvider(config)
        config_manager.add_provider(dict_provider)

    return ConfigurableVisionProvider(
        provider=provider,
        config_manager=config_manager,
        config_prefix=config_prefix,
    )
