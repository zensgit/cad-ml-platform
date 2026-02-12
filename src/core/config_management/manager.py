"""Configuration Manager.

Provides configuration management:
- Multi-source configuration
- Hot reload
- Change notifications
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union

from src.core.config_management.core import (
    ConfigSchema,
    ConfigSource,
    ConfigValidator,
    ConfigValue,
    DictConfigSource,
    EnvironmentConfigSource,
    FileConfigSource,
    ValidationResult,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ConfigChange:
    """Record of a configuration change."""
    key: str
    old_value: Any
    new_value: Any
    source: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ConfigManager:
    """Centralized configuration management."""

    def __init__(
        self,
        validator: Optional[ConfigValidator] = None,
    ):
        self._sources: List[ConfigSource] = []
        self._validator = validator or ConfigValidator()
        self._cache: Dict[str, ConfigValue] = {}
        self._listeners: List[Callable[[ConfigChange], None]] = []
        self._lock = threading.RLock()

    def add_source(self, source: ConfigSource) -> "ConfigManager":
        """Add configuration source."""
        with self._lock:
            self._sources.append(source)
            # Sort by priority (highest first)
            self._sources.sort(key=lambda s: -s.priority)
            self._rebuild_cache()
        return self

    def add_schema(self, schema: ConfigSchema) -> "ConfigManager":
        """Add configuration schema."""
        self._validator.add_schema(schema)
        return self

    def add_listener(
        self,
        listener: Callable[[ConfigChange], None],
    ) -> "ConfigManager":
        """Add change listener."""
        self._listeners.append(listener)
        return self

    def get(
        self,
        key: str,
        default: Optional[T] = None,
        value_type: Optional[Type[T]] = None,
    ) -> Optional[T]:
        """Get configuration value."""
        with self._lock:
            if key in self._cache:
                value = self._cache[key].value
                if value_type:
                    return self._convert(value, value_type)
                return value

            # Search sources
            for source in self._sources:
                value = source.get(key)
                if value is not None:
                    self._cache[key] = ConfigValue(
                        key=key,
                        value=value,
                        source=source.name,
                    )
                    if value_type:
                        return self._convert(value, value_type)
                    return value

            return default

    def get_string(self, key: str, default: str = "") -> str:
        """Get string value."""
        return self.get(key, default, str) or default

    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer value."""
        return self.get(key, default, int) or default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float value."""
        return self.get(key, default, float) or default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean value."""
        value = self.get(key, default, bool)
        return value if value is not None else default

    def get_list(self, key: str, default: Optional[List] = None) -> List:
        """Get list value."""
        return self.get(key, default or [], list) or default or []

    def get_dict(self, key: str, default: Optional[Dict] = None) -> Dict:
        """Get dict value."""
        return self.get(key, default or {}, dict) or default or {}

    def set(self, key: str, value: Any, source: str = "runtime") -> None:
        """Set configuration value at runtime."""
        with self._lock:
            old_value = self._cache.get(key)

            self._cache[key] = ConfigValue(
                key=key,
                value=value,
                source=source,
            )

            # Notify listeners
            change = ConfigChange(
                key=key,
                old_value=old_value.value if old_value else None,
                new_value=value,
                source=source,
            )
            self._notify_listeners(change)

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        with self._lock:
            result = {}

            # Merge all sources (lowest priority first)
            for source in reversed(self._sources):
                result.update(source.get_all())

            return result

    def validate(self) -> ValidationResult:
        """Validate current configuration."""
        config = self.get_all()
        return self._validator.validate(config)

    def reload(self) -> List[ConfigChange]:
        """Reload configuration from all sources."""
        with self._lock:
            old_cache = self._cache.copy()
            changes = []

            # Reload file sources
            for source in self._sources:
                if isinstance(source, FileConfigSource):
                    source.reload()

            # Rebuild cache
            self._rebuild_cache()

            # Find changes
            all_keys = set(old_cache.keys()) | set(self._cache.keys())

            for key in all_keys:
                old = old_cache.get(key)
                new = self._cache.get(key)

                old_value = old.value if old else None
                new_value = new.value if new else None

                if old_value != new_value:
                    change = ConfigChange(
                        key=key,
                        old_value=old_value,
                        new_value=new_value,
                        source=new.source if new else "removed",
                    )
                    changes.append(change)
                    self._notify_listeners(change)

            logger.info(f"Configuration reloaded, {len(changes)} changes")
            return changes

    def _rebuild_cache(self) -> None:
        """Rebuild configuration cache from sources."""
        self._cache.clear()

        # Process sources in priority order (highest first)
        for source in self._sources:
            for key, value in self._flatten(source.get_all()).items():
                if key not in self._cache:  # Higher priority wins
                    self._cache[key] = ConfigValue(
                        key=key,
                        value=value,
                        source=source.name,
                    )

    def _flatten(
        self,
        data: Dict[str, Any],
        prefix: str = "",
    ) -> Dict[str, Any]:
        """Flatten nested dict to dotted keys."""
        result = {}

        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                result.update(self._flatten(value, full_key))
            else:
                result[full_key] = value

        return result

    def _convert(self, value: Any, target_type: Type[T]) -> T:
        """Convert value to target type."""
        if isinstance(value, target_type):
            return value

        if target_type == str:
            return str(value)
        elif target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)
        elif target_type == bool:
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes", "on")
            return bool(value)
        elif target_type == list:
            if isinstance(value, str):
                return value.split(",")
            return list(value)

        return value

    def _notify_listeners(self, change: ConfigChange) -> None:
        """Notify listeners of change."""
        for listener in self._listeners:
            try:
                listener(change)
            except Exception as e:
                logger.error(f"Config listener error: {e}")


class ConfigWatcher:
    """Watches for configuration file changes."""

    def __init__(
        self,
        manager: ConfigManager,
        paths: List[Union[str, Path]],
        interval_seconds: float = 5.0,
    ):
        self._manager = manager
        self._paths = [Path(p) for p in paths]
        self._interval = interval_seconds
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._mtimes: Dict[Path, float] = {}

    def start(self) -> None:
        """Start watching."""
        if self._running:
            return

        self._running = True
        self._update_mtimes()
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info(f"Config watcher started for {len(self._paths)} paths")

    def stop(self) -> None:
        """Stop watching."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Config watcher stopped")

    def _watch_loop(self) -> None:
        """Watch loop."""
        while self._running:
            try:
                if self._check_changes():
                    self._manager.reload()
            except Exception as e:
                logger.error(f"Config watcher error: {e}")

            time.sleep(self._interval)

    def _check_changes(self) -> bool:
        """Check if any files changed."""
        changed = False

        for path in self._paths:
            if path.exists():
                mtime = path.stat().st_mtime
                if path in self._mtimes:
                    if mtime > self._mtimes[path]:
                        logger.debug(f"Config file changed: {path}")
                        changed = True
                self._mtimes[path] = mtime

        return changed

    def _update_mtimes(self) -> None:
        """Update modification times."""
        for path in self._paths:
            if path.exists():
                self._mtimes[path] = path.stat().st_mtime


class AsyncConfigWatcher:
    """Async config watcher."""

    def __init__(
        self,
        manager: ConfigManager,
        paths: List[Union[str, Path]],
        interval_seconds: float = 5.0,
    ):
        self._manager = manager
        self._paths = [Path(p) for p in paths]
        self._interval = interval_seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._mtimes: Dict[Path, float] = {}

    async def start(self) -> None:
        """Start watching."""
        if self._running:
            return

        self._running = True
        self._update_mtimes()
        self._task = asyncio.create_task(self._watch_loop())

    async def stop(self) -> None:
        """Stop watching."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _watch_loop(self) -> None:
        """Watch loop."""
        while self._running:
            try:
                if self._check_changes():
                    self._manager.reload()
            except Exception as e:
                logger.error(f"Config watcher error: {e}")

            await asyncio.sleep(self._interval)

    def _check_changes(self) -> bool:
        """Check if any files changed."""
        changed = False

        for path in self._paths:
            if path.exists():
                mtime = path.stat().st_mtime
                if path in self._mtimes and mtime > self._mtimes[path]:
                    changed = True
                self._mtimes[path] = mtime

        return changed

    def _update_mtimes(self) -> None:
        """Update modification times."""
        for path in self._paths:
            if path.exists():
                self._mtimes[path] = path.stat().st_mtime


def create_config_manager(
    config_files: Optional[List[Union[str, Path]]] = None,
    env_prefix: str = "",
    defaults: Optional[Dict[str, Any]] = None,
) -> ConfigManager:
    """Create a configured ConfigManager."""
    manager = ConfigManager()

    # Add default source (lowest priority)
    if defaults:
        manager.add_source(DictConfigSource(defaults, "defaults", 0))

    # Add file sources
    if config_files:
        for i, path in enumerate(config_files):
            manager.add_source(FileConfigSource(path, 50 + i))

    # Add environment source (highest priority)
    if env_prefix:
        manager.add_source(EnvironmentConfigSource(env_prefix, 100))

    return manager
