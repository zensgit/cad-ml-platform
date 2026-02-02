"""Configuration Manager Implementation.

Provides centralized configuration management with multiple sources.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ConfigPriority(int, Enum):
    """Configuration source priority (higher = more important)."""

    DEFAULT = 0
    FILE = 10
    ENVIRONMENT = 20
    CONSUL = 30
    ETCD = 30
    OVERRIDE = 100


@dataclass
class ConfigValue(Generic[T]):
    """A configuration value with metadata."""

    key: str
    value: T
    source: str
    priority: int
    version: int = 1
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.key)


@dataclass
class ConfigChangeEvent:
    """Event emitted when configuration changes."""

    key: str
    old_value: Any
    new_value: Any
    source: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ConfigSource(ABC):
    """Abstract base class for configuration sources."""

    def __init__(self, name: str, priority: int = ConfigPriority.DEFAULT):
        self.name = name
        self.priority = priority

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get a configuration value."""
        pass

    @abstractmethod
    async def get_all(self, prefix: str = "") -> Dict[str, Any]:
        """Get all configuration values with optional prefix filter."""
        pass

    async def set(self, key: str, value: Any) -> bool:
        """Set a configuration value (if supported)."""
        return False

    async def delete(self, key: str) -> bool:
        """Delete a configuration value (if supported)."""
        return False

    async def watch(self, key: str, callback: Callable[[ConfigChangeEvent], None]) -> None:
        """Watch for changes to a key (if supported)."""
        pass

    async def close(self) -> None:
        """Close the configuration source."""
        pass


class ConfigManager:
    """Central configuration manager supporting multiple sources."""

    def __init__(self, app_name: str = "cad-ml-platform"):
        self.app_name = app_name
        self._sources: List[ConfigSource] = []
        self._cache: Dict[str, ConfigValue] = {}
        self._watchers: Dict[str, List[Callable[[ConfigChangeEvent], None]]] = {}
        self._lock: Optional[asyncio.Lock] = None
        self._refresh_task: Optional[asyncio.Task] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def add_source(self, source: ConfigSource) -> None:
        """Add a configuration source."""
        self._sources.append(source)
        # Sort by priority (highest first)
        self._sources.sort(key=lambda s: s.priority, reverse=True)
        logger.info(f"Added config source: {source.name} (priority={source.priority})")

    def remove_source(self, name: str) -> None:
        """Remove a configuration source by name."""
        self._sources = [s for s in self._sources if s.name != name]

    async def get(
        self,
        key: str,
        default: Optional[T] = None,
        value_type: Optional[type] = None,
    ) -> Optional[T]:
        """Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if not found
            value_type: Type to cast value to

        Returns:
            Configuration value or default
        """
        # Check cache first
        if key in self._cache:
            cached = self._cache[key]
            value = cached.value
            if value_type:
                value = self._cast_value(value, value_type)
            return value

        # Try sources in priority order
        for source in self._sources:
            try:
                value = await source.get(key)
                if value is not None:
                    # Cache the value
                    self._cache[key] = ConfigValue(
                        key=key,
                        value=value,
                        source=source.name,
                        priority=source.priority,
                    )
                    if value_type:
                        value = self._cast_value(value, value_type)
                    return value
            except Exception as e:
                logger.warning(f"Error getting {key} from {source.name}: {e}")

        return default

    async def get_int(self, key: str, default: int = 0) -> int:
        """Get integer configuration value."""
        return await self.get(key, default, int)

    async def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float configuration value."""
        return await self.get(key, default, float)

    async def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean configuration value."""
        return await self.get(key, default, bool)

    async def get_str(self, key: str, default: str = "") -> str:
        """Get string configuration value."""
        return await self.get(key, default, str)

    async def get_list(self, key: str, default: Optional[List] = None) -> List:
        """Get list configuration value."""
        value = await self.get(key, default)
        if isinstance(value, str):
            # Try to parse as JSON
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                # Split by comma
                return [v.strip() for v in value.split(",")]
        return value if value is not None else (default or [])

    async def get_dict(self, key: str, default: Optional[Dict] = None) -> Dict:
        """Get dictionary configuration value."""
        value = await self.get(key, default)
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return default or {}
        return value if value is not None else (default or {})

    def _cast_value(self, value: Any, value_type: type) -> Any:
        """Cast value to specified type."""
        if value is None:
            return None

        if value_type == bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes", "on")
            return bool(value)

        if value_type == int:
            return int(value)

        if value_type == float:
            return float(value)

        if value_type == str:
            return str(value)

        return value

    async def set(self, key: str, value: Any, source_name: Optional[str] = None) -> bool:
        """Set a configuration value.

        Args:
            key: Configuration key
            value: Value to set
            source_name: Specific source to set in (optional)

        Returns:
            True if set successfully
        """
        async with self._get_lock():
            old_value = self._cache.get(key)

            # Try to set in sources
            for source in self._sources:
                if source_name and source.name != source_name:
                    continue
                try:
                    if await source.set(key, value):
                        # Update cache
                        self._cache[key] = ConfigValue(
                            key=key,
                            value=value,
                            source=source.name,
                            priority=source.priority,
                            version=(old_value.version + 1) if old_value else 1,
                        )

                        # Notify watchers
                        await self._notify_watchers(ConfigChangeEvent(
                            key=key,
                            old_value=old_value.value if old_value else None,
                            new_value=value,
                            source=source.name,
                        ))

                        return True
                except Exception as e:
                    logger.error(f"Error setting {key} in {source.name}: {e}")

            return False

    async def delete(self, key: str) -> bool:
        """Delete a configuration value."""
        async with self._get_lock():
            old_value = self._cache.pop(key, None)

            for source in self._sources:
                try:
                    await source.delete(key)
                except Exception as e:
                    logger.warning(f"Error deleting {key} from {source.name}: {e}")

            if old_value:
                await self._notify_watchers(ConfigChangeEvent(
                    key=key,
                    old_value=old_value.value,
                    new_value=None,
                    source="manager",
                ))

            return old_value is not None

    async def get_all(self, prefix: str = "") -> Dict[str, Any]:
        """Get all configuration values with optional prefix."""
        result: Dict[str, Any] = {}

        # Collect from all sources (lower priority first, so higher priority overwrites)
        for source in reversed(self._sources):
            try:
                values = await source.get_all(prefix)
                result.update(values)
            except Exception as e:
                logger.warning(f"Error getting all from {source.name}: {e}")

        return result

    def watch(self, key: str, callback: Callable[[ConfigChangeEvent], None]) -> None:
        """Watch for changes to a configuration key.

        Args:
            key: Configuration key to watch
            callback: Function called when value changes
        """
        if key not in self._watchers:
            self._watchers[key] = []
        self._watchers[key].append(callback)

        # Also register with sources that support watching
        for source in self._sources:
            try:
                asyncio.create_task(source.watch(key, callback))
            except Exception:
                pass

    def unwatch(self, key: str, callback: Optional[Callable] = None) -> None:
        """Stop watching a configuration key."""
        if key in self._watchers:
            if callback:
                self._watchers[key] = [c for c in self._watchers[key] if c != callback]
            else:
                del self._watchers[key]

    async def _notify_watchers(self, event: ConfigChangeEvent) -> None:
        """Notify watchers of a configuration change."""
        callbacks = self._watchers.get(event.key, [])
        # Also notify wildcard watchers
        callbacks.extend(self._watchers.get("*", []))

        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in config watcher callback: {e}")

    async def refresh(self) -> None:
        """Refresh all cached configuration values."""
        async with self._get_lock():
            keys = list(self._cache.keys())
            for key in keys:
                old_cached = self._cache.get(key)

                # Re-fetch from sources
                for source in self._sources:
                    try:
                        value = await source.get(key)
                        if value is not None:
                            if old_cached and old_cached.value != value:
                                # Value changed
                                self._cache[key] = ConfigValue(
                                    key=key,
                                    value=value,
                                    source=source.name,
                                    priority=source.priority,
                                    version=old_cached.version + 1,
                                )
                                await self._notify_watchers(ConfigChangeEvent(
                                    key=key,
                                    old_value=old_cached.value,
                                    new_value=value,
                                    source=source.name,
                                ))
                            break
                    except Exception as e:
                        logger.warning(f"Error refreshing {key} from {source.name}: {e}")

    async def start_auto_refresh(self, interval_seconds: float = 60.0) -> None:
        """Start automatic configuration refresh."""
        if self._refresh_task is not None:
            return

        async def refresh_loop():
            while True:
                try:
                    await asyncio.sleep(interval_seconds)
                    await self.refresh()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Config refresh error: {e}")

        self._refresh_task = asyncio.create_task(refresh_loop())
        logger.info(f"Started config auto-refresh (interval={interval_seconds}s)")

    async def stop_auto_refresh(self) -> None:
        """Stop automatic configuration refresh."""
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
            self._refresh_task = None

    async def close(self) -> None:
        """Close the configuration manager."""
        await self.stop_auto_refresh()
        for source in self._sources:
            await source.close()

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_keys": len(self._cache),
            "sources": [{"name": s.name, "priority": s.priority} for s in self._sources],
            "watchers": {k: len(v) for k, v in self._watchers.items()},
        }


# Global configuration manager
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def config_value(
    key: str,
    default: Optional[T] = None,
    value_type: Optional[type] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to inject configuration value.

    Args:
        key: Configuration key
        default: Default value
        value_type: Type to cast to

    Example:
        @config_value("app.timeout", default=30, value_type=int)
        async def process(timeout: int):
            ...
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            manager = get_config_manager()
            value = await manager.get(key, default, value_type)
            # Inject as first argument or keyword
            param_name = key.split(".")[-1]
            kwargs[param_name] = value
            return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            loop = asyncio.new_event_loop()
            try:
                manager = get_config_manager()
                value = loop.run_until_complete(manager.get(key, default, value_type))
                param_name = key.split(".")[-1]
                kwargs[param_name] = value
                return func(*args, **kwargs)
            finally:
                loop.close()

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
