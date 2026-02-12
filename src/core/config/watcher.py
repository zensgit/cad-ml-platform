"""Configuration Watcher Implementation.

Provides reactive configuration watching capabilities.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from src.core.config.manager import ConfigChangeEvent, ConfigManager

logger = logging.getLogger(__name__)

WatchCallback = Callable[[ConfigChangeEvent], None]


@dataclass
class WatchSubscription:
    """A subscription to configuration changes."""

    key: str
    callback: WatchCallback
    created_at: datetime = field(default_factory=datetime.utcnow)
    call_count: int = 0
    last_called: Optional[datetime] = None


class ConfigWatcher:
    """Reactive configuration watcher with advanced features."""

    def __init__(self, manager: Optional[ConfigManager] = None):
        self._manager = manager
        self._subscriptions: Dict[str, List[WatchSubscription]] = {}
        self._debounce_timers: Dict[str, asyncio.Task] = {}
        self._batch_events: List[ConfigChangeEvent] = []
        self._batch_callback: Optional[WatchCallback] = None
        self._batch_interval: float = 0
        self._batch_task: Optional[asyncio.Task] = None

    @property
    def manager(self) -> ConfigManager:
        if self._manager is None:
            from src.core.config.manager import get_config_manager
            self._manager = get_config_manager()
        return self._manager

    def watch(
        self,
        key: str,
        callback: WatchCallback,
        debounce_ms: int = 0,
    ) -> WatchSubscription:
        """Watch a configuration key for changes.

        Args:
            key: Configuration key to watch (supports wildcards like "app.*")
            callback: Function called when value changes
            debounce_ms: Debounce time in milliseconds

        Returns:
            WatchSubscription for unsubscribing
        """
        subscription = WatchSubscription(key=key, callback=callback)

        if key not in self._subscriptions:
            self._subscriptions[key] = []
        self._subscriptions[key].append(subscription)

        # Register with manager
        if debounce_ms > 0:
            wrapped_callback = self._create_debounced_callback(key, callback, debounce_ms)
        else:
            wrapped_callback = callback

        self.manager.watch(key, wrapped_callback)

        logger.debug(f"Watching config key: {key}")
        return subscription

    def unwatch(self, subscription: WatchSubscription) -> None:
        """Stop watching a configuration key."""
        if subscription.key in self._subscriptions:
            self._subscriptions[subscription.key] = [
                s for s in self._subscriptions[subscription.key]
                if s != subscription
            ]
            if not self._subscriptions[subscription.key]:
                del self._subscriptions[subscription.key]

        self.manager.unwatch(subscription.key, subscription.callback)
        logger.debug(f"Unwatched config key: {subscription.key}")

    def _create_debounced_callback(
        self,
        key: str,
        callback: WatchCallback,
        debounce_ms: int,
    ) -> WatchCallback:
        """Create a debounced callback."""
        async def debounced_callback(event: ConfigChangeEvent) -> None:
            # Cancel existing timer
            if key in self._debounce_timers:
                self._debounce_timers[key].cancel()

            async def delayed_call():
                await asyncio.sleep(debounce_ms / 1000)
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Debounced callback error: {e}")
                finally:
                    self._debounce_timers.pop(key, None)

            self._debounce_timers[key] = asyncio.create_task(delayed_call())

        return debounced_callback

    def watch_multiple(
        self,
        keys: List[str],
        callback: WatchCallback,
    ) -> List[WatchSubscription]:
        """Watch multiple configuration keys.

        Args:
            keys: List of keys to watch
            callback: Function called when any value changes

        Returns:
            List of subscriptions
        """
        return [self.watch(key, callback) for key in keys]

    def watch_prefix(
        self,
        prefix: str,
        callback: WatchCallback,
    ) -> WatchSubscription:
        """Watch all keys under a prefix.

        Args:
            prefix: Key prefix to watch
            callback: Function called when any matching value changes

        Returns:
            Subscription
        """
        pattern = f"{prefix}*" if not prefix.endswith("*") else prefix
        return self.watch(pattern, callback)

    def enable_batch_mode(
        self,
        callback: WatchCallback,
        interval_ms: int = 1000,
    ) -> None:
        """Enable batch mode for config changes.

        In batch mode, changes are collected and delivered together.

        Args:
            callback: Function called with batch of changes
            interval_ms: Batch interval in milliseconds
        """
        self._batch_callback = callback
        self._batch_interval = interval_ms / 1000

        async def batch_loop():
            while True:
                await asyncio.sleep(self._batch_interval)
                if self._batch_events and self._batch_callback:
                    events = self._batch_events.copy()
                    self._batch_events.clear()
                    try:
                        if asyncio.iscoroutinefunction(self._batch_callback):
                            await self._batch_callback(events)
                        else:
                            self._batch_callback(events)
                    except Exception as e:
                        logger.error(f"Batch callback error: {e}")

        self._batch_task = asyncio.create_task(batch_loop())

        # Register wildcard watcher to collect events
        self.manager.watch("*", self._collect_batch_event)

    def _collect_batch_event(self, event: ConfigChangeEvent) -> None:
        """Collect event for batch processing."""
        self._batch_events.append(event)

    def disable_batch_mode(self) -> None:
        """Disable batch mode."""
        if self._batch_task:
            self._batch_task.cancel()
            try:
                asyncio.get_event_loop().run_until_complete(self._batch_task)
            except (asyncio.CancelledError, RuntimeError):
                pass
            self._batch_task = None

        self._batch_callback = None
        self._batch_events.clear()
        self.manager.unwatch("*", self._collect_batch_event)

    def get_subscriptions(self) -> Dict[str, int]:
        """Get subscription counts by key."""
        return {k: len(v) for k, v in self._subscriptions.items()}


class ReactiveConfig:
    """Reactive configuration value that updates automatically."""

    def __init__(
        self,
        manager: ConfigManager,
        key: str,
        default: Any = None,
    ):
        self._manager = manager
        self._key = key
        self._default = default
        self._value: Any = None
        self._loaded = False
        self._callbacks: List[Callable[[Any, Any], None]] = []
        self._subscription: Optional[WatchSubscription] = None

    async def _load(self) -> None:
        """Load initial value."""
        self._value = await self._manager.get(self._key, self._default)
        self._loaded = True

    async def get(self) -> Any:
        """Get current value."""
        if not self._loaded:
            await self._load()
        return self._value

    def subscribe(self, callback: Callable[[Any, Any], None]) -> None:
        """Subscribe to value changes.

        Args:
            callback: Function(old_value, new_value)
        """
        self._callbacks.append(callback)

        # Setup watcher if first subscriber
        if len(self._callbacks) == 1:
            self._manager.watch(self._key, self._on_change)

    def unsubscribe(self, callback: Callable[[Any, Any], None]) -> None:
        """Unsubscribe from value changes."""
        self._callbacks = [c for c in self._callbacks if c != callback]

        if not self._callbacks:
            self._manager.unwatch(self._key, self._on_change)

    def _on_change(self, event: ConfigChangeEvent) -> None:
        """Handle configuration change."""
        old_value = self._value
        self._value = event.new_value

        for callback in self._callbacks:
            try:
                callback(old_value, self._value)
            except Exception as e:
                logger.error(f"Reactive config callback error: {e}")


def create_reactive_config(
    key: str,
    default: Any = None,
    manager: Optional[ConfigManager] = None,
) -> ReactiveConfig:
    """Create a reactive configuration value.

    Args:
        key: Configuration key
        default: Default value
        manager: Optional config manager

    Returns:
        ReactiveConfig instance
    """
    from src.core.config.manager import get_config_manager
    return ReactiveConfig(
        manager=manager or get_config_manager(),
        key=key,
        default=default,
    )
