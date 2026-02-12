"""Feature Toggle Store.

Provides toggle storage backends:
- In-memory store
- File-based store
- Remote store interface
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from src.core.feature_toggles_enhanced.toggle import (
    EvaluationContext,
    FeatureToggle,
    ToggleListener,
    ToggleMetadata,
    ToggleRule,
    ToggleState,
    ToggleType,
    AlwaysOnRule,
    AlwaysOffRule,
    PercentageRule,
    UserIdRule,
    AttributeRule,
    TimeBasedRule,
)

logger = logging.getLogger(__name__)


class ToggleStore(ABC):
    """Abstract base class for toggle storage."""

    @abstractmethod
    async def get(self, name: str) -> Optional[FeatureToggle]:
        """Get a toggle by name."""
        pass

    @abstractmethod
    async def get_all(self) -> List[FeatureToggle]:
        """Get all toggles."""
        pass

    @abstractmethod
    async def save(self, toggle: FeatureToggle) -> None:
        """Save a toggle."""
        pass

    @abstractmethod
    async def delete(self, name: str) -> bool:
        """Delete a toggle."""
        pass

    @abstractmethod
    async def exists(self, name: str) -> bool:
        """Check if toggle exists."""
        pass

    async def list_toggles(self) -> List[str]:
        """List toggle names (backward-compatible helper)."""
        toggles = await self.get_all()
        return [toggle.name for toggle in toggles]


class InMemoryToggleStore(ToggleStore):
    """In-memory toggle storage."""

    def __init__(self):
        self._toggles: Dict[str, FeatureToggle] = {}
        self._listeners: List[ToggleListener] = []
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def add_listener(self, listener: ToggleListener) -> None:
        """Add a toggle change listener."""
        self._listeners.append(listener)

    async def get(self, name: str) -> Optional[FeatureToggle]:
        async with self._get_lock():
            return self._toggles.get(name)

    async def get_all(self) -> List[FeatureToggle]:
        async with self._get_lock():
            return list(self._toggles.values())

    async def save(self, toggle: FeatureToggle) -> None:
        async with self._get_lock():
            old_state = None
            if toggle.name in self._toggles:
                old_state = self._toggles[toggle.name].state

            toggle.metadata.updated_at = datetime.utcnow()
            self._toggles[toggle.name] = toggle

            # Notify listeners
            if old_state != toggle.state:
                for listener in self._listeners:
                    try:
                        listener.on_toggle_changed(toggle.name, old_state, toggle.state)
                    except Exception as e:
                        logger.error(f"Listener error: {e}")

    async def delete(self, name: str) -> bool:
        async with self._get_lock():
            if name in self._toggles:
                del self._toggles[name]
                return True
            return False

    async def exists(self, name: str) -> bool:
        async with self._get_lock():
            return name in self._toggles

    async def list_toggles(self) -> List[str]:
        async with self._get_lock():
            return list(self._toggles.keys())


class FileToggleStore(ToggleStore):
    """File-based toggle storage."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self._toggles: Dict[str, FeatureToggle] = {}
        self._loaded = False
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def _ensure_loaded(self) -> None:
        """Load toggles from file if not already loaded."""
        if self._loaded:
            return

        if self.file_path.exists():
            try:
                content = self.file_path.read_text()
                data = json.loads(content)
                for toggle_data in data.get("toggles", []):
                    toggle = self._deserialize_toggle(toggle_data)
                    self._toggles[toggle.name] = toggle
            except Exception as e:
                logger.error(f"Failed to load toggles from file: {e}")

        self._loaded = True

    async def _save_to_file(self) -> None:
        """Save toggles to file."""
        data = {
            "toggles": [
                self._serialize_toggle(t) for t in self._toggles.values()
            ],
            "updated_at": datetime.utcnow().isoformat(),
        }
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(json.dumps(data, indent=2))

    def _serialize_toggle(self, toggle: FeatureToggle) -> Dict[str, Any]:
        """Serialize toggle to dict."""
        return {
            "name": toggle.name,
            "state": toggle.state.value,
            "default_value": toggle.default_value,
            "variants": toggle.variants,
            "metadata": {
                "description": toggle.metadata.description,
                "toggle_type": toggle.metadata.toggle_type.value,
                "owner": toggle.metadata.owner,
                "tags": list(toggle.metadata.tags),
            },
            "rules": [self._serialize_rule(r) for r in toggle.rules],
        }

    def _serialize_rule(self, rule: ToggleRule) -> Dict[str, Any]:
        """Serialize rule to dict."""
        if isinstance(rule, AlwaysOnRule):
            return {"type": "always_on"}
        elif isinstance(rule, AlwaysOffRule):
            return {"type": "always_off"}
        elif isinstance(rule, PercentageRule):
            return {
                "type": "percentage",
                "percentage": rule.percentage,
                "sticky": rule.sticky,
                "sticky_key": rule.sticky_key,
            }
        elif isinstance(rule, UserIdRule):
            return {
                "type": "user_id",
                "user_ids": list(rule.user_ids),
                "include": rule.include,
            }
        elif isinstance(rule, AttributeRule):
            return {
                "type": "attribute",
                "attribute": rule.attribute,
                "operator": rule.operator,
                "value": rule.value,
            }
        else:
            return {"type": "unknown"}

    def _deserialize_toggle(self, data: Dict[str, Any]) -> FeatureToggle:
        """Deserialize toggle from dict."""
        metadata = ToggleMetadata(
            name=data["name"],
            description=data.get("metadata", {}).get("description", ""),
            toggle_type=ToggleType(data.get("metadata", {}).get("toggle_type", "release")),
            owner=data.get("metadata", {}).get("owner", ""),
            tags=set(data.get("metadata", {}).get("tags", [])),
        )

        rules = [
            self._deserialize_rule(r) for r in data.get("rules", [])
            if self._deserialize_rule(r) is not None
        ]

        return FeatureToggle(
            name=data["name"],
            state=ToggleState(data.get("state", "off")),
            default_value=data.get("default_value", False),
            variants=data.get("variants", {}),
            metadata=metadata,
            rules=rules,
        )

    def _deserialize_rule(self, data: Dict[str, Any]) -> Optional[ToggleRule]:
        """Deserialize rule from dict."""
        rule_type = data.get("type")

        if rule_type == "always_on":
            return AlwaysOnRule()
        elif rule_type == "always_off":
            return AlwaysOffRule()
        elif rule_type == "percentage":
            return PercentageRule(
                percentage=data["percentage"],
                sticky=data.get("sticky", True),
                sticky_key=data.get("sticky_key", "user_id"),
            )
        elif rule_type == "user_id":
            return UserIdRule(
                user_ids=set(data["user_ids"]),
                include=data.get("include", True),
            )
        elif rule_type == "attribute":
            return AttributeRule(
                attribute=data["attribute"],
                operator=data["operator"],
                value=data["value"],
            )
        else:
            return None

    async def get(self, name: str) -> Optional[FeatureToggle]:
        async with self._get_lock():
            await self._ensure_loaded()
            return self._toggles.get(name)

    async def get_all(self) -> List[FeatureToggle]:
        async with self._get_lock():
            await self._ensure_loaded()
            return list(self._toggles.values())

    async def save(self, toggle: FeatureToggle) -> None:
        async with self._get_lock():
            await self._ensure_loaded()
            toggle.metadata.updated_at = datetime.utcnow()
            self._toggles[toggle.name] = toggle
            await self._save_to_file()

    async def delete(self, name: str) -> bool:
        async with self._get_lock():
            await self._ensure_loaded()
            if name in self._toggles:
                del self._toggles[name]
                await self._save_to_file()
                return True
            return False

    async def exists(self, name: str) -> bool:
        async with self._get_lock():
            await self._ensure_loaded()
            return name in self._toggles


class CachingToggleStore(ToggleStore):
    """Caching wrapper for toggle stores."""

    def __init__(
        self,
        delegate: ToggleStore,
        cache_ttl: float = 60.0,
    ):
        self.delegate = delegate
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, tuple[FeatureToggle, float]] = {}
        self._all_cache: Optional[tuple[List[FeatureToggle], float]] = None
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _is_valid(self, cached_at: float) -> bool:
        return time.time() - cached_at < self.cache_ttl

    async def get(self, name: str) -> Optional[FeatureToggle]:
        async with self._get_lock():
            if name in self._cache:
                toggle, cached_at = self._cache[name]
                if self._is_valid(cached_at):
                    return toggle

            toggle = await self.delegate.get(name)
            if toggle:
                self._cache[name] = (toggle, time.time())
            return toggle

    async def get_all(self) -> List[FeatureToggle]:
        async with self._get_lock():
            if self._all_cache:
                toggles, cached_at = self._all_cache
                if self._is_valid(cached_at):
                    return toggles

            toggles = await self.delegate.get_all()
            self._all_cache = (toggles, time.time())
            return toggles

    async def save(self, toggle: FeatureToggle) -> None:
        await self.delegate.save(toggle)
        async with self._get_lock():
            self._cache[toggle.name] = (toggle, time.time())
            self._all_cache = None  # Invalidate

    async def delete(self, name: str) -> bool:
        result = await self.delegate.delete(name)
        async with self._get_lock():
            self._cache.pop(name, None)
            self._all_cache = None
        return result

    async def exists(self, name: str) -> bool:
        return await self.delegate.exists(name)

    def invalidate(self, name: Optional[str] = None) -> None:
        """Invalidate cache."""
        if name:
            self._cache.pop(name, None)
        else:
            self._cache.clear()
        self._all_cache = None
