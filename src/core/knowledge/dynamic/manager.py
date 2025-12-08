"""
Knowledge Manager.

Central management component for dynamic knowledge base:
- Hot-reload support
- Version tracking
- Event notifications
- Cache management
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from src.core.knowledge.dynamic.models import (
    KnowledgeCategory,
    KnowledgeEntry,
    GeometryPattern,
)
from src.core.knowledge.dynamic.store import KnowledgeStore, JSONKnowledgeStore

logger = logging.getLogger(__name__)


class KnowledgeManager:
    """
    Central manager for dynamic knowledge base.

    Features:
    - Hot-reload from storage
    - Optimized lookup caches
    - Change notification callbacks
    - Thread-safe operations
    """

    _instance: Optional["KnowledgeManager"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        store: Optional[KnowledgeStore] = None,
        auto_reload_interval: int = 0,  # seconds, 0 = disabled
    ):
        """Initialize the knowledge manager.

        Args:
            store: Knowledge storage backend
            auto_reload_interval: Auto-reload interval in seconds (0 to disable)
        """
        # Prevent re-initialization
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._store = store or JSONKnowledgeStore()
        self._auto_reload_interval = auto_reload_interval
        self._last_version = self._store.get_version()

        # Caches
        self._keyword_cache: Dict[str, List[KnowledgeEntry]] = {}
        self._pattern_cache: Dict[str, List[KnowledgeEntry]] = {}
        self._geometry_patterns: List[GeometryPattern] = []
        self._category_cache: Dict[KnowledgeCategory, List[KnowledgeEntry]] = {}

        # Change notification
        self._change_callbacks: List[Callable[[str], None]] = []

        # Threading
        self._cache_lock = threading.RLock()
        self._reload_thread: Optional[threading.Thread] = None
        self._stop_reload = threading.Event()

        # Build initial caches
        self._rebuild_caches()

        # Start auto-reload if enabled
        if auto_reload_interval > 0:
            self._start_auto_reload()

        self._initialized = True
        logger.info(f"KnowledgeManager initialized, version: {self._last_version}")

    def _rebuild_caches(self) -> None:
        """Rebuild all lookup caches."""
        with self._cache_lock:
            self._keyword_cache.clear()
            self._pattern_cache.clear()
            self._geometry_patterns.clear()
            self._category_cache.clear()

            all_rules = self._store.get_all()

            for rule in all_rules:
                if not rule.enabled:
                    continue

                # Category cache
                if rule.category not in self._category_cache:
                    self._category_cache[rule.category] = []
                self._category_cache[rule.category].append(rule)

                # Keyword cache
                for keyword in rule.keywords:
                    kw_lower = keyword.lower()
                    if kw_lower not in self._keyword_cache:
                        self._keyword_cache[kw_lower] = []
                    self._keyword_cache[kw_lower].append(rule)

                # Pattern cache
                for pattern in rule.ocr_patterns:
                    if pattern not in self._pattern_cache:
                        self._pattern_cache[pattern] = []
                    self._pattern_cache[pattern].append(rule)

                # Geometry patterns
                if isinstance(rule, GeometryPattern):
                    self._geometry_patterns.append(rule)

            logger.debug(
                f"Cache rebuilt: {len(all_rules)} rules, "
                f"{len(self._keyword_cache)} keywords, "
                f"{len(self._pattern_cache)} patterns, "
                f"{len(self._geometry_patterns)} geometry patterns"
            )

    def _start_auto_reload(self) -> None:
        """Start the auto-reload background thread."""
        def reload_loop():
            while not self._stop_reload.is_set():
                time.sleep(self._auto_reload_interval)
                if self._stop_reload.is_set():
                    break
                self.check_and_reload()

        self._reload_thread = threading.Thread(
            target=reload_loop,
            daemon=True,
            name="KnowledgeAutoReload",
        )
        self._reload_thread.start()
        logger.info(f"Auto-reload started, interval: {self._auto_reload_interval}s")

    def stop_auto_reload(self) -> None:
        """Stop the auto-reload background thread."""
        self._stop_reload.set()
        if self._reload_thread:
            self._reload_thread.join(timeout=5)
            self._reload_thread = None
        logger.info("Auto-reload stopped")

    def check_and_reload(self) -> bool:
        """Check if knowledge has changed and reload if needed."""
        current_version = self._store.get_version()
        if current_version != self._last_version:
            self.reload()
            return True
        return False

    def reload(self) -> None:
        """Force reload knowledge from storage."""
        if hasattr(self._store, "reload"):
            self._store.reload()

        self._rebuild_caches()
        self._last_version = self._store.get_version()

        # Notify listeners
        for callback in self._change_callbacks:
            try:
                callback(self._last_version)
            except Exception as e:
                logger.error(f"Error in change callback: {e}")

        logger.info(f"Knowledge reloaded, version: {self._last_version}")

    def on_change(self, callback: Callable[[str], None]) -> None:
        """Register a callback for knowledge changes.

        Args:
            callback: Function to call when knowledge changes (receives version string)
        """
        self._change_callbacks.append(callback)

    # ==================== Query Methods ====================

    def get_rules_by_category(
        self, category: KnowledgeCategory
    ) -> List[KnowledgeEntry]:
        """Get all enabled rules for a category."""
        with self._cache_lock:
            return self._category_cache.get(category, []).copy()

    def match_keywords(self, text: str) -> Dict[str, List[KnowledgeEntry]]:
        """Find all rules matching keywords in text.

        Args:
            text: Text to search

        Returns:
            Dict mapping matched keywords to their rules
        """
        text_lower = text.lower()
        matches = {}

        with self._cache_lock:
            for keyword, rules in self._keyword_cache.items():
                if keyword in text_lower:
                    matches[keyword] = rules

        return matches

    def match_patterns(self, text: str) -> Dict[str, List[KnowledgeEntry]]:
        """Find all rules matching OCR patterns in text.

        Args:
            text: Text to search

        Returns:
            Dict mapping matched patterns to their rules
        """
        matches = {}

        with self._cache_lock:
            for pattern, rules in self._pattern_cache.items():
                try:
                    if re.search(pattern, text, re.IGNORECASE):
                        matches[pattern] = rules
                except re.error:
                    pass  # Skip invalid patterns

        return matches

    def match_geometry(
        self,
        geometric_features: Dict[str, float],
        entity_counts: Dict[str, int],
    ) -> List[GeometryPattern]:
        """Find all geometry patterns matching the given features.

        Args:
            geometric_features: Geometric feature dict
            entity_counts: Entity kind counts

        Returns:
            List of matching geometry patterns
        """
        matches = []

        with self._cache_lock:
            for pattern in self._geometry_patterns:
                if pattern.matches(geometric_features, entity_counts):
                    matches.append(pattern)

        return sorted(matches, key=lambda p: -p.priority)

    def get_part_hints(
        self,
        text: str,
        geometric_features: Optional[Dict[str, float]] = None,
        entity_counts: Optional[Dict[str, int]] = None,
    ) -> Dict[str, float]:
        """Get part type hints from all matching rules.

        Args:
            text: OCR text to analyze
            geometric_features: Optional geometric features
            entity_counts: Optional entity counts

        Returns:
            Dict mapping part types to aggregated scores
        """
        hints: Dict[str, float] = {}

        # Keyword matches
        keyword_matches = self.match_keywords(text)
        for keyword, rules in keyword_matches.items():
            for rule in rules:
                for part, score in rule.part_hints.items():
                    hints[part] = hints.get(part, 0) + score

        # Pattern matches
        pattern_matches = self.match_patterns(text)
        for pattern, rules in pattern_matches.items():
            for rule in rules:
                for part, score in rule.part_hints.items():
                    hints[part] = hints.get(part, 0) + score

        # Geometry matches
        if geometric_features or entity_counts:
            geo_features = geometric_features or {}
            ent_counts = entity_counts or {}
            geometry_matches = self.match_geometry(geo_features, ent_counts)
            for pattern in geometry_matches:
                for part, score in pattern.part_hints.items():
                    hints[part] = hints.get(part, 0) + score

        # Normalize to max 1.0
        for part in hints:
            hints[part] = min(hints[part], 1.0)

        return hints

    # ==================== CRUD Methods ====================

    def add_rule(self, rule: KnowledgeEntry) -> str:
        """Add a new rule.

        Args:
            rule: Rule to add

        Returns:
            Rule ID
        """
        rule_id = self._store.save(rule)
        self._rebuild_caches()
        self._notify_change()
        return rule_id

    def update_rule(self, rule: KnowledgeEntry) -> str:
        """Update an existing rule.

        Args:
            rule: Rule to update

        Returns:
            Rule ID
        """
        rule_id = self._store.save(rule)
        self._rebuild_caches()
        self._notify_change()
        return rule_id

    def delete_rule(self, rule_id: str) -> bool:
        """Delete a rule by ID.

        Args:
            rule_id: ID of rule to delete

        Returns:
            True if deleted, False if not found
        """
        result = self._store.delete(rule_id)
        if result:
            self._rebuild_caches()
            self._notify_change()
        return result

    def get_rule(self, rule_id: str) -> Optional[KnowledgeEntry]:
        """Get a rule by ID."""
        return self._store.get(rule_id)

    def search_rules(
        self,
        query: str,
        category: Optional[KnowledgeCategory] = None,
        limit: int = 50,
    ) -> List[KnowledgeEntry]:
        """Search rules by keyword."""
        return self._store.search(query, category, limit)

    def _notify_change(self) -> None:
        """Notify listeners of changes."""
        version = self._store.get_version()
        self._last_version = version
        for callback in self._change_callbacks:
            try:
                callback(version)
            except Exception as e:
                logger.error(f"Error in change callback: {e}")

    # ==================== Import/Export ====================

    def export_knowledge(self) -> Dict[str, Any]:
        """Export all knowledge as a dictionary."""
        return self._store.export_all()

    def import_knowledge(self, data: Dict[str, Any], merge: bool = True) -> int:
        """Import knowledge from a dictionary.

        Args:
            data: Knowledge data to import
            merge: If True, merge with existing; if False, replace all

        Returns:
            Number of rules imported
        """
        count = self._store.import_all(data, merge)
        self._rebuild_caches()
        self._notify_change()
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        if hasattr(self._store, "get_stats"):
            stats = self._store.get_stats()
        else:
            stats = {
                "total_rules": len(self._store.get_all()),
                "version": self._store.get_version(),
            }

        stats["cache_stats"] = {
            "keywords": len(self._keyword_cache),
            "patterns": len(self._pattern_cache),
            "geometry_patterns": len(self._geometry_patterns),
        }

        return stats

    def get_version(self) -> str:
        """Get current knowledge version."""
        return self._last_version


# Global instance accessor
_manager_instance: Optional[KnowledgeManager] = None


def get_knowledge_manager(
    store: Optional[KnowledgeStore] = None,
    auto_reload_interval: int = 0,
) -> KnowledgeManager:
    """Get the global KnowledgeManager instance.

    Args:
        store: Optional storage backend (used only on first call)
        auto_reload_interval: Auto-reload interval in seconds

    Returns:
        KnowledgeManager singleton instance
    """
    global _manager_instance

    if _manager_instance is None:
        _manager_instance = KnowledgeManager(
            store=store,
            auto_reload_interval=auto_reload_interval,
        )

    return _manager_instance
