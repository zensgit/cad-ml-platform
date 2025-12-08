"""
Knowledge storage backends.

Provides abstract interface and concrete implementations for
persisting knowledge rules.
"""

from __future__ import annotations

import json
import logging
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.core.knowledge.dynamic.models import (
    KnowledgeCategory,
    KnowledgeEntry,
    create_rule_from_dict,
)

logger = logging.getLogger(__name__)


class KnowledgeStore(ABC):
    """Abstract base class for knowledge storage."""

    @abstractmethod
    def get(self, rule_id: str) -> Optional[KnowledgeEntry]:
        """Get a single rule by ID."""
        pass

    @abstractmethod
    def get_all(self, category: Optional[KnowledgeCategory] = None) -> List[KnowledgeEntry]:
        """Get all rules, optionally filtered by category."""
        pass

    @abstractmethod
    def save(self, rule: KnowledgeEntry) -> str:
        """Save a rule, return its ID."""
        pass

    @abstractmethod
    def delete(self, rule_id: str) -> bool:
        """Delete a rule by ID."""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        category: Optional[KnowledgeCategory] = None,
        limit: int = 50,
    ) -> List[KnowledgeEntry]:
        """Search rules by keyword."""
        pass

    @abstractmethod
    def get_version(self) -> str:
        """Get current knowledge version/timestamp."""
        pass

    @abstractmethod
    def export_all(self) -> Dict[str, Any]:
        """Export all knowledge as a dictionary."""
        pass

    @abstractmethod
    def import_all(self, data: Dict[str, Any], merge: bool = True) -> int:
        """Import knowledge from dictionary. Returns count of imported rules."""
        pass


class JSONKnowledgeStore(KnowledgeStore):
    """JSON file-based knowledge storage."""

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/knowledge",
        auto_save: bool = True,
    ):
        """Initialize JSON store.

        Args:
            data_dir: Directory for JSON files
            auto_save: Whether to auto-save after modifications
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.auto_save = auto_save

        self._rules: Dict[str, KnowledgeEntry] = {}
        self._lock = threading.RLock()
        self._version = datetime.now().isoformat()
        self._dirty = False

        self._load_all()

    def _get_file_path(self, category: KnowledgeCategory) -> Path:
        """Get JSON file path for a category."""
        return self.data_dir / f"{category.value}_rules.json"

    def _load_all(self) -> None:
        """Load all rules from JSON files."""
        with self._lock:
            self._rules.clear()

            for category in KnowledgeCategory:
                file_path = self._get_file_path(category)
                if file_path.exists():
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        for rule_data in data.get("rules", []):
                            rule = create_rule_from_dict(rule_data)
                            self._rules[rule.id] = rule

                        logger.info(
                            f"Loaded {len(data.get('rules', []))} rules from {file_path}"
                        )
                    except Exception as e:
                        logger.error(f"Error loading {file_path}: {e}")

            self._version = datetime.now().isoformat()
            self._dirty = False

    def _save_category(self, category: KnowledgeCategory) -> None:
        """Save rules for a specific category."""
        file_path = self._get_file_path(category)

        rules = [
            rule.to_dict()
            for rule in self._rules.values()
            if rule.category == category
        ]

        data = {
            "version": self._version,
            "category": category.value,
            "count": len(rules),
            "updated_at": datetime.now().isoformat(),
            "rules": rules,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(rules)} rules to {file_path}")

    def _save_all(self) -> None:
        """Save all rules to JSON files."""
        with self._lock:
            for category in KnowledgeCategory:
                self._save_category(category)
            self._dirty = False

    def get(self, rule_id: str) -> Optional[KnowledgeEntry]:
        """Get a single rule by ID."""
        with self._lock:
            return self._rules.get(rule_id)

    def get_all(self, category: Optional[KnowledgeCategory] = None) -> List[KnowledgeEntry]:
        """Get all rules, optionally filtered by category."""
        with self._lock:
            if category is None:
                rules = list(self._rules.values())
            else:
                rules = [r for r in self._rules.values() if r.category == category]

            # Sort by priority (descending) then name
            return sorted(rules, key=lambda r: (-r.priority, r.name))

    def save(self, rule: KnowledgeEntry) -> str:
        """Save a rule, return its ID."""
        with self._lock:
            rule.updated_at = datetime.now().isoformat()
            self._rules[rule.id] = rule
            self._dirty = True
            self._version = datetime.now().isoformat()

            if self.auto_save:
                self._save_category(rule.category)

            return rule.id

    def save_batch(self, rules: List[KnowledgeEntry]) -> int:
        """Save multiple rules at once."""
        with self._lock:
            categories_affected = set()

            for rule in rules:
                rule.updated_at = datetime.now().isoformat()
                self._rules[rule.id] = rule
                categories_affected.add(rule.category)

            self._dirty = True
            self._version = datetime.now().isoformat()

            if self.auto_save:
                for category in categories_affected:
                    self._save_category(category)

            return len(rules)

    def delete(self, rule_id: str) -> bool:
        """Delete a rule by ID."""
        with self._lock:
            if rule_id in self._rules:
                category = self._rules[rule_id].category
                del self._rules[rule_id]
                self._dirty = True
                self._version = datetime.now().isoformat()

                if self.auto_save:
                    self._save_category(category)

                return True
            return False

    def search(
        self,
        query: str,
        category: Optional[KnowledgeCategory] = None,
        limit: int = 50,
    ) -> List[KnowledgeEntry]:
        """Search rules by keyword."""
        query_lower = query.lower()
        results = []

        with self._lock:
            for rule in self._rules.values():
                if category and rule.category != category:
                    continue

                # Search in name, chinese_name, description, keywords
                searchable = " ".join([
                    rule.name,
                    rule.chinese_name,
                    rule.description,
                    " ".join(rule.keywords),
                ]).lower()

                if query_lower in searchable:
                    results.append(rule)

                if len(results) >= limit:
                    break

        return sorted(results, key=lambda r: (-r.priority, r.name))

    def get_version(self) -> str:
        """Get current knowledge version/timestamp."""
        return self._version

    def reload(self) -> None:
        """Reload all rules from disk."""
        self._load_all()
        logger.info(f"Reloaded knowledge base, version: {self._version}")

    def export_all(self) -> Dict[str, Any]:
        """Export all knowledge as a dictionary."""
        with self._lock:
            categories_data = {}

            for category in KnowledgeCategory:
                rules = [
                    r.to_dict()
                    for r in self._rules.values()
                    if r.category == category
                ]
                categories_data[category.value] = {
                    "count": len(rules),
                    "rules": rules,
                }

            return {
                "version": self._version,
                "exported_at": datetime.now().isoformat(),
                "total_rules": len(self._rules),
                "categories": categories_data,
            }

    def import_all(self, data: Dict[str, Any], merge: bool = True) -> int:
        """Import knowledge from dictionary."""
        imported_count = 0

        with self._lock:
            if not merge:
                self._rules.clear()

            categories = data.get("categories", {})
            for category_name, category_data in categories.items():
                for rule_data in category_data.get("rules", []):
                    try:
                        rule = create_rule_from_dict(rule_data)
                        rule.source = "imported"
                        self._rules[rule.id] = rule
                        imported_count += 1
                    except Exception as e:
                        logger.error(f"Error importing rule: {e}")

            self._dirty = True
            self._version = datetime.now().isoformat()

            if self.auto_save:
                self._save_all()

        logger.info(f"Imported {imported_count} rules")
        return imported_count

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge store."""
        with self._lock:
            stats = {
                "total_rules": len(self._rules),
                "version": self._version,
                "categories": {},
            }

            for category in KnowledgeCategory:
                count = sum(
                    1 for r in self._rules.values()
                    if r.category == category
                )
                enabled = sum(
                    1 for r in self._rules.values()
                    if r.category == category and r.enabled
                )
                stats["categories"][category.value] = {  # type: ignore[index]
                    "total": count,
                    "enabled": enabled,
                }

            return stats


class InMemoryKnowledgeStore(KnowledgeStore):
    """In-memory knowledge storage (for testing)."""

    def __init__(self) -> None:
        self._rules: Dict[str, KnowledgeEntry] = {}
        self._version = datetime.now().isoformat()
        self._lock = threading.RLock()

    def get(self, rule_id: str) -> Optional[KnowledgeEntry]:
        with self._lock:
            return self._rules.get(rule_id)

    def get_all(self, category: Optional[KnowledgeCategory] = None) -> List[KnowledgeEntry]:
        with self._lock:
            if category is None:
                return list(self._rules.values())
            return [r for r in self._rules.values() if r.category == category]

    def save(self, rule: KnowledgeEntry) -> str:
        with self._lock:
            rule.updated_at = datetime.now().isoformat()
            self._rules[rule.id] = rule
            self._version = datetime.now().isoformat()
            return rule.id

    def delete(self, rule_id: str) -> bool:
        with self._lock:
            if rule_id in self._rules:
                del self._rules[rule_id]
                self._version = datetime.now().isoformat()
                return True
            return False

    def search(
        self,
        query: str,
        category: Optional[KnowledgeCategory] = None,
        limit: int = 50,
    ) -> List[KnowledgeEntry]:
        query_lower = query.lower()
        results = []

        with self._lock:
            for rule in self._rules.values():
                if category and rule.category != category:
                    continue

                searchable = " ".join([
                    rule.name,
                    rule.chinese_name,
                    rule.description,
                    " ".join(rule.keywords),
                ]).lower()

                if query_lower in searchable:
                    results.append(rule)

                if len(results) >= limit:
                    break

        return results

    def get_version(self) -> str:
        return self._version

    def export_all(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "version": self._version,
                "rules": [r.to_dict() for r in self._rules.values()],
            }

    def import_all(self, data: Dict[str, Any], merge: bool = True) -> int:
        count = 0
        with self._lock:
            if not merge:
                self._rules.clear()

            for rule_data in data.get("rules", []):
                rule = create_rule_from_dict(rule_data)
                self._rules[rule.id] = rule
                count += 1

            self._version = datetime.now().isoformat()

        return count
