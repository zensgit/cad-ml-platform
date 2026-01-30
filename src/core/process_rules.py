"""Process recommendation rules loader.

Phase 1: YAML based static mapping (material x complexity x volume_range)->processes.
If YAML missing falls back to embedded defaults.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Union

import yaml

DEFAULT_RULES = {
    "steel": {
        "low": [
            {"max_volume": 1e4, "primary": "cnc_machining", "alternatives": ["sheet_metal"]},
            {
                "max_volume": 5e5,
                "primary": "cnc_machining",
                "alternatives": ["casting", "additive"],
            },
        ],
        "medium": [
            {"max_volume": 1e6, "primary": "casting", "alternatives": ["cnc_finish", "additive"]},
        ],
        "high": [
            {
                "max_volume": math.inf,
                "primary": "casting",
                "alternatives": ["additive", "cnc_finish"],
            },
        ],
    },
    "aluminum": {
        "low": [
            {"max_volume": 5e4, "primary": "cnc_machining", "alternatives": ["die_casting"]},
            {
                "max_volume": 2e5,
                "primary": "die_casting",
                "alternatives": ["cnc_finish", "additive"],
            },
        ],
        "medium": [
            {"max_volume": 1e6, "primary": "die_casting", "alternatives": ["cnc_finish"]},
        ],
        "high": [
            {"max_volume": math.inf, "primary": "die_casting", "alternatives": ["cnc_finish"]},
        ],
    },
}

_rules_cache: Optional[Dict[str, Any]] = None
_rules_mtime: Optional[float] = None


def load_rules(path: Optional[str] = None, *, force_reload: bool = False) -> Dict[str, Any]:
    global _rules_cache, _rules_mtime
    path = path or os.getenv("PROCESS_RULES_FILE", "config/process_rules.yaml")
    if not force_reload and _rules_cache is not None:
        # hot reload if file changed (mtime differs)
        try:
            if os.path.exists(path):
                mtime = os.path.getmtime(path)
                if _rules_mtime and mtime != _rules_mtime:
                    force_reload = True
        except Exception:
            pass
        if not force_reload:
            return _rules_cache
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                if isinstance(data, dict) and data:
                    _rules_cache = data
                    _rules_mtime = os.path.getmtime(path)
                    _rules_cache.setdefault("__meta__", {})
                    _rules_cache["__meta__"]["version"] = os.getenv("PROCESS_RULE_VERSION", "v1")
                    return _rules_cache
        except Exception:
            pass
    if _rules_cache is None or force_reload:
        _rules_cache = DEFAULT_RULES
        _rules_mtime = None
        _rules_cache.setdefault("__meta__", {})
        _rules_cache["__meta__"]["version"] = os.getenv("PROCESS_RULE_VERSION", "v1")
    return _rules_cache


def recommend(material: str, complexity: str, volume: float) -> Dict[str, Any]:
    rules = load_rules()
    material_rules = rules.get(material.lower()) or rules.get("steel", {})
    complexity_rules: List[Dict[str, Any]] = material_rules.get(complexity, [])
    version = rules.get("__meta__", {}).get("version", "v1")
    for rule in complexity_rules:
        if volume <= rule.get("max_volume", math.inf):
            return {
                "primary": rule.get("primary"),
                "alternatives": rule.get("alternatives", []),
                "matched_volume_threshold": rule.get("max_volume"),
                "rule_version": version,
            }
    # Fallback if no rule matched
    return {
        "primary": "cnc_machining",
        "alternatives": ["additive"],
        "matched_volume_threshold": None,
        "rule_version": version,
    }


__all__ = ["recommend", "load_rules"]
