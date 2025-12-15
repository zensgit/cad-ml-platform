"""
JSON-based drawing difference and similarity utilities.

Provides a minimal, pluggable diff algorithm that compares two
standardized drawing JSONs and produces:
- diff structure: added/removed/modified per top-level section
- similarity score in [0,1]

This module is intentionally simple and dependency-free so it can be
integrated quickly into the layered retrieval verification step. It can
be enhanced later with geometry-aware comparison and weights.
"""

from __future__ import annotations

import fnmatch
from typing import Any, Dict, List, Optional, Tuple


def _normalize_value(v: Any, case_insensitive: bool) -> Any:
    # Try to normalize numeric strings to numbers
    if isinstance(v, str):
        s = v.strip()
        if case_insensitive:
            s_norm = s.lower()
        else:
            s_norm = s
        try:
            if "." in s_norm:
                return float(s_norm)
            return int(s_norm)
        except Exception:
            return s_norm
    return v


def _normalize_structure(
    obj: Any,
    path: str,
    *,
    list_path_modes: Optional[Dict[str, str]] = None,
    case_insensitive: bool = False,
) -> Any:
    """Apply path-specific normalization before flattening.

    - list_path_modes supports:
      - "index" (default): keep original order and index addressing
      - "unordered": sort list elements to make order-insensitive
      - "key:<field>": for list[dict], convert to dict keyed by <field>
    """
    modes = list_path_modes or {}

    def _match_mode(p: str) -> Optional[str]:
        # support exact, suffix, and glob-style matching
        best: Tuple[int, Optional[str]] = (0, None)
        for k, v in modes.items():
            if p == k or p.endswith("." + k) or fnmatch.fnmatch(p, k):
                l = len(k)
                if l >= best[0]:
                    best = (l, v)
        return best[1]

    def _val_key(v: Any) -> Any:
        if isinstance(v, dict):
            # key on sorted items
            return tuple((str(k), _val_key(v[k])) for k in sorted(v.keys()))
        if isinstance(v, list):
            return tuple(_val_key(x) for x in v)
        if isinstance(v, str) and case_insensitive:
            return v.lower()
        return v

    if isinstance(obj, list):
        mode = _match_mode(path) or "index"
        if mode == "unordered":
            try:
                return sorted(
                    [
                        _normalize_structure(
                            v,
                            f"{path}[]",
                            list_path_modes=list_path_modes,
                            case_insensitive=case_insensitive,
                        )
                        for v in obj
                    ],
                    key=_val_key,
                )
            except Exception:
                # fallback: keep as-is on error
                return obj
        if mode.startswith("key:"):
            key_field = mode.split(":", 1)[1] or "id"
            out: Dict[str, Any] = {}
            for i, item in enumerate(obj):
                if isinstance(item, dict):
                    k = item.get(key_field)
                    sk = str(k) if k is not None else f"_{i}"
                    out[sk] = _normalize_structure(
                        item,
                        f"{path}[{sk}]",
                        list_path_modes=list_path_modes,
                        case_insensitive=case_insensitive,
                    )
                else:
                    # non-dict elements: fall back to index addressing
                    out[str(i)] = _normalize_structure(
                        item,
                        f"{path}[{i}]",
                        list_path_modes=list_path_modes,
                        case_insensitive=case_insensitive,
                    )
            return out
        # index mode: normalize each element in place
        return [
            _normalize_structure(
                v,
                f"{path}[{i}]",
                list_path_modes=list_path_modes,
                case_insensitive=case_insensitive,
            )
            for i, v in enumerate(obj)
        ]
    if isinstance(obj, dict):
        return {
            str(k): _normalize_structure(
                v,
                f"{path}.{k}" if path else str(k),
                list_path_modes=list_path_modes,
                case_insensitive=case_insensitive,
            )
            for k, v in obj.items()
        }
    # primitives
    return obj


def _flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dict/list JSON into a path->value mapping.

    - dict keys are joined by "."
    - list items are addressed by "[i]"
    - values are compared as-is (primitive types recommended)
    """
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]" if prefix else f"[{i}]"
            out.update(_flatten(v, key))
    else:
        out[prefix] = obj
    return out


def compare_json(
    left: Dict[str, Any],
    right: Dict[str, Any],
    *,
    tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
    case_insensitive: bool = False,
    key_weights: Optional[Dict[str, float]] = None,
    weight_prefixes: Optional[Dict[str, float]] = None,
    ignore_paths: Optional[List[str]] = None,
    list_path_modes: Optional[Dict[str, str]] = None,
    max_diff_paths: Optional[int] = None,
    min_similarity: Optional[float] = None,
) -> Tuple[Dict[str, Any], float]:
    """Compute a simple diff and similarity between two JSON dicts.

    Returns a tuple of (diff, similarity):
      - diff: {
          "added": [paths],
          "removed": [paths],
          "modified": [{"path": str, "left": Any, "right": Any}],
          "unchanged": [paths]
        }
      - similarity: 1 - (#changes / #total_paths)

    Notes:
      - When both sides lack paths, similarity defaults to 1.0
      - This is a generic algorithm; geometry-aware weighting can be
        added later.
    """
    # Apply structural normalization prior to flattening (list modes)
    left_norm = _normalize_structure(
        left, "", list_path_modes=list_path_modes, case_insensitive=case_insensitive
    )
    right_norm = _normalize_structure(
        right, "", list_path_modes=list_path_modes, case_insensitive=case_insensitive
    )

    left_flat = _flatten(left_norm)
    right_flat = _flatten(right_norm)

    left_keys = set(left_flat.keys())
    right_keys = set(right_flat.keys())

    # Helpers: weight resolution and ignore matching
    weights_exact = key_weights or {}
    weights_prefix = weight_prefixes or {}
    ignore = ignore_paths or []

    def _is_ignored(k: str) -> bool:
        for patt in ignore:
            if fnmatch.fnmatch(k, patt) or k.endswith("." + patt) or k == patt:
                return True
        # weight <= 0 also treated as ignored
        return False

    def w(k: str) -> float:
        if _is_ignored(k):
            return 0.0
        if k in weights_exact:
            return float(weights_exact[k])
        # choose the longest matching prefix/pattern
        best_len = -1
        best_val: Optional[float] = None
        for patt, val in weights_prefix.items():
            if fnmatch.fnmatch(k, patt) or k.startswith(patt) or k.endswith("." + patt):
                l = len(patt)
                if l > best_len:
                    best_len = l
                    best_val = float(val)
        return float(best_val) if best_val is not None else 1.0

    added_all = list(right_keys - left_keys)
    removed_all = list(left_keys - right_keys)
    common_all = list(left_keys & right_keys)

    # Filter out ignored (zero-weight) keys from each bucket
    added = sorted([k for k in added_all if w(k) > 0.0])
    removed = sorted([k for k in removed_all if w(k) > 0.0])
    common = sorted([k for k in common_all if w(k) > 0.0])

    modified: List[Dict[str, Any]] = []
    unchanged: List[str] = []
    # Precompute totals for early-exit: totals depend only on weights and presence
    totals = sum(w(k) for k in added) + sum(w(k) for k in removed) + sum(w(k) for k in common)
    # current changes start with added+removed
    current_changes = sum(w(k) for k in added) + sum(w(k) for k in removed)

    early_exit_triggered = False

    for k in common:
        lv, rv = left_flat[k], right_flat[k]
        # Normalize for tolerant comparison
        ln = _normalize_value(lv, case_insensitive)
        rn = _normalize_value(rv, case_insensitive)
        equal = False
        try:
            if isinstance(ln, (int, float)) and isinstance(rn, (int, float)):
                diff = abs(float(ln) - float(rn))
                # absolute or relative tolerance
                rel_ok = False
                if relative_tolerance and relative_tolerance > 0.0:
                    base = max(abs(float(ln)), abs(float(rn)), 1e-12)
                    rel_ok = (diff / base) <= float(relative_tolerance)
                equal = (diff <= tolerance) or rel_ok
            else:
                equal = ln == rn
        except Exception:
            equal = lv == rv

        if equal:
            unchanged.append(k)
        else:
            modified.append({"path": k, "left": lv, "right": rv})
            current_changes += w(k)

        # Early-exit if even best-case can't reach min_similarity
        if min_similarity is not None and totals > 0:
            sim_upper = max(0.0, 1.0 - current_changes / float(totals))
            if sim_upper < float(min_similarity):
                early_exit_triggered = True
                break

    total_paths = len(added) + len(removed) + len(modified) + len(unchanged)
    if total_paths == 0:
        similarity = 1.0
    else:
        # Weighted changes/total (reuse w)
        changes = (
            sum(w(k) for k in added)
            + sum(w(k) for k in removed)
            + sum(w(m["path"]) for m in modified)
        )
        totals = (
            sum(w(k) for k in added)
            + sum(w(k) for k in removed)
            + sum(w(k) for k in unchanged)
            + sum(w(m["path"]) for m in modified)
        )
        similarity = 1.0 if totals <= 0 else max(0.0, 1.0 - changes / float(totals))

    # Apply diff truncation if needed
    diff = {
        "added": added,
        "removed": removed,
        "modified": modified,
        "unchanged": unchanged,
    }
    if max_diff_paths is not None and max_diff_paths > 0:

        def cap_list(lst, limit):
            return lst if len(lst) <= limit else lst[:limit]

        total_count = len(added) + len(removed) + len(modified)
        if total_count > max_diff_paths:
            # Roughly distribute the cap across categories
            per = max(1, max_diff_paths // 3)
            diff["added"] = cap_list(added, per)
            diff["removed"] = cap_list(removed, per)
            diff["modified"] = cap_list(
                modified, max_diff_paths - len(diff["added"]) - len(diff["removed"])
            )
            diff["meta"] = {
                "diff_truncated": True,
                "total_paths": total_count,
                "returned_paths": len(diff["added"]) + len(diff["removed"]) + len(diff["modified"]),
            }
    # annotate early-exit in meta
    if early_exit_triggered:
        meta = diff.setdefault("meta", {})
        meta["early_exit"] = True
        meta["early_exit_min_similarity"] = float(min_similarity or 0.0)
        meta["returned_paths"] = meta.get(
            "returned_paths",
            len(diff.get("added", []))
            + len(diff.get("removed", []))
            + len(diff.get("modified", [])),
        )
    return diff, similarity
