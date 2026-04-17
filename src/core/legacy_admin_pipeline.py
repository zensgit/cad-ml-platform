from __future__ import annotations

import hashlib
import os
from typing import Any, Callable, Optional


def run_process_rules_audit_pipeline(
    *,
    raw: bool,
    load_rules_fn: Callable[..., dict[str, Any]],
    rules_path: str,
    path_exists: Callable[[str], bool] = os.path.exists,
    file_opener: Callable[..., Any] = open,
) -> dict[str, Any]:
    rules = load_rules_fn(force_reload=True)
    version = rules.get("__meta__", {}).get("version", "v1")
    materials = sorted([material for material in rules.keys() if not material.startswith("__")])
    complexities: dict[str, list[str]] = {}
    for material in materials:
        candidate = rules.get(material, {})
        if isinstance(candidate, dict):
            complexities[material] = sorted(
                [name for name in candidate.keys() if isinstance(candidate.get(name), list)]
            )

    file_hash: Optional[str] = None
    try:
        if path_exists(rules_path):
            with file_opener(rules_path, "rb") as handle:
                file_hash = hashlib.sha256(handle.read()).hexdigest()[:16]
    except Exception:
        file_hash = None

    return {
        "version": version,
        "source": rules_path if path_exists(rules_path) else "embedded-defaults",
        "hash": file_hash,
        "materials": materials,
        "complexities": complexities,
        "raw": rules if raw else {},
    }


def run_faiss_rebuild_pipeline(
    *,
    vector_store_backend: str,
    store_factory: Callable[[], Any],
) -> dict[str, Any]:
    if vector_store_backend != "faiss":
        return {"rebuilt": False, "reason": "backend_not_faiss"}

    store = store_factory()
    rebuilt = bool(store.rebuild())  # type: ignore[attr-defined]
    return {
        "rebuilt": rebuilt,
        "message": "Index rebuilt successfully" if rebuilt else "Rebuild failed",
    }


__all__ = ["run_process_rules_audit_pipeline", "run_faiss_rebuild_pipeline"]
