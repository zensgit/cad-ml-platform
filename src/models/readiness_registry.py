"""Runtime model readiness registry.

The registry reports local model evidence without eagerly loading heavyweight
checkpoints. It is intentionally read-only and cheap after the first checksum
calculation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_CHECKSUM_CACHE: Dict[Tuple[str, int, int], str] = {}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _split_ids(raw: str) -> List[str]:
    return [item.strip() for item in re.split(r"[,\s]+", raw or "") if item.strip()]


def _path_list(paths: Iterable[Optional[str]]) -> List[str]:
    return [str(path).strip() for path in paths if str(path or "").strip()]


def _path_exists(path: str) -> bool:
    try:
        return Path(path).expanduser().exists()
    except OSError:
        return False


def _all_paths_exist(paths: Sequence[str]) -> bool:
    return bool(paths) and all(_path_exists(path) for path in paths)


def _checksum_file(path: str) -> Optional[str]:
    try:
        resolved = Path(path).expanduser()
        stat = resolved.stat()
        cache_key = (str(resolved), int(stat.st_mtime_ns), int(stat.st_size))
        cached = _CHECKSUM_CACHE.get(cache_key)
        if cached is not None:
            return cached
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()[:16]
        _CHECKSUM_CACHE[cache_key] = value
        return value
    except OSError:
        return None


def _combined_checksum(paths: Sequence[str]) -> tuple[Optional[str], Dict[str, str]]:
    checksums: Dict[str, str] = {}
    parts: List[str] = []
    for path in paths:
        checksum = _checksum_file(path)
        if checksum is None:
            continue
        checksums[path] = checksum
        parts.append(f"{path}:{checksum}")
    if not parts:
        return None, checksums
    if len(parts) == 1:
        return parts[0].rsplit(":", 1)[1], checksums
    combined = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return combined, checksums


def _torch_available() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except Exception:
        return False


def _module_object(module_name: str, attr_name: str) -> Any:
    module = sys.modules.get(module_name)
    if module is None:
        return None
    return getattr(module, attr_name, None)


@dataclass(frozen=True)
class ModelReadinessItem:
    name: str
    enabled: bool
    checkpoint_paths: List[str] = field(default_factory=list)
    checkpoint_exists: bool = False
    loaded: bool = False
    status: str = "disabled"
    version: Optional[str] = None
    checksum: Optional[str] = None
    fallback_mode: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "checkpoint_paths": list(self.checkpoint_paths),
            "checkpoint_exists": self.checkpoint_exists,
            "loaded": self.loaded,
            "status": self.status,
            "version": self.version,
            "checksum": self.checksum,
            "fallback_mode": self.fallback_mode,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ModelReadinessSnapshot:
    ok: bool
    degraded: bool
    status: str
    strict: bool
    required: List[str]
    generated_at: float
    items: List[ModelReadinessItem]
    degraded_reasons: List[str] = field(default_factory=list)
    blocking_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "degraded": self.degraded,
            "status": self.status,
            "strict": self.strict,
            "required": list(self.required),
            "generated_at": self.generated_at,
            "degraded_reasons": list(self.degraded_reasons),
            "blocking_reasons": list(self.blocking_reasons),
            "items": {item.name: item.to_dict() for item in self.items},
        }


def _status_from_evidence(
    *,
    enabled: bool,
    loaded: bool,
    checkpoint_exists: bool,
    checkpoint_required: bool,
    fallback_mode: Optional[str],
    error: Optional[str] = None,
) -> str:
    if not enabled:
        return "disabled"
    if error:
        return "error"
    if loaded:
        return "loaded"
    if checkpoint_exists:
        return "available"
    if fallback_mode:
        return "fallback"
    if checkpoint_required:
        return "missing"
    return "available"


def _item(
    *,
    name: str,
    enabled: bool,
    paths: Sequence[str],
    loaded: bool,
    checkpoint_required: bool,
    fallback_mode: Optional[str],
    version: Optional[str] = None,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ModelReadinessItem:
    checkpoint_exists = _all_paths_exist(paths)
    checksum, per_path_checksums = _combined_checksum(paths) if checkpoint_exists else (None, {})
    item_metadata = dict(metadata or {})
    if per_path_checksums:
        item_metadata["checkpoint_checksums"] = per_path_checksums
    status = _status_from_evidence(
        enabled=enabled,
        loaded=loaded,
        checkpoint_exists=checkpoint_exists,
        checkpoint_required=checkpoint_required,
        fallback_mode=fallback_mode,
        error=error,
    )
    return ModelReadinessItem(
        name=name,
        enabled=enabled,
        checkpoint_paths=list(paths),
        checkpoint_exists=checkpoint_exists,
        loaded=loaded,
        status=status,
        version=version,
        checksum=checksum,
        fallback_mode=fallback_mode if status == "fallback" else None,
        error=error,
        metadata=item_metadata,
    )


def _graph2d_enabled_default() -> bool:
    try:
        from src.ml.hybrid_config import get_config

        return bool(get_config().graph2d.enabled)
    except Exception:
        return True


def _v16_item() -> ModelReadinessItem:
    enabled = not _env_bool("DISABLE_V16_CLASSIFIER", False)
    v6_path = os.getenv("CAD_CLASSIFIER_MODEL", "models/cad_classifier_v6.pt")
    v14_path = os.getenv("V16_V14_MODEL_PATH", "models/cad_classifier_v14_ensemble.pt")
    classifier = _module_object("src.core.analyzer", "_v16_classifier")
    loaded = bool(classifier is not None and getattr(classifier, "loaded", False))
    error = _module_object("src.core.analyzer", "_v16_classifier_load_error")
    paths = _path_list([v6_path, v14_path])
    fallback = "v6_or_rule_based_classifier" if enabled and not _all_paths_exist(paths) else None
    return _item(
        name="v16_classifier",
        enabled=enabled,
        paths=paths,
        loaded=loaded,
        error=error,
        checkpoint_required=True,
        fallback_mode=fallback,
        version=os.getenv("V16_MODEL_VERSION", "v16"),
        metadata={
            "torch_available": _torch_available(),
            "speed_mode": getattr(classifier, "speed_mode", os.getenv("V16_SPEED_MODE", "fast")),
        },
    )


def _graph2d_item() -> ModelReadinessItem:
    enabled = _env_bool("GRAPH2D_ENABLED", _graph2d_enabled_default())
    path = os.getenv(
        "GRAPH2D_MODEL_PATH",
        "models/graph2d_training_dxf_oda_titleblock_distill_20260210.pth",
    )
    classifier = _module_object("src.ml.vision_2d", "_graph2d")
    loaded = bool(classifier is not None and getattr(classifier, "_loaded", False))
    error = getattr(classifier, "_load_error", None) if classifier is not None else None
    paths = _path_list([path])
    fallback = (
        "filename_titleblock_process_rules"
        if enabled and not _all_paths_exist(paths)
        else None
    )
    return _item(
        name="graph2d",
        enabled=enabled,
        paths=paths,
        loaded=loaded,
        error=error,
        checkpoint_required=True,
        fallback_mode=fallback,
        version=os.getenv("GRAPH2D_MODEL_VERSION"),
        metadata={"torch_available": _torch_available()},
    )


def _uvnet_item() -> ModelReadinessItem:
    enabled = _env_bool("UVNET_ENABLED", True)
    path = os.getenv("UVNET_MODEL_PATH", "models/uvnet_v1.pth")
    encoder = _module_object("src.ml.vision_3d", "_encoder")
    loaded = bool(encoder is not None and getattr(encoder, "_loaded", False))
    error = getattr(encoder, "_load_error", None) if encoder is not None else None
    paths = _path_list([path])
    fallback = "mock_brep_embedding" if enabled and not _all_paths_exist(paths) else None
    return _item(
        name="uvnet",
        enabled=enabled,
        paths=paths,
        loaded=loaded,
        error=error,
        checkpoint_required=True,
        fallback_mode=fallback,
        version=os.getenv("UVNET_MODEL_VERSION"),
        metadata={"torch_available": _torch_available()},
    )


def _pointnet_item() -> ModelReadinessItem:
    enabled = _env_bool("POINTNET_ENABLED", True)
    path = os.getenv("POINTNET_MODEL_PATH", "").strip()
    analyzer = _module_object("src.api.v1.pointcloud", "_analyzer")
    loaded = bool(analyzer is not None and getattr(analyzer, "_model_loaded", False))
    error = getattr(analyzer, "_load_error", None) if analyzer is not None else None
    paths = _path_list([path])
    fallback = (
        "statistical_pointcloud_features"
        if enabled and not _all_paths_exist(paths)
        else None
    )
    return _item(
        name="pointnet",
        enabled=enabled,
        paths=paths,
        loaded=loaded,
        error=error,
        checkpoint_required=True,
        fallback_mode=fallback,
        version=os.getenv("POINTNET_MODEL_VERSION"),
        metadata={"torch_available": _torch_available()},
    )


def _ocr_item() -> ModelReadinessItem:
    enabled = _env_bool("OCR_ENABLED", True)
    path = os.getenv("OCR_MODEL_PATH", "").strip()
    manager = _module_object("src.api.v1.ocr", "_manager")
    loaded = bool(manager is not None and getattr(manager, "providers", None))
    error = _module_object("src.api.v1.ocr", "_manager_load_error")
    paths = _path_list([path])
    provider = os.getenv("OCR_PROVIDER_DEFAULT", "paddle")
    fallback = "provider_managed" if enabled and not paths else None
    return _item(
        name="ocr_provider",
        enabled=enabled,
        paths=paths,
        loaded=loaded,
        error=error,
        checkpoint_required=bool(paths),
        fallback_mode=fallback,
        version=provider,
        metadata={"provider": provider},
    )


def _embedding_item() -> ModelReadinessItem:
    enabled = _env_bool("DOMAIN_EMBEDDINGS_ENABLED", True)
    path = os.getenv("DOMAIN_EMBEDDING_MODEL_PATH", "models/embeddings/manufacturing_v2")
    paths = _path_list([path])
    fallback = "tfidf_fallback" if enabled and not _all_paths_exist(paths) else None
    return _item(
        name="embedding_model",
        enabled=enabled,
        paths=paths,
        loaded=False,
        checkpoint_required=True,
        fallback_mode=fallback,
        version=os.getenv("DOMAIN_EMBEDDING_MODEL_VERSION"),
        metadata={"base_model": "paraphrase-multilingual-MiniLM-L12-v2"},
    )


def build_model_readiness_snapshot() -> ModelReadinessSnapshot:
    items = [
        _v16_item(),
        _graph2d_item(),
        _uvnet_item(),
        _pointnet_item(),
        _ocr_item(),
        _embedding_item(),
    ]
    required = _split_ids(os.getenv("MODEL_READINESS_REQUIRED_MODELS", ""))
    strict = _env_bool("MODEL_READINESS_STRICT", False)

    degraded_reasons: List[str] = []
    blocking_reasons: List[str] = []
    required_set = set(required)
    degraded_statuses = {"fallback", "missing", "error"}
    ready_statuses = {"loaded", "available", "disabled"}

    for item in items:
        if item.enabled and item.status in degraded_statuses:
            reason = f"{item.name}:{item.status}"
            if item.fallback_mode:
                reason = f"{reason}:{item.fallback_mode}"
            degraded_reasons.append(reason)
        if item.name in required_set and item.status not in {"loaded", "available"}:
            blocking_reasons.append(f"{item.name}:{item.status}")
        elif strict and item.enabled and item.status not in ready_statuses:
            blocking_reasons.append(f"{item.name}:{item.status}")

    degraded = bool(degraded_reasons)
    ok = not blocking_reasons
    status = "ready"
    if not ok:
        status = "not_ready"
    elif degraded:
        status = "degraded"

    return ModelReadinessSnapshot(
        ok=ok,
        degraded=degraded,
        status=status,
        strict=strict,
        required=required,
        generated_at=time.time(),
        items=items,
        degraded_reasons=degraded_reasons,
        blocking_reasons=blocking_reasons,
    )


def model_readiness_summary_for_check() -> Dict[str, Any]:
    snapshot = build_model_readiness_snapshot()
    details = []
    if snapshot.blocking_reasons:
        details.append("blocking=" + ",".join(snapshot.blocking_reasons))
    if snapshot.degraded_reasons:
        details.append("degraded=" + ",".join(snapshot.degraded_reasons))
    return {
        "ok": snapshot.ok,
        "degraded": snapshot.degraded,
        "detail": "; ".join(details) if details else None,
        "snapshot": snapshot.to_dict(),
    }


__all__ = [
    "ModelReadinessItem",
    "ModelReadinessSnapshot",
    "build_model_readiness_snapshot",
    "model_readiness_summary_for_check",
]
