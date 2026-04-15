"""Helpers for collecting shadow classification evidence in analyze flows."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time as _t_perf
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.core.classification.part_family import normalize_part_family_prediction
from src.utils.analysis_metrics import (
    analysis_hybrid_rejections_total,
    analysis_part_classifier_requests_total,
    analysis_part_classifier_seconds,
    analysis_part_classifier_skipped_total,
)

logger = logging.getLogger(__name__)

DEFAULT_GRAPH2D_DRAWING_LABELS = {
    "零件图",
    "机械制图",
    "装配图",
    "练习零件图",
    "原理图",
    "模板",
}

DEFAULT_GRAPH2D_COARSE_LABELS = {
    "传动件",
    "壳体类",
    "轴类",
    "连接件",
    "其他",
}


def _safe_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%s; using default %.2f", name, raw, default)
        return float(default)


def _graph2d_is_drawing_type(label: Optional[str]) -> bool:
    if not label:
        return False
    raw = os.getenv("GRAPH2D_DRAWING_TYPE_LABELS", "").strip()
    if raw:
        labels = {item.strip() for item in raw.split(",") if item.strip()}
    else:
        labels = DEFAULT_GRAPH2D_DRAWING_LABELS
    return label.strip() in labels


def _graph2d_is_coarse_label(label: Optional[str]) -> bool:
    if not label:
        return False
    raw = os.getenv("GRAPH2D_COARSE_LABELS", "").strip()
    if raw:
        labels = {item.strip() for item in raw.split(",") if item.strip()}
    else:
        labels = DEFAULT_GRAPH2D_COARSE_LABELS
    return label.strip() in labels


def _enrich_graph2d_prediction(
    graph2d_result: Dict[str, Any],
    *,
    graph2d_ensemble_enabled: bool,
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    graph2d_min_conf = _safe_float_env("GRAPH2D_MIN_CONF", 0.0)
    if "GRAPH2D_MIN_CONF" not in os.environ:
        try:
            from src.ml.hybrid_config import get_config

            graph2d_min_conf = float(get_config().graph2d.min_confidence)
        except Exception:
            pass
    graph2d_min_margin = _safe_float_env("GRAPH2D_MIN_MARGIN", 0.0)
    if "GRAPH2D_MIN_MARGIN" not in os.environ:
        try:
            from src.ml.hybrid_config import get_config

            graph2d_min_margin = float(getattr(get_config().graph2d, "min_margin", 0.0))
        except Exception:
            pass

    graph2d_result["min_confidence"] = graph2d_min_conf
    graph2d_result["min_margin"] = graph2d_min_margin
    graph2d_result["ensemble_enabled"] = graph2d_ensemble_enabled

    if graph2d_result.get("status") == "model_unavailable":
        return graph2d_result, None

    graph2d_conf = float(graph2d_result.get("confidence", 0.0))
    graph2d_margin_raw = None
    try:
        if graph2d_result.get("margin") is not None:
            graph2d_margin_raw = float(graph2d_result.get("margin"))
    except Exception:
        graph2d_margin_raw = None

    graph2d_passed_margin = True
    if graph2d_margin_raw is not None:
        graph2d_passed_margin = graph2d_margin_raw >= graph2d_min_margin

    graph2d_allow_raw = os.getenv("GRAPH2D_ALLOW_LABELS", "").strip()
    graph2d_exclude_raw = os.getenv("GRAPH2D_EXCLUDE_LABELS", "").strip()
    if not graph2d_allow_raw or not graph2d_exclude_raw:
        try:
            from src.ml.hybrid_config import get_config

            cfg = get_config()
            if not graph2d_allow_raw:
                graph2d_allow_raw = str(cfg.graph2d.allow_labels or "").strip()
            if not graph2d_exclude_raw:
                graph2d_exclude_raw = str(cfg.graph2d.exclude_labels or "").strip()
        except Exception:
            pass

    if not graph2d_exclude_raw:
        graph2d_exclude_raw = "other"

    graph2d_allow = {
        label.strip() for label in graph2d_allow_raw.split(",") if label.strip()
    }
    graph2d_exclude = {
        label.strip() for label in graph2d_exclude_raw.split(",") if label.strip()
    }
    graph2d_label = str(graph2d_result.get("label") or "").strip()
    graph2d_is_drawing_type = _graph2d_is_drawing_type(graph2d_label)
    graph2d_is_coarse_label = _graph2d_is_coarse_label(graph2d_label)
    graph2d_allowed = not graph2d_allow or graph2d_label in graph2d_allow

    graph2d_result["passed_threshold"] = graph2d_conf >= graph2d_min_conf
    graph2d_result["passed_margin"] = graph2d_passed_margin
    graph2d_result["excluded"] = graph2d_label in graph2d_exclude
    graph2d_result["allowed"] = graph2d_allowed
    graph2d_result["is_drawing_type"] = graph2d_is_drawing_type
    graph2d_result["is_coarse_label"] = graph2d_is_coarse_label

    graph2d_fusable = None
    if (
        graph2d_result["passed_threshold"]
        and graph2d_result.get("passed_margin", True)
        and graph2d_allowed
        and not graph2d_result["excluded"]
        and not graph2d_is_drawing_type
    ):
        graph2d_fusable = graph2d_result

    return graph2d_result, graph2d_fusable


def _build_graph2d_soft_override_suggestion(
    *,
    graph2d_result: Optional[Dict[str, Any]],
    cls_payload: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if graph2d_result is None:
        return None

    graph2d_min_conf_default = 0.17
    try:
        if graph2d_result.get("min_confidence") is not None:
            graph2d_min_conf_default = float(graph2d_result.get("min_confidence"))
    except Exception:
        graph2d_min_conf_default = 0.17

    soft_override_min_conf = _safe_float_env(
        "GRAPH2D_SOFT_OVERRIDE_MIN_CONF", graph2d_min_conf_default
    )

    if graph2d_result.get("status") == "model_unavailable":
        return {
            "eligible": False,
            "threshold": soft_override_min_conf,
            "label": None,
            "confidence": float(graph2d_result.get("confidence", 0.0) or 0.0),
            "reason": "graph2d_unavailable",
        }

    graph2d_label = str(graph2d_result.get("label") or "").strip()
    graph2d_conf = float(graph2d_result.get("confidence", 0.0))
    graph2d_allowed = bool(graph2d_result.get("allowed", True))
    graph2d_excluded = bool(graph2d_result.get("excluded", False))
    graph2d_passed_margin = bool(graph2d_result.get("passed_margin", True))
    graph2d_min_margin = float(graph2d_result.get("min_margin", 0.0) or 0.0)
    graph2d_is_drawing_type = bool(graph2d_result.get("is_drawing_type", False))
    graph2d_is_coarse_label = bool(graph2d_result.get("is_coarse_label", False))

    eligible = True
    reason = "eligible"
    if cls_payload.get("confidence_source") != "rules":
        eligible = False
        reason = "confidence_source_not_rules"
    elif str(cls_payload.get("rule_version") or "") != "v1":
        eligible = False
        reason = "rule_version_not_v1"
    elif graph2d_excluded:
        eligible = False
        reason = "graph2d_excluded"
    elif not graph2d_allowed:
        eligible = False
        reason = "graph2d_not_allowed"
    elif graph2d_is_drawing_type:
        eligible = False
        reason = "graph2d_drawing_type"
    elif graph2d_is_coarse_label:
        eligible = False
        reason = "graph2d_coarse_label"
    elif not graph2d_passed_margin:
        eligible = False
        reason = "below_margin"
    elif graph2d_conf < soft_override_min_conf:
        eligible = False
        reason = "below_threshold"

    return {
        "eligible": eligible,
        "threshold": soft_override_min_conf,
        "min_margin": graph2d_min_margin,
        "passed_margin": graph2d_passed_margin,
        "label": graph2d_label,
        "confidence": graph2d_conf,
        "reason": reason,
    }


def _resolve_history_sequence_file_path(
    *,
    file_name: str,
    file_format: str,
    analysis_options: Any,
) -> tuple[Optional[str], Optional[str]]:
    allowed_root: Optional[Path] = None
    allowed_root_raw = os.getenv("HISTORY_SEQUENCE_ALLOWED_ROOT", "").strip()
    if allowed_root_raw:
        root = Path(allowed_root_raw).expanduser()
        if root.exists() and root.is_dir():
            allowed_root = root.resolve()
        else:
            logger.warning("HISTORY_SEQUENCE_ALLOWED_ROOT is invalid: %s", root)

    def _resolve_existing_h5(path_raw: str) -> Optional[Path]:
        candidate = Path(path_raw).expanduser()
        try:
            resolved = candidate.resolve(strict=True)
        except Exception:
            return None
        if not resolved.is_file() or resolved.suffix.lower() != ".h5":
            return None
        if allowed_root is not None:
            try:
                resolved.relative_to(allowed_root)
            except ValueError:
                return None
        return resolved

    explicit = str(getattr(analysis_options, "history_file_path", "") or "").strip()
    if explicit:
        path = _resolve_existing_h5(explicit)
        if path is not None:
            return str(path), "options"
        logger.warning("history_file_path from options is invalid: %s", explicit)

    env_path_raw = os.getenv("HISTORY_SEQUENCE_FILE_PATH", "").strip()
    if env_path_raw:
        env_path = _resolve_existing_h5(env_path_raw)
        if env_path is not None:
            return str(env_path), "env"
        logger.warning("HISTORY_SEQUENCE_FILE_PATH is invalid: %s", env_path_raw)

    if str(file_format or "").strip().lower() not in {"dxf", "dwg"}:
        return None, None

    sidecar_dir_raw = os.getenv("HISTORY_SEQUENCE_SIDECAR_DIR", "").strip()
    if not sidecar_dir_raw:
        return None, None
    sidecar_dir = Path(sidecar_dir_raw).expanduser()
    if not sidecar_dir.exists() or not sidecar_dir.is_dir():
        logger.warning("HISTORY_SEQUENCE_SIDECAR_DIR is invalid: %s", sidecar_dir)
        return None, None

    file_name_clean = Path(str(file_name or "")).name
    stem = Path(file_name_clean).stem
    if not stem:
        return None, None

    exact = _resolve_existing_h5(str(sidecar_dir / f"{stem}.h5"))
    if exact is not None:
        return str(exact), "sidecar_exact"

    suffix = _resolve_existing_h5(str(sidecar_dir / f"{stem}_1.h5"))
    if suffix is not None:
        return str(suffix), "sidecar_suffix_1"

    allow_glob = os.getenv(
        "HISTORY_SEQUENCE_SIDECAR_GLOB_ENABLED", "true"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if allow_glob:
        try:
            matches = sorted(sidecar_dir.glob(f"{stem}*.h5"))
        except Exception:
            matches = []
        for candidate in matches:
            resolved = _resolve_existing_h5(str(candidate))
            if resolved is not None:
                return str(resolved), "sidecar_glob"

    return None, None


def _predict_ml_from_features(features: Mapping[str, Any]) -> Dict[str, Any]:
    from src.core.feature_extractor import FeatureExtractor
    from src.ml.classifier import predict

    vec_for_model = FeatureExtractor().flatten(features)
    return predict(vec_for_model)


def _get_classifier_provider(provider_name: str) -> Any:
    from src.core.providers import ProviderRegistry, bootstrap_core_provider_registry

    bootstrap_core_provider_registry()
    return ProviderRegistry.get("classifier", provider_name)


def _make_classifier_request(**kwargs: Any) -> Any:
    from src.core.providers.classifier import ClassifierRequest

    return ClassifierRequest(**kwargs)


def _apply_ml_overlay(
    payload: Optional[Dict[str, Any]],
    *,
    features: Mapping[str, Any],
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    cls_payload = dict(payload or {})
    ml_result: Optional[Dict[str, Any]] = None
    try:
        ml_result = _predict_ml_from_features(features)
        if ml_result.get("predicted_type"):
            cls_payload["ml_predicted_type"] = ml_result["predicted_type"]
            cls_payload["model_version"] = ml_result.get("model_version")
        else:
            cls_payload["model_version"] = ml_result.get("status")
    except Exception:
        ml_result = None
        cls_payload["model_version"] = "ml_error"
    return cls_payload, ml_result


async def _run_graph2d_shadow(
    payload: Optional[Dict[str, Any]],
    *,
    file_name: Optional[str],
    file_format: str,
    content: bytes,
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    cls_payload = dict(payload or {})
    graph2d_result: Optional[Dict[str, Any]] = None
    graph2d_fusable: Optional[Dict[str, Any]] = None
    graph2d_enabled = os.getenv("GRAPH2D_ENABLED", "false").lower() == "true"
    if not (graph2d_enabled and file_format == "dxf"):
        return cls_payload, graph2d_result, graph2d_fusable

    try:
        graph2d_ensemble_enabled = (
            os.getenv("GRAPH2D_ENSEMBLE_ENABLED", "false").lower() == "true"
        )
        provider_name = "graph2d_ensemble" if graph2d_ensemble_enabled else "graph2d"
        provider = _get_classifier_provider(provider_name)
        graph2d_result = await provider.process(
            _make_classifier_request(
                filename=file_name,
                file_bytes=content,
            )
        )
        if isinstance(graph2d_result, dict):
            graph2d_result, graph2d_fusable = _enrich_graph2d_prediction(
                graph2d_result,
                graph2d_ensemble_enabled=graph2d_ensemble_enabled,
            )
            cls_payload["graph2d_prediction"] = graph2d_result
    except Exception:
        graph2d_result = None

    return cls_payload, graph2d_result, graph2d_fusable


async def _run_hybrid_shadow(
    payload: Optional[Dict[str, Any]],
    *,
    file_name: Optional[str],
    file_format: str,
    content: bytes,
    analysis_options: Any,
    graph2d_result: Optional[Dict[str, Any]],
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    cls_payload = dict(payload or {})
    hybrid_result: Optional[Dict[str, Any]] = None
    hybrid_enabled = os.getenv("HYBRID_CLASSIFIER_ENABLED", "true").lower() == "true"
    if not (hybrid_enabled and file_format == "dxf"):
        return cls_payload, hybrid_result

    try:
        history_file_path, history_file_path_source = _resolve_history_sequence_file_path(
            file_name=str(file_name or ""),
            file_format=file_format,
            analysis_options=analysis_options,
        )
        cls_payload["history_sequence_input"] = {
            "resolved": bool(history_file_path),
            "source": history_file_path_source,
            "file_name": Path(history_file_path).name if history_file_path else None,
        }

        provider = _get_classifier_provider("hybrid")
        hybrid_result = await provider.process(
            _make_classifier_request(
                filename=file_name,
                file_bytes=content,
                history_file_path=history_file_path,
            ),
            graph2d_result=graph2d_result,
        )
        cls_payload["filename_prediction"] = hybrid_result.get("filename_prediction")
        cls_payload["titleblock_prediction"] = hybrid_result.get("titleblock_prediction")
        cls_payload["history_prediction"] = hybrid_result.get("history_prediction")
        cls_payload["process_prediction"] = hybrid_result.get("process_prediction")
        cls_payload["decision_path"] = hybrid_result.get("decision_path")
        cls_payload["source_contributions"] = hybrid_result.get("source_contributions")
        cls_payload["fusion_metadata"] = hybrid_result.get("fusion_metadata")
        cls_payload["hybrid_explanation"] = hybrid_result.get("explanation")
        cls_payload["hybrid_decision"] = hybrid_result

        hybrid_rejection = hybrid_result.get("rejection")
        if isinstance(hybrid_rejection, dict):
            cls_payload["hybrid_rejection"] = hybrid_rejection
            cls_payload["hybrid_rejected"] = True
            rejection_reason = (
                str(hybrid_rejection.get("reason") or "unknown").strip() or "unknown"
            )
            rejection_source = (
                str(
                    hybrid_rejection.get("raw_source")
                    or hybrid_result.get("source")
                    or "unknown"
                ).strip()
                or "unknown"
            )
            analysis_hybrid_rejections_total.labels(
                reason=rejection_reason,
                raw_source=rejection_source,
            ).inc()
        else:
            cls_payload["hybrid_rejected"] = False

        hybrid_label = str(hybrid_result.get("label") or "").strip()
        if hybrid_label:
            cls_payload["fine_part_type"] = hybrid_label
            cls_payload["fine_confidence"] = float(
                hybrid_result.get("confidence", 0.0) or 0.0
            )
            cls_payload["fine_source"] = hybrid_result.get("source")
            cls_payload["fine_rule_version"] = "HybridClassifier-v1"
    except Exception as exc:
        cls_payload["hybrid_error"] = str(exc)

    return cls_payload, hybrid_result


async def _run_part_classifier_shadow(
    payload: Optional[Dict[str, Any]],
    *,
    file_name: Optional[str],
    file_format: str,
    content: bytes,
) -> Dict[str, Any]:
    cls_payload = dict(payload or {})
    part_provider_enabled = (
        os.getenv("PART_CLASSIFIER_PROVIDER_ENABLED", "false").lower() == "true"
    )
    if not part_provider_enabled:
        return cls_payload

    shadow_formats_raw = os.getenv(
        "PART_CLASSIFIER_PROVIDER_SHADOW_FORMATS", "dxf,dwg"
    ).strip()
    shadow_formats = {
        token.strip().lower() for token in shadow_formats_raw.split(",") if token.strip()
    }
    if file_format not in shadow_formats:
        analysis_part_classifier_skipped_total.labels(
            reason="format_not_supported"
        ).inc()
        return cls_payload

    provider_name = os.getenv("PART_CLASSIFIER_PROVIDER_NAME", "v16").strip() or "v16"
    timeout_seconds = _safe_float_env("PART_CLASSIFIER_PROVIDER_TIMEOUT_SECONDS", 2.0)
    max_mb = _safe_float_env("PART_CLASSIFIER_PROVIDER_MAX_MB", 10.0)
    size_mb = len(content) / (1024 * 1024)
    tmp_path: Optional[str] = None
    status_label = "error"
    part_result: Dict[str, Any]
    started_at = None

    try:
        started_at = _t_perf.perf_counter()
        if size_mb > max_mb:
            status_label = "skipped"
            part_result = {
                "status": "file_too_large",
                "error": "skipped due to file size",
                "max_mb": float(max_mb),
                "size_mb": round(size_mb, 3),
            }
            analysis_part_classifier_skipped_total.labels(reason="file_too_large").inc()
        else:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{file_format}"
            ) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            provider = _get_classifier_provider(provider_name)
            try:
                timeout = float(min(max(timeout_seconds, 0.01), 10.0))
                part_result = await asyncio.wait_for(
                    provider.process(
                        _make_classifier_request(
                            filename=file_name,
                            file_path=tmp_path,
                        )
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                status_label = "timeout"
                part_result = {
                    "status": "timeout",
                    "error": f"timeout after {timeout_seconds:.2f}s",
                }
            except Exception as exc:  # noqa: BLE001
                status_label = "error"
                part_result = {
                    "status": "error",
                    "error": str(exc),
                }

            raw_status = str(part_result.get("status") or "").strip().lower()
            if raw_status == "ok":
                status_label = "success"
            elif raw_status in {"timeout"}:
                status_label = "timeout"
            elif raw_status in {"unavailable", "no_prediction", "model_unavailable"}:
                status_label = "unavailable"
            elif raw_status == "file_too_large":
                status_label = "skipped"
            else:
                status_label = "error"

        if isinstance(part_result, dict):
            part_result.setdefault("provider", provider_name)
        cls_payload["part_classifier_prediction"] = part_result
        cls_payload.update(
            normalize_part_family_prediction(part_result, provider_name=provider_name)
        )
    except Exception as exc:  # noqa: BLE001
        status_label = "error"
        part_result = {
            "status": "error",
            "error": str(exc),
            "provider": provider_name,
        }
        cls_payload["part_classifier_prediction"] = part_result
        cls_payload.update(
            normalize_part_family_prediction(part_result, provider_name=provider_name)
        )
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        if started_at is not None:
            try:
                elapsed = _t_perf.perf_counter() - started_at
                analysis_part_classifier_seconds.labels(provider=provider_name).observe(
                    elapsed
                )
            except Exception:
                pass
        try:
            analysis_part_classifier_requests_total.labels(
                status=status_label,
                provider=provider_name,
            ).inc()
        except Exception:
            pass

    return cls_payload


async def build_shadow_classification_context(
    payload: Optional[Dict[str, Any]],
    *,
    features: Mapping[str, Any],
    file_name: Optional[str],
    file_format: str,
    content: bytes,
    analysis_options: Any,
) -> Dict[str, Any]:
    cls_payload, ml_result = _apply_ml_overlay(payload, features=features)
    cls_payload, graph2d_result, graph2d_fusable = await _run_graph2d_shadow(
        cls_payload,
        file_name=file_name,
        file_format=file_format,
        content=content,
    )
    cls_payload, hybrid_result = await _run_hybrid_shadow(
        cls_payload,
        file_name=file_name,
        file_format=file_format,
        content=content,
        analysis_options=analysis_options,
        graph2d_result=graph2d_result,
    )
    cls_payload = await _run_part_classifier_shadow(
        cls_payload,
        file_name=file_name,
        file_format=file_format,
        content=content,
    )

    soft_override_suggestion = _build_graph2d_soft_override_suggestion(
        graph2d_result=graph2d_result,
        cls_payload=cls_payload,
    )
    if soft_override_suggestion is not None:
        cls_payload["soft_override_suggestion"] = soft_override_suggestion

    return {
        "payload": cls_payload,
        "ml_result": ml_result,
        "graph2d_result": graph2d_result,
        "graph2d_fusable": graph2d_fusable,
        "hybrid_result": hybrid_result,
    }


__all__ = [
    "_build_graph2d_soft_override_suggestion",
    "_enrich_graph2d_prediction",
    "_graph2d_is_drawing_type",
    "_resolve_history_sequence_file_path",
    "_safe_float_env",
    "build_shadow_classification_context",
]
