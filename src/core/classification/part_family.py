"""Normalize part-family predictions (coarse labels) into a stable API shape."""

from __future__ import annotations

from typing import Any, Dict


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float(default)
    if f != f:  # NaN
        return float(default)
    return float(f)


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def normalize_part_family_prediction(
    prediction: Any, *, provider_name: str
) -> Dict[str, Any]:
    """Normalize a provider prediction payload into stable part-family fields.

    The provider payload is expected to be a dict (for example the output of
    `classifier/v16` or `classifier/v6` providers).

    Returns a dict with keys:
    - part_family, part_family_confidence, part_family_source, part_family_model_version
    - part_family_needs_review, part_family_review_reason, part_family_top2
    - part_family_error (present only when not ok)
    """
    source = f"provider:{_safe_str(provider_name) or 'unknown'}"

    base: Dict[str, Any] = {
        "part_family": None,
        "part_family_confidence": None,
        "part_family_source": source,
        "part_family_model_version": None,
        "part_family_needs_review": None,
        "part_family_review_reason": None,
        "part_family_top2": None,
    }

    if not isinstance(prediction, dict):
        base["part_family_error"] = {
            "code": "invalid_payload",
            "message": f"Expected dict, got {type(prediction).__name__}",
        }
        return base

    status = _safe_str(prediction.get("status")).lower() or "unknown"
    if status != "ok":
        msg = _safe_str(prediction.get("error")) or status
        base["part_family_error"] = {"code": status, "message": msg}
        return base

    label = _safe_str(prediction.get("label"))
    if not label:
        base["part_family_error"] = {"code": "missing_label", "message": "Empty label"}
        return base

    conf = _clamp01(_safe_float(prediction.get("confidence"), 0.0))
    base["part_family"] = label
    base["part_family_confidence"] = conf

    model_version = prediction.get("model_version")
    if model_version is not None:
        base["part_family_model_version"] = _safe_str(model_version) or None

    if "needs_review" in prediction:
        try:
            base["part_family_needs_review"] = bool(prediction.get("needs_review"))
        except Exception:
            base["part_family_needs_review"] = None
    if "review_reason" in prediction:
        reason = _safe_str(prediction.get("review_reason"))
        base["part_family_review_reason"] = reason or None

    top2_label = _safe_str(prediction.get("top2_category"))
    if top2_label:
        top2_conf_raw = prediction.get("top2_confidence")
        if top2_conf_raw is not None:
            base["part_family_top2"] = {
                "label": top2_label,
                "confidence": _clamp01(_safe_float(top2_conf_raw, 0.0)),
            }

    return base
