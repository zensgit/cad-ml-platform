"""Precision pass after recall — honest geometric verification.

Does **not** invent geometric scores from visual similarity.
When both query and candidate geom-json payloads are present, scores with
``PrecisionVerifier`` (L4). Otherwise labels ``vision_only_unverified`` or
``missing_geom_json``. Injectable hook is for tests / DI only.

Does not call training paths, hosted LLMs, or eval_integrity_gate.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from .models import CandidateDecision, CandidateState, RejectionReason

logger = logging.getLogger(__name__)

LOW_PRECISION_THRESHOLD = 0.55

PrecisionFn = Callable[[str, bytes, List[CandidateDecision]], List[CandidateDecision]]
_PRECISION: Optional[PrecisionFn] = None


def set_precision_hook(fn: Optional[PrecisionFn]) -> None:
    """Register a process-local precision function (tests / DI)."""
    global _PRECISION
    _PRECISION = fn


def get_precision_hook() -> Optional[PrecisionFn]:
    return _PRECISION


def _append_reason(candidate: CandidateDecision, reason: str) -> None:
    if reason not in candidate.rejection_reasons:
        candidate.rejection_reasons = list(candidate.rejection_reasons) + [reason]


def _methods(candidate: CandidateDecision) -> List[str]:
    raw = candidate.verification.get("methods") if candidate.verification else None
    if isinstance(raw, list):
        return [str(m) for m in raw]
    return []


def _has_method(candidate: CandidateDecision, method: str) -> bool:
    return method in _methods(candidate)


def _is_live_vision(candidate: CandidateDecision) -> bool:
    model = str((candidate.provenance or {}).get("model") or "")
    return "dedup2d-vision" in model or _has_method(candidate, "dedup2d-vision")


def _parse_query_geom(file_bytes: bytes) -> Optional[Dict[str, Any]]:
    if not file_bytes:
        return None
    stripped = file_bytes.lstrip()
    if not stripped.startswith(b"{") and not stripped.startswith(b"["):
        return None
    try:
        obj = json.loads(file_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _try_l4_score(
    query_geom: Dict[str, Any], candidate: CandidateDecision
) -> Optional[float]:
    right = (candidate.provenance or {}).get("geom_json")
    if not isinstance(right, dict):
        return None
    try:
        from src.core.dedupcad_precision import PrecisionVerifier

        scored = PrecisionVerifier().score_pair(query_geom, right)
        return float(scored.score)
    except Exception:
        logger.warning("review_reuse_l4_score_failed", exc_info=True)
        return None


def _apply_l4_score(candidate: CandidateDecision, score: float) -> None:
    candidate.scores = dict(candidate.scores)
    candidate.scores["geometric"] = score
    methods = _methods(candidate)
    if "precision-l4" not in methods:
        methods.append("precision-l4")
    verification = dict(candidate.verification or {})
    verification["methods"] = methods
    verification["verdict"] = verification.get("verdict") or candidate.state.value
    candidate.verification = verification
    if score < LOW_PRECISION_THRESHOLD:
        _append_reason(candidate, RejectionReason.low_precision_score.value)


def apply_precision(
    candidates: List[CandidateDecision],
    *,
    file_name: str,
    file_bytes: bytes,
) -> List[CandidateDecision]:
    """Run the precision stage. Hook wins; otherwise honest default labeling."""
    hook = get_precision_hook()
    if hook is not None:
        return list(hook(file_name, file_bytes, list(candidates)))

    query_geom = _parse_query_geom(file_bytes)
    out: List[CandidateDecision] = []
    for original in candidates:
        candidate = original.model_copy(deep=True)
        if candidate.state == CandidateState.insufficient_evidence:
            out.append(candidate)
            continue

        l4 = _try_l4_score(query_geom, candidate) if query_geom else None
        if l4 is not None:
            _apply_l4_score(candidate, l4)
            out.append(candidate)
            continue

        geometric = candidate.scores.get("geometric")
        has_numeric_geom = isinstance(geometric, (int, float))
        if has_numeric_geom and _has_method(candidate, "precision-l4"):
            if float(geometric) < LOW_PRECISION_THRESHOLD:
                _append_reason(candidate, RejectionReason.low_precision_score.value)
            out.append(candidate)
            continue

        if _is_live_vision(candidate) and not has_numeric_geom:
            _append_reason(candidate, RejectionReason.vision_only_unverified.value)
            out.append(candidate)
            continue

        if not has_numeric_geom:
            _append_reason(candidate, RejectionReason.missing_geom_json.value)
        elif float(geometric) < LOW_PRECISION_THRESHOLD:
            _append_reason(candidate, RejectionReason.low_precision_score.value)
        out.append(candidate)
    return out
