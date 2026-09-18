"""Precision pass after recall — honest geometric verification.

Does **not** invent geometric scores from visual similarity.
When both query and candidate geom-json payloads are present, scores with
``PrecisionVerifier`` (L4). Query geom comes from JSON uploads or local DXF
extract; candidate geom comes from hit ``geom_json`` or the geom store.
Otherwise labels ``vision_only_unverified`` or ``missing_geom_json``.
Injectable hook is for tests / DI only. DWG is not auto-converted.

Does not call training paths, hosted LLMs, or eval_integrity_gate.
"""

from __future__ import annotations

import json
import logging
import math
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .files import file_suffix
from .models import CandidateDecision, CandidateState, RejectionReason

logger = logging.getLogger(__name__)

LOW_PRECISION_THRESHOLD = 0.55
_PROVISIONAL_UNVERIFIED = frozenset(
    {
        RejectionReason.vision_only_unverified.value,
        RejectionReason.missing_geom_json.value,
    }
)

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


def _looks_like_file_hash(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _extract_dxf_geom(file_bytes: bytes) -> Optional[Dict[str, Any]]:
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)
        from src.core.dedupcad_precision.cad_pipeline import (
            extract_geom_json_from_dxf,
        )

        geom = extract_geom_json_from_dxf(tmp_path)
        return geom if _is_geom_json(geom) else None
    except Exception:
        logger.debug("review_reuse_dxf_geom_extract_failed", exc_info=True)
        return None
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


_GEOM_ENTITY_TYPES = frozenset(
    {
        "LINE",
        "CIRCLE",
        "ARC",
        "LWPOLYLINE",
        "POLYLINE",
        "ELLIPSE",
        "SPLINE",
        "INSERT",
    }
)


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _positive_number(value: Any) -> bool:
    if not _finite_number(value):
        return False
    return float(value) > 0.0


def _is_finite_unit_score(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return math.isfinite(number) and 0.0 <= number <= 1.0


def _xy(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return False
    return _finite_number(value[0]) and _finite_number(value[1])


def _nonzero_xy(value: Any) -> bool:
    return _xy(value) and (float(value[0]) != 0.0 or float(value[1]) != 0.0)


def _distinct_xy(left: Any, right: Any) -> bool:
    if not _xy(left) or not _xy(right):
        return False
    return float(left[0]) != float(right[0]) or float(left[1]) != float(right[1])


def _has_two_distinct_xy(points: Any) -> bool:
    if not isinstance(points, list):
        return False
    seen: List[tuple[float, float]] = []
    for point in points:
        if not _xy(point):
            continue
        pair = (float(point[0]), float(point[1]))
        if any(pair != other for other in seen):
            return True
        if pair not in seen:
            seen.append(pair)
    return False


def _is_geom_entity(ent: Any) -> bool:
    """True for a supported, nondegenerate geometric primitive."""
    if not isinstance(ent, dict):
        return False
    et = str(ent.get("type") or "").upper()
    if et not in _GEOM_ENTITY_TYPES:
        return False
    if et == "LINE":
        return _distinct_xy(ent.get("start"), ent.get("end"))
    if et == "CIRCLE":
        return _xy(ent.get("center")) and _positive_number(ent.get("radius"))
    if et == "ARC":
        start_a = ent.get("start_angle")
        end_a = ent.get("end_angle")
        return (
            _xy(ent.get("center"))
            and _positive_number(ent.get("radius"))
            and _finite_number(start_a)
            and _finite_number(end_a)
            and float(start_a) != float(end_a)
        )
    if et in ("LWPOLYLINE", "POLYLINE"):
        return _has_two_distinct_xy(ent.get("points"))
    if et == "ELLIPSE":
        ratio = ent.get("ratio", 1.0)
        return (
            _xy(ent.get("center"))
            and _nonzero_xy(ent.get("major"))
            and _positive_number(ratio)
        )
    if et == "SPLINE":
        return _has_two_distinct_xy(ent.get("control_points"))
    if et == "INSERT":
        name = ent.get("block")
        bhash = ent.get("block_hash")
        has_id = (isinstance(name, str) and bool(name.strip())) or (
            isinstance(bhash, str) and bool(bhash.strip())
        )
        if not (has_id and _xy(ent.get("insert"))):
            return False
        if "scale" in ent:
            scale = ent.get("scale")
            if not _xy(scale):
                return False
            if float(scale[0]) == 0.0 or float(scale[1]) == 0.0:
                return False
        return True
    return False


def _is_geom_json(obj: Any) -> bool:
    """True only for declared v2-like geometry with a real geometric entity.

    A non-empty ``entities`` list is not enough: ``{"entities":[{}]}``
    normalizes to UNKNOWN and must not be labeled precision-l4.
    """
    if not isinstance(obj, dict):
        return False
    entities = obj.get("entities")
    if not isinstance(entities, list) or not entities:
        return False
    return any(_is_geom_entity(e) for e in entities)


def _parse_query_geom(
    file_bytes: bytes, file_name: str = ""
) -> Optional[Dict[str, Any]]:
    if not file_bytes:
        return None
    suffix = file_suffix(file_name)
    if suffix == ".json":
        try:
            obj = json.loads(file_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        return obj if _is_geom_json(obj) else None
    if suffix == ".dxf":
        geom = _extract_dxf_geom(file_bytes)
        return geom if _is_geom_json(geom) else None
    return None


def _candidate_geom(
    candidate: CandidateDecision, geom_store: Any
) -> Optional[Dict[str, Any]]:
    right = (candidate.provenance or {}).get("geom_json")
    if _is_geom_json(right):
        return right
    cid = candidate.candidate_id or ""
    if geom_store is None or not _looks_like_file_hash(cid):
        return None
    try:
        loaded = geom_store.load(cid)
    except Exception:
        logger.debug("review_reuse_geom_store_load_failed", exc_info=True)
        return None
    return loaded if _is_geom_json(loaded) else None


def _canonical_geom(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Uppercase entity types so PrecisionVerifier keeps coordinates."""
    out = dict(obj)
    ents = obj.get("entities")
    if not isinstance(ents, list):
        return out
    canon: List[Any] = []
    for ent in ents:
        if not isinstance(ent, dict):
            canon.append(ent)
            continue
        item = dict(ent)
        typ = item.get("type")
        if isinstance(typ, str):
            item["type"] = typ.upper()
        canon.append(item)
    out["entities"] = canon
    return out


def _try_l4_score(
    query_geom: Dict[str, Any],
    candidate: CandidateDecision,
    geom_store: Any = None,
) -> Optional[float]:
    if not _is_geom_json(query_geom):
        return None
    right = _candidate_geom(candidate, geom_store)
    if not _is_geom_json(right):
        return None
    try:
        from src.core.dedupcad_precision import PrecisionVerifier

        scored = PrecisionVerifier().score_pair(
            _canonical_geom(query_geom), _canonical_geom(right)
        )
        return float(scored.score)
    except Exception:
        logger.warning("review_reuse_l4_score_failed", exc_info=True)
        return None


def _strip_stale_l4(candidate: CandidateDecision) -> None:
    methods = [m for m in _methods(candidate) if m != "precision-l4"]
    verification = dict(candidate.verification or {})
    verification["methods"] = methods
    try:
        level = int(verification.get("level") or 0)
    except (TypeError, ValueError):
        level = 0
    if level >= 4:
        verification["level"] = 0
    candidate.verification = verification


def _clear_provisional_unverified(candidate: CandidateDecision) -> None:
    candidate.rejection_reasons = [
        r for r in candidate.rejection_reasons if r not in _PROVISIONAL_UNVERIFIED
    ]


def _clear_stale_low_precision(candidate: CandidateDecision) -> None:
    low = RejectionReason.low_precision_score.value
    candidate.rejection_reasons = [
        r for r in candidate.rejection_reasons if r != low
    ]


def _ensure_l4_level(candidate: CandidateDecision) -> None:
    verification = dict(candidate.verification or {})
    methods = _methods(candidate)
    if "precision-l4" not in methods:
        methods.append("precision-l4")
    verification["methods"] = methods
    try:
        prev_level = int(verification.get("level") or 0)
    except (TypeError, ValueError):
        prev_level = 0
    verification["level"] = max(prev_level, 4)
    candidate.verification = verification


def _mark_low_precision(candidate: CandidateDecision) -> None:
    _append_reason(candidate, RejectionReason.low_precision_score.value)
    candidate.state = CandidateState.different
    verification = dict(candidate.verification or {})
    verification["verdict"] = CandidateState.different.value
    candidate.verification = verification


def _had_low_precision(candidate: CandidateDecision) -> bool:
    return RejectionReason.low_precision_score.value in (
        candidate.rejection_reasons or []
    )


def _independent_rejections(candidate: CandidateDecision) -> List[str]:
    """Reasons that are not stale low-precision or provisional unverified labels."""
    return [
        reason
        for reason in candidate.rejection_reasons or []
        if reason != RejectionReason.low_precision_score.value
        and reason not in _PROVISIONAL_UNVERIFIED
    ]


def _restore_after_high_l4(
    candidate: CandidateDecision, *, had_low_precision: bool
) -> None:
    """Restore similar only when `different` was a stale low-precision rejection.

    Reasonless adapter ``different`` and independent reasons such as
    ``version_gate_filtered`` stay ``different``.
    """
    if (
        candidate.state == CandidateState.different
        and had_low_precision
        and not _independent_rejections(candidate)
    ):
        candidate.state = CandidateState.similar
    verification = dict(candidate.verification or {})
    verification["verdict"] = candidate.state.value
    candidate.verification = verification


def _apply_trusted_l4(candidate: CandidateDecision, score: float) -> None:
    had_low = _had_low_precision(candidate)
    _ensure_l4_level(candidate)
    _clear_provisional_unverified(candidate)
    _clear_stale_low_precision(candidate)
    if score < LOW_PRECISION_THRESHOLD:
        _mark_low_precision(candidate)
        return
    _restore_after_high_l4(candidate, had_low_precision=had_low)


def _apply_l4_score(candidate: CandidateDecision, score: float) -> None:
    candidate.scores = dict(candidate.scores)
    candidate.scores["geometric"] = score
    _apply_trusted_l4(candidate, score)


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

    query_geom: Optional[Dict[str, Any]] = None
    query_geom_loaded = False
    geom_store: Any = None
    geom_store_failed = False

    def _query_geom() -> Optional[Dict[str, Any]]:
        nonlocal query_geom, query_geom_loaded
        if not query_geom_loaded:
            query_geom = _parse_query_geom(file_bytes, file_name)
            query_geom_loaded = True
        return query_geom

    def _geom_store() -> Any:
        nonlocal geom_store, geom_store_failed
        if geom_store is not None or geom_store_failed:
            return geom_store
        try:
            from src.core.dedupcad_precision import create_geom_store

            geom_store = create_geom_store()
        except Exception:
            logger.debug("review_reuse_geom_store_init_failed", exc_info=True)
            geom_store_failed = True
        return geom_store

    out: List[CandidateDecision] = []
    for original in candidates:
        candidate = original.model_copy(deep=True)
        if candidate.state == CandidateState.insufficient_evidence:
            out.append(candidate)
            continue

        geometric = candidate.scores.get("geometric")
        has_numeric_geom = isinstance(geometric, (int, float)) and not isinstance(
            geometric, bool
        )
        if _has_method(candidate, "precision-l4"):
            if _is_finite_unit_score(geometric):
                _apply_trusted_l4(candidate, float(geometric))
                out.append(candidate)
                continue
            _strip_stale_l4(candidate)
            if has_numeric_geom:
                scores = dict(candidate.scores)
                scores.pop("geometric", None)
                candidate.scores = scores
                has_numeric_geom = False
                geometric = None

        has_cand_geom = _is_geom_json(
            (candidate.provenance or {}).get("geom_json")
        )
        hash_id = _looks_like_file_hash(candidate.candidate_id or "")
        if has_cand_geom or hash_id:
            q = _query_geom()
            l4 = (
                _try_l4_score(
                    q,
                    candidate,
                    None if has_cand_geom else _geom_store(),
                )
                if q
                else None
            )
            if l4 is not None:
                _apply_l4_score(candidate, l4)
                out.append(candidate)
                continue

        if _is_live_vision(candidate):
            _strip_stale_l4(candidate)
            _append_reason(candidate, RejectionReason.vision_only_unverified.value)
            out.append(candidate)
            continue

        if not has_numeric_geom or not _has_method(candidate, "precision-l4"):
            _append_reason(candidate, RejectionReason.missing_geom_json.value)
        elif float(geometric) < LOW_PRECISION_THRESHOLD:
            _mark_low_precision(candidate)
        out.append(candidate)
    return out
