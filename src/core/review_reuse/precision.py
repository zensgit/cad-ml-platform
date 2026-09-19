"""Precision pass after recall — honest geometric verification.

Does **not** invent geometric scores from visual similarity.
When both query and candidate geom-json payloads are present, scores with
``PrecisionVerifier`` (L4). Query geom comes from JSON uploads or local DXF
extract; candidate geom comes from hit ``geom_json`` or the geom store.
Inline ``geom_json`` is used only for local L4 and stripped before
persist/export so live unscoped hits cannot leak across tenants.
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


def strip_transient_candidate_geom(
    candidates: List[CandidateDecision],
) -> List[CandidateDecision]:
    """Drop inline geom_json so tenant GET/export cannot leak live hits."""
    for candidate in candidates:
        provenance = dict(candidate.provenance or {})
        if "geom_json" in provenance:
            provenance.pop("geom_json", None)
            candidate.provenance = provenance
    return candidates


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
    }
)
# Geometric types we cannot certify faithfully. Presence fails the whole
# payload: dropping them would let an incidental LINE match two different
# block/spline drawings as precision-l4 1.0.
_UNSCORED_GEOM_TYPES = frozenset(
    {
        "INSERT",
        "SPLINE",
        "LEADER",
        "MULTILEADER",
        "POINT",
        "SOLID",
        "TRACE",
        "3DFACE",
        "3DSOLID",
        "MESH",
        "POLYFACE",
        "REGION",
        "BODY",
        "SURFACE",
        "XLINE",
        "RAY",
        "MLINE",
        "HATCH",
        "IMAGE",
        "WIPEOUT",
        "HELIX",
    }
)
_ANNOTATION_ENTITY_TYPES = frozenset(
    {
        "TEXT",
        "MTEXT",
        "DIMENSION",
        "ATTRIB",
        "ATTDEF",
    }
)


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


# PrecisionVerifier.normalize_v2 rounds coords/radii/angles to 3 decimals.
# Sub-0.001 primitives that collapse to zero-length after that rounding must
# not be admitted as L4 geometry.
_L4_QUANT_NDIGITS = 3
# Hard matcher cap (vendor Hungarian bound). Never raise this to the
# untrusted exploded entity count.
_L4_MAX_MATCH_ENTITIES = 128
# PrecisionVerifier compares only the first 16 spline control points.
_L4_MAX_SPLINE_CTRL = 16


def _quantized(value: Any) -> Optional[float]:
    if not _finite_number(value):
        return None
    return round(float(value), _L4_QUANT_NDIGITS)


def _quantized_xy(value: Any) -> Optional[tuple[float, float]]:
    if not _xy(value):
        return None
    x = _quantized(value[0])
    y = _quantized(value[1])
    if x is None or y is None:
        return None
    return (x, y)


def _positive_number(value: Any) -> bool:
    quantized = _quantized(value)
    return quantized is not None and quantized > 0.0


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
    pair = _quantized_xy(value)
    return pair is not None and pair != (0.0, 0.0)


def _distinct_xy(left: Any, right: Any) -> bool:
    a = _quantized_xy(left)
    b = _quantized_xy(right)
    return a is not None and b is not None and a != b


def _has_two_distinct_xy(points: Any) -> bool:
    if not isinstance(points, list):
        return False
    seen: List[tuple[float, float]] = []
    for point in points:
        pair = _quantized_xy(point)
        if pair is None:
            return False
        if pair not in seen:
            seen.append(pair)
    return len(seen) >= 2


def _ccw_sweep_deg(start: float, end: float) -> Optional[float]:
    """DXF ARC is CCW from start to end. 0 means a full wrap, not a tiny arc."""
    sweep = (float(end) - float(start)) % 360.0
    if sweep == 0.0:
        return None
    return sweep


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
        start_q = _quantized(ent.get("start_angle"))
        end_q = _quantized(ent.get("end_angle"))
        return (
            _xy(ent.get("center"))
            and _positive_number(ent.get("radius"))
            and start_q is not None
            and end_q is not None
            and _ccw_sweep_deg(start_q, end_q) is not None
        )
    if et in ("LWPOLYLINE", "POLYLINE"):
        if _polyline_closed(ent) is None:
            return False
        pts = ent.get("points")
        if not isinstance(pts, list) or len(pts) > _L4_MAX_MATCH_ENTITIES:
            return False
        return _has_two_distinct_xy(pts)
    if et == "ELLIPSE":
        ratio = ent.get("ratio", 1.0)
        if not (
            _xy(ent.get("center"))
            and _nonzero_xy(ent.get("major"))
            and _positive_number(ratio)
        ):
            return False
        # Omitted params mean a full ellipse. Equal supplied params are
        # a zero-sweep, same as a degenerate ARC.
        if "start_param" not in ent and "end_param" not in ent:
            return True
        start_p = _quantized(ent.get("start_param"))
        end_p = _quantized(ent.get("end_param"))
        if start_p is None or end_p is None or start_p == end_p:
            return False
        # Vendor ELLIPSE cost uses only |end-start|, so 0..π vs π..2π
        # would score 1.0. Admit full ellipses only.
        span = abs(end_p - start_p)
        return abs(span - 2.0 * math.pi) <= 10 ** (-_L4_QUANT_NDIGITS)
    if et == "SPLINE":
        cps = ent.get("control_points")
        if not isinstance(cps, list) or len(cps) > _L4_MAX_SPLINE_CTRL:
            return False
        return _has_two_distinct_xy(cps)
    return False


def _is_supported_geom_type(ent: Any) -> bool:
    if not isinstance(ent, dict):
        return False
    return str(ent.get("type") or "").upper() in _GEOM_ENTITY_TYPES


def _is_unscored_geom_type(ent: Any) -> bool:
    if not isinstance(ent, dict):
        return False
    return str(ent.get("type") or "").upper() in _UNSCORED_GEOM_TYPES


def _is_geom_json(obj: Any) -> bool:
    """True only for declared v2-like geometry with a real geometric entity.

    A non-empty ``entities`` list is not enough: ``{"entities":[{}]}``
    normalizes to UNKNOWN and must not be labeled precision-l4.
    One valid primitive also is not enough when the same list still
    contains a malformed supported type (zero-radius CIRCLE next to a
    LINE): that junk would otherwise be scored and can fake a match.
    """
    if not isinstance(obj, dict):
        return False
    entities = obj.get("entities")
    if not isinstance(entities, list) or not entities:
        return False
    if len(entities) > _L4_MAX_MATCH_ENTITIES:
        return False
    has_valid = False
    for entity in entities:
        if not isinstance(entity, dict):
            return False
        et = str(entity.get("type") or "").upper()
        if not et:
            return False
        if _is_unscored_geom_type(entity):
            return False
        if _is_supported_geom_type(entity) and not _is_geom_entity(entity):
            return False
        if _is_geom_entity(entity):
            has_valid = True
            continue
        if et in _ANNOTATION_ENTITY_TYPES:
            continue
        # Unknown geometric type (POINT, MESH, …) must not be dropped.
        return False
    return has_valid


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
        # Malformed supported primitives must not reach PrecisionVerifier.
        if _is_supported_geom_type(item) and not _is_geom_entity(item):
            continue
        # Omitted ellipse params mean a full ellipse. normalize_v2 would
        # otherwise default both to 0.0 (zero-sweep) vs DXF 0..2π.
        if item.get("type") == "ELLIPSE":
            if "start_param" not in item and "end_param" not in item:
                item["start_param"] = 0.0
                item["end_param"] = 2.0 * math.pi
        canon.append(item)
    out["entities"] = canon
    return out


def _bulge_is_unsafe(raw: Any) -> bool:
    """Any nonzero or non-finite bulge is unsafe to explode into a chord."""
    if raw is None:
        return False
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return True
    number = float(raw)
    return (not math.isfinite(number)) or number != 0.0


def _vertex_has_unsafe_bulge(point: Any) -> bool:
    if isinstance(point, dict) and "bulge" in point:
        return _bulge_is_unsafe(point.get("bulge"))
    if isinstance(point, (list, tuple)) and len(point) >= 5:
        return _bulge_is_unsafe(point[4])
    return False


def _polyline_has_bulge(entity: Dict[str, Any]) -> bool:
    """True when a polyline encodes a curved segment we would drop as LINE."""
    raw_bulges = entity.get("bulges")
    if isinstance(raw_bulges, list) and any(
        _bulge_is_unsafe(item) for item in raw_bulges
    ):
        return True
    pts = entity.get("points")
    if not isinstance(pts, list):
        return False
    return any(_vertex_has_unsafe_bulge(point) for point in pts)


def _width_is_unsafe(raw: Any) -> bool:
    """Any nonzero or non-finite width is unsafe to explode into a LINE."""
    if raw is None:
        return False
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return True
    number = float(raw)
    return (not math.isfinite(number)) or number != 0.0


def _polyline_has_unsafe_width(entity: Dict[str, Any]) -> bool:
    """True when exploding would drop a nonzero polyline width."""
    if entity.get("has_width") is True:
        return True
    if "const_width" in entity and _width_is_unsafe(entity.get("const_width")):
        return True
    if "width" in entity:
        raw = entity.get("width")
        if isinstance(raw, list):
            if any(_width_is_unsafe(item) for item in raw):
                return True
        elif _width_is_unsafe(raw):
            return True
    widths = entity.get("widths")
    if isinstance(widths, list) and any(_width_is_unsafe(item) for item in widths):
        return True
    pts = entity.get("points")
    if not isinstance(pts, list):
        return False
    for point in pts:
        if isinstance(point, dict):
            for key in ("start_width", "end_width", "width"):
                if key in point and _width_is_unsafe(point.get(key)):
                    return True
        elif isinstance(point, (list, tuple)) and len(point) >= 4:
            if _width_is_unsafe(point[2]) or _width_is_unsafe(point[3]):
                return True
    return False


def _insert_identities(
    geom: Dict[str, Any],
) -> List[tuple[str, Optional[tuple[float, float]], str]]:
    found: List[tuple[str, Optional[tuple[float, float]], str]] = []
    ents = geom.get("entities")
    if not isinstance(ents, list):
        return found
    for entity in ents:
        if not isinstance(entity, dict):
            continue
        if str(entity.get("type") or "").upper() != "INSERT":
            continue
        name = str(entity.get("block") or "")
        bhash = entity.get("block_hash")
        hashed = bhash.strip() if isinstance(bhash, str) else ""
        if not hashed:
            continue
        found.append((name, _quantized_xy(entity.get("insert")), hashed))
    return found


def _insert_block_hash_conflict(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    """Same INSERT identity with a different block_hash must not pass L4."""
    left_ids = _insert_identities(left)
    right_ids = _insert_identities(right)
    if not left_ids or not right_ids:
        return False
    used: set[int] = set()
    for name, pos, left_hash in left_ids:
        for idx, (rname, rpos, right_hash) in enumerate(right_ids):
            if idx in used:
                continue
            if name == rname and pos == rpos:
                used.add(idx)
                if left_hash != right_hash:
                    return True
                break
    return False


def _spline_degree(entity: Dict[str, Any]) -> Optional[int]:
    raw = entity.get("degree", 3)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    if not math.isfinite(float(raw)):
        return None
    return int(raw)


def _spline_signatures(geom: Dict[str, Any]) -> List[tuple[int, Optional[int]]]:
    """Control-count plus degree; vendor matching ignores degree/knots."""
    sigs: List[tuple[int, Optional[int]]] = []
    ents = geom.get("entities")
    if not isinstance(ents, list):
        return sigs
    for entity in ents:
        if not isinstance(entity, dict):
            continue
        if str(entity.get("type") or "").upper() != "SPLINE":
            continue
        cps = entity.get("control_points")
        n = len(cps) if isinstance(cps, list) else 0
        sigs.append((n, _spline_degree(entity)))
    return sorted(sigs)


def _arc_records(
    geom: Dict[str, Any],
) -> Optional[List[tuple[tuple[float, float], float, float, float]]]:
    """(center, radius, CCW sweep, start mod 360). None if unmeasurable."""
    records: List[tuple[tuple[float, float], float, float, float]] = []
    ents = geom.get("entities")
    if not isinstance(ents, list):
        return records
    for entity in ents:
        if not isinstance(entity, dict):
            continue
        if str(entity.get("type") or "").upper() != "ARC":
            continue
        center = _quantized_xy(entity.get("center"))
        radius = _quantized(entity.get("radius"))
        start_q = _quantized(entity.get("start_angle"))
        end_q = _quantized(entity.get("end_angle"))
        if (
            center is None
            or radius is None
            or radius <= 0.0
            or start_q is None
            or end_q is None
        ):
            return None
        raw_sweep = _ccw_sweep_deg(start_q, end_q)
        if raw_sweep is None:
            return None
        quantized = _quantized(raw_sweep)
        start_mod = _quantized(start_q % 360.0)
        if quantized is None or start_mod is None:
            return None
        records.append((center, radius, quantized, start_mod))
    return records


def _arc_angle_tie(
    sweep: float, start: float, rsweep: float, rstart: float
) -> float:
    """Tie-break for equal center+radius ARCs so reorder still L4."""
    dstart = abs(start - rstart) % 360.0
    dstart = min(dstart, 360.0 - dstart)
    return abs(sweep - rsweep) + dstart


def _arc_sweeps_conflict(
    left: Dict[str, Any],
    right: Dict[str, Any],
    *,
    angle_tol: float,
    center_tol: float,
    radius_tol: float,
) -> bool:
    """True when a matched ARC's CCW sweep differs beyond angle_tol.

    Pair by nearest center/radius (same positional assignment the scorer
    uses) then compare CCW sweeps. Sweep-compatible matching would pair
    a nearby sliver with a near-full arc and still score L4 ~0.9.
    """
    if not math.isfinite(angle_tol) or angle_tol < 0.0:
        return True
    if not math.isfinite(center_tol) or center_tol < 0.0:
        return True
    if not math.isfinite(radius_tol) or radius_tol < 0.0:
        return True
    left_r = _arc_records(left)
    right_r = _arc_records(right)
    if left_r is None or right_r is None:
        return True
    if len(left_r) != len(right_r):
        return True
    n = len(left_r)
    if n == 0:
        return False
    inf = 1e9
    # Angle is a tie-break only when center+radius already match. A 1e-6
    # sweep term can outweigh a 0.001 quantized step and pair swapped
    # nearby ARCs, so L4 would miss the sweep conflict (~0.995).
    angle_tie = 1e-6
    cost = [[inf] * n for _ in range(n)]
    for i, (center, radius, sweep, start) in enumerate(left_r):
        for j, (rcenter, rradius, rsweep, rstart) in enumerate(right_r):
            dx = rcenter[0] - center[0]
            dy = rcenter[1] - center[1]
            dist = math.hypot(dx, dy)
            if dist > center_tol:
                continue
            dr = abs(rradius - radius)
            if dr > radius_tol:
                continue
            base = dist + dr
            cost[i][j] = base
            if base == 0.0:
                cost[i][j] = base + angle_tie * _arc_angle_tie(
                    sweep, start, rsweep, rstart
                )
    from src.core.dedupcad_precision.vendor.entities_match import _hungarian

    assign, _total = _hungarian(cost)
    for i, j in enumerate(assign):
        if j < 0 or cost[i][j] >= inf:
            return True
        if abs(left_r[i][2] - right_r[j][2]) > angle_tol:
            return True
    return False


def _polyline_closed(entity: Dict[str, Any]) -> Optional[bool]:
    """True/False when ``closed`` is a real bool; None if malformed.

    ``bool("false")`` is True, so a string flag must not explode a closer.
    Missing ``closed`` means open.
    """
    if "closed" not in entity:
        return False
    raw = entity.get("closed")
    if isinstance(raw, bool):
        return raw
    return None


def _explode_polyline(entity: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Turn a polyline into absolute LINE segments.

    PrecisionVerifier canonicalizes each polyline (translate/scale/rotate)
    before matching, so identical shapes at different coordinates score ~1.0.
    LINE matching is positional.
    """
    closed = _polyline_closed(entity)
    if closed is None:
        return []
    pts = entity.get("points")
    if (
        not isinstance(pts, list)
        or len(pts) < 2
        or len(pts) > _L4_MAX_MATCH_ENTITIES
    ):
        return []
    vertices: List[List[float]] = []
    for point in pts:
        if not _xy(point):
            return []
        vertices.append([float(point[0]), float(point[1])])
    if len(vertices) < 2:
        return []
    segments = list(zip(vertices, vertices[1:]))
    if closed and len(vertices) >= 3:
        segments.append((vertices[-1], vertices[0]))
    lines: List[Dict[str, Any]] = []
    for start, end in segments:
        if not _distinct_xy(start, end):
            continue
        lines.append({"type": "LINE", "start": start, "end": end})
    return lines


def _geometry_only_geom(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Canonical geom restricted to validated geometric primitives.

    PrecisionVerifier fuses text/layers/dimensions/HATCH into ``score``.
    Shared TEXT or HATCH next to different LINEs can then exceed
    LOW_PRECISION_THRESHOLD and be certified as L4 similar.
    Polylines are exploded to LINEs so translated clones stay positional.
    """
    out = _canonical_geom(obj)
    ents = out.get("entities")
    if isinstance(ents, list):
        cleaned: List[Any] = []
        for entity in ents:
            if not _is_geom_entity(entity):
                continue
            item = dict(entity)
            # Per-entity layer names still feed layer_mismatch_penalty even
            # when w_layers=0. Geometry-only L4 must ignore CAD layers.
            item.pop("layer", None)
            et = str(item.get("type") or "").upper()
            if et in ("LWPOLYLINE", "POLYLINE"):
                # Straight-LINE explode drops bulge and width; fail closed
                # rather than certify a semicircle or a thick stroke as a
                # zero-width chord between the same endpoints.
                if _polyline_has_bulge(item) or _polyline_has_unsafe_width(item):
                    out["entities"] = []
                    return out
                cleaned.extend(_explode_polyline(item))
            else:
                cleaned.append(item)
        out["entities"] = cleaned
    out.pop("text_content", None)
    out.pop("dimensions", None)
    out.pop("hatches", None)
    return out


def _entity_count(geom: Dict[str, Any]) -> int:
    ents = geom.get("entities")
    return len(ents) if isinstance(ents, list) else 0


def _penalize_unmatched(score: float, n_left: int, n_right: int) -> float:
    """entities_similarity truncates to min(len(A), len(B)); extra entities
    would otherwise leave a subset match at 1.0."""
    denom = max(n_left, n_right)
    if denom <= 0:
        return score
    return float(score) * (min(n_left, n_right) / float(denom))


def _drawing_units(geom: Dict[str, Any]) -> Optional[int]:
    """INSUNITS from geom JSON. None means the payload never declared units."""
    raw = geom.get("insunits")
    if raw is None:
        info = geom.get("file_info")
        if isinstance(info, dict):
            raw = info.get("insunits")
    if raw is None:
        return None
    if isinstance(raw, bool):
        return 0
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if not math.isfinite(raw) or raw != math.floor(raw):
            return 0
        return int(raw)
    return 0


def _units_conflict(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    """Inch vs mm (or unknown $INSUNITS) must not score as identical L4."""
    left_u = _drawing_units(left)
    right_u = _drawing_units(right)
    if left_u is None and right_u is None:
        return False
    if left_u is None or right_u is None:
        return True
    if left_u <= 0 or right_u <= 0:
        return True
    return left_u != right_u


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
    if _units_conflict(query_geom, right):
        return None
    try:
        from dataclasses import replace

        from src.core.dedupcad_precision import PrecisionVerifier
        from src.core.dedupcad_precision.vendor.v2_normalize import normalize_v2

        # Start from PrecisionVerifier() then force entities_geom_hash off.
        # replace(Settings()) re-enables the bag-of-features fallback via
        # DEDUPCAD2_ENTITIES_GEOM_HASH. Cloning PrecisionVerifier().settings
        # is not enough: CAD_ML_PLATFORM_L4_ENTITIES_GEOM_HASH=1 would still
        # certify layout-shifted clones as geometric matches.
        cfg = replace(
            PrecisionVerifier().settings,
            w_text=0.0,
            w_layers=0.0,
            w_dimensions=0.0,
            w_hatch_extra=0.0,
            entities_geom_hash=False,
            layer_mismatch_penalty=0.0,
            max_match_entities=_L4_MAX_MATCH_ENTITIES,
        )
        # Cap before copy/explode: a million-vertex polyline must not
        # materialize LINEs and then get rejected.
        if (
            _entity_count(query_geom) > _L4_MAX_MATCH_ENTITIES
            or _entity_count(right) > _L4_MAX_MATCH_ENTITIES
        ):
            return None
        left = _geometry_only_geom(query_geom)
        right_g = _geometry_only_geom(right)
        left_n = _entity_count(left)
        right_n = _entity_count(right_g)
        # Truncating a large drawing to the cap would certify a prefix match.
        if left_n > _L4_MAX_MATCH_ENTITIES or right_n > _L4_MAX_MATCH_ENTITIES:
            return None
        # Re-check after verifier quantization: sub-0.001 LINEs/radii collapse
        # to zero-length and would otherwise score as identical L4 matches.
        if not _is_geom_json(normalize_v2(left, cfg)) or not _is_geom_json(
            normalize_v2(right_g, cfg)
        ):
            return None
        if _insert_block_hash_conflict(left, right_g):
            return 0.0
        if _spline_signatures(left) != _spline_signatures(right_g):
            return 0.0
        if _arc_sweeps_conflict(
            left,
            right_g,
            angle_tol=float(cfg.tol_arc_angle_deg),
            center_tol=float(cfg.tol_circle_center),
            radius_tol=float(cfg.tol_circle_radius),
        ):
            return 0.0
        scored = PrecisionVerifier(settings=cfg).score_pair(left, right_g)
        score = scored.score
        if not _is_finite_unit_score(score):
            return None
        score = _penalize_unmatched(float(score), left_n, right_n)
        if not _is_finite_unit_score(score):
            return None
        return float(score)
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
        return strip_transient_candidate_geom(
            list(hook(file_name, file_bytes, list(candidates)))
        )

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
        has_cand_geom = _is_geom_json(
            (candidate.provenance or {}).get("geom_json")
        )
        hash_id = _looks_like_file_hash(candidate.candidate_id or "")
        # Local geometry-only L4 wins over a remote fused precision_score.
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
    return strip_transient_candidate_geom(out)
