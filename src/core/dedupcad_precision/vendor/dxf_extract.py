"""
DXF extractor for DedupCAD 2.0 standalone.
Relies on ezdxf to parse DXF and produce a simple JSON structure.
"""

import json
import math
import re
from pathlib import Path  # ensure Path available for cache directory logic
from typing import Any, Dict, List, Optional

# Settings import (works for both package and script modes)
try:  # pragma: no cover - flexible import
    from .config import get_settings  # type: ignore
except Exception:  # pragma: no cover
    try:
        from config import get_settings  # type: ignore
    except Exception:  # pragma: no cover

        def get_settings():  # type: ignore
            class _S:
                pass

            return _S()


try:
    import ezdxf  # type: ignore
except Exception as e:  # pragma: no cover - optional dependency
    ezdxf = None


# Bump when extract payload fields change. v10 does not flatten polyface
# meshes (flag 64) or non-planar primitives into 2D coordinates.
_EXTRACT_CACHE_VERSION = 10
# Classic POLYLINE group-70: 8=3D, 16=polygon mesh, 32=mesh closed in N,
# 64=polyface. Bit 32 alone is not a polyface.
_NON_2D_POLYLINE_FLAGS = 8 | 16 | 32 | 64


def _header_insunits(doc: Any) -> int:
    """DXF $INSUNITS; 0 means unitless/unknown and must not be L4-certified."""
    try:
        raw = doc.header.get("$INSUNITS")
        if raw is None:
            raw = getattr(doc, "units", 0)
        if isinstance(raw, bool):
            return 0
        if isinstance(raw, float):
            if not math.isfinite(raw) or raw != math.floor(raw):
                return 0
            number = int(raw)
        elif isinstance(raw, int):
            number = raw
        else:
            parsed = float(str(raw).strip())
            if not math.isfinite(parsed) or parsed != math.floor(parsed):
                return 0
            number = int(parsed)
        return number if 1 <= number <= 24 else 0
    except Exception:
        return 0


# HEADER is at the start of a DXF; never slurp ENTITIES/BLOCKS for $INSUNITS.
_HEADER_SCAN_MAX_BYTES = 1_048_576
_BINARY_DXF_SENTINEL = b"AutoCAD Binary DXF\r\n\x1a\x00"


def _read_dxf_header_text(path: str) -> Optional[str]:
    """HEADER section text, or None if missing or not closed within the cap."""
    try:
        with open(path, "rb") as handle:
            raw = handle.read(_HEADER_SCAN_MAX_BYTES)
    except OSError:
        return None
    text = raw.decode("latin-1", errors="ignore")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    start = re.search(
        r"(?:^|\n)[ \t]*0[ \t]*\n[ \t]*SECTION[ \t]*\n[ \t]*2[ \t]*\n"
        r"[ \t]*HEADER[ \t]*\n",
        text,
        flags=re.IGNORECASE,
    )
    if start is None:
        return None
    rest = text[start.end() :]
    end = re.search(
        r"(?:^|\n)[ \t]*0[ \t]*\n[ \t]*ENDSEC[ \t]*(?:\n|\Z)",
        rest,
        flags=re.IGNORECASE,
    )
    if end is None:
        return None
    return rest[: end.start()]


def _is_binary_dxf(path: str) -> bool:
    try:
        with open(path, "rb") as handle:
            prefix = handle.read(len(_BINARY_DXF_SENTINEL))
    except OSError:
        return False
    return prefix == _BINARY_DXF_SENTINEL


def _insunits_tokens_unsafe(tokens: List[Any]) -> bool:
    """True when any $INSUNITS token is junk or they disagree."""
    if not tokens:
        return False
    seen: Optional[float] = None
    for raw_token in tokens:
        try:
            number = float(str(raw_token).strip())
        except (TypeError, ValueError):
            return True
        if (not math.isfinite(number)) or number != math.floor(number):
            return True
        if seen is None:
            seen = number
        elif number != seen:
            return True
    return False


def _raw_binary_insunits_unsafe(path: str) -> bool:
    """Scan binary HEADER $INSUNITS; fail closed on conflict or junk.

    Group 70 is an integer in binary DXF, so ASCII truncation does not
    apply. Duplicate declarations still must agree.
    """
    try:
        with open(path, "rb") as handle:
            raw = handle.read(_HEADER_SCAN_MAX_BYTES)
    except OSError:
        return True
    if not raw.startswith(_BINARY_DXF_SENTINEL):
        return True
    try:
        from ezdxf.lldxf.tagger import binary_tags_loader
    except Exception:
        return True
    in_header = False
    saw_section = False
    expecting_70 = False
    header_closed = False
    tokens: List[Any] = []
    try:
        for tag in binary_tags_loader(raw):
            code = int(tag.code)
            value = tag.value
            if not in_header:
                if code == 0 and str(value).strip().upper() == "SECTION":
                    saw_section = True
                    continue
                if (
                    saw_section
                    and code == 2
                    and str(value).strip().upper() == "HEADER"
                ):
                    in_header = True
                    saw_section = False
                    continue
                saw_section = False
                continue
            if code == 0 and str(value).strip().upper() == "ENDSEC":
                header_closed = True
                break
            if expecting_70:
                if code != 70:
                    return True
                tokens.append(value)
                expecting_70 = False
                continue
            if code == 9 and str(value).strip().upper() == "$INSUNITS":
                expecting_70 = True
    except Exception:
        if not header_closed:
            return True
    if expecting_70 or not header_closed:
        return True
    return _insunits_tokens_unsafe(tokens)


def _raw_insunits_non_integral(path: str) -> bool:
    """True when raw HEADER $INSUNITS is unsafe before ezdxf truncates.

    Fail closed on a fractional/non-finite/unparseable group-70 token, or
    when repeated $INSUNITS declarations disagree. The first match is not
    enough: ezdxf may apply a later truncated value. Binary DXF has no
    ASCII HEADER framing; scan its tags instead of wiping units.
    """
    if _is_binary_dxf(path):
        return _raw_binary_insunits_unsafe(path)
    header = _read_dxf_header_text(path)
    if header is None:
        # Unbounded/missing ASCII HEADER: do not trust ezdxf's truncated value.
        return True
    return _ascii_header_insunits_unsafe(header)


def _ascii_header_insunits_unsafe(header: str) -> bool:
    """Bind each ``$INSUNITS`` to its group-70 value, skipping group 999.

    An adjacency regex misses a comment between the variable and the
    integer, then ezdxf truncates ``4.9`` to ``4``. Unbound declarations
    fail closed.
    """
    lines = header.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if len(lines) % 2 != 0:
        return True
    pairs = [
        (lines[index].strip(), lines[index + 1].strip())
        for index in range(0, len(lines), 2)
    ]
    tokens: List[str] = []
    index = 0
    while index < len(pairs):
        code, value = pairs[index]
        if code == "9" and value.upper() == "$INSUNITS":
            bound = index + 1
            while bound < len(pairs) and pairs[bound][0] == "999":
                bound += 1
            if bound >= len(pairs) or pairs[bound][0] != "70":
                return True
            tokens.append(pairs[bound][1])
            index = bound + 1
            continue
        index += 1
    return _insunits_tokens_unsafe(tokens)


def _classic_polyline_is_non_2d(entity: Any) -> bool:
    """3D polylines, polygon meshes, and polyfaces are not 2D L4 geometry."""
    flags = getattr(getattr(entity, "dxf", None), "flags", 0)
    try:
        return bool(int(flags or 0) & _NON_2D_POLYLINE_FLAGS)
    except (TypeError, ValueError):
        return True


def _offset_from_xy_plane(raw: Any) -> bool:
    """True when Z/elevation is non-finite or survives L4's 3-decimal quant."""
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return True
    if not math.isfinite(number):
        return True
    return round(number, 3) != 0.0


def _vec3(raw: Any) -> Optional[tuple[float, float, float]]:
    if raw is None:
        return None
    try:
        return (float(raw[0]), float(raw[1]), float(raw[2]))
    except (TypeError, ValueError, IndexError):
        try:
            return (float(raw.x), float(raw.y), float(raw.z))
        except (TypeError, ValueError, AttributeError):
            return None


def _extrusion_leaves_xy_plane(entity: Any) -> bool:
    """Default extrusion (0, 0, 1) is the XY plane. Anything else is not."""
    dxf = getattr(entity, "dxf", None)
    if dxf is None:
        return True
    try:
        raw = dxf.extrusion
    except AttributeError:
        return False
    except Exception:
        return True
    coords = _vec3(raw)
    if coords is None:
        return True
    if not all(math.isfinite(value) for value in coords):
        return True
    rounded = tuple(round(value, 3) for value in coords)
    return rounded != (0.0, 0.0, 1.0)


def _dxf_primitive_non_planar(entity: Any, et: str) -> bool:
    """LINE/CIRCLE/ARC/ELLIPSE/LWPOLYLINE/POLYLINE that are not XY-planar."""
    if et not in ("LINE", "CIRCLE", "ARC", "ELLIPSE", "LWPOLYLINE", "POLYLINE"):
        return False
    if _extrusion_leaves_xy_plane(entity):
        return True
    dxf = getattr(entity, "dxf", None)
    if dxf is None:
        return True
    try:
        if et == "LINE":
            return _offset_from_xy_plane(dxf.start.z) or _offset_from_xy_plane(
                dxf.end.z
            )
        if et in ("CIRCLE", "ARC"):
            return _offset_from_xy_plane(dxf.center.z)
        if et == "ELLIPSE":
            return _offset_from_xy_plane(dxf.center.z) or _offset_from_xy_plane(
                dxf.major_axis.z
            )
        if et == "LWPOLYLINE":
            return _offset_from_xy_plane(getattr(dxf, "elevation", 0.0))
        elev = getattr(dxf, "elevation", None)
        elev_z = getattr(elev, "z", 0.0 if elev is None else elev)
        if _offset_from_xy_plane(elev_z):
            return True
        for vertex in entity.vertices:
            loc = vertex.dxf.location
            if _offset_from_xy_plane(getattr(loc, "z", 0.0)):
                return True
    except Exception:
        return True
    return False


def _dxf_polyline_closed(entity: Any, et: str) -> bool:
    """Classic POLYLINE uses is_closed; LWPOLYLINE uses closed."""
    if et == "POLYLINE":
        if hasattr(entity, "is_closed"):
            return bool(entity.is_closed)
        flags = getattr(getattr(entity, "dxf", None), "flags", 0) or 0
        try:
            return bool(int(flags) & 1)
        except Exception:
            return False
    return bool(getattr(entity, "closed", False))


def _width_nonzero(raw: Any) -> bool:
    if raw is None:
        return False
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return True
    return (not math.isfinite(number)) or number != 0.0


def _polyline_xy_and_bulges(
    entity: Any, et: str
) -> tuple[List[List[float]], List[float], bool]:
    """Keep bulge/width so ReviewReuse can refuse unsafe LINE explode."""
    pts: List[List[float]] = []
    bulges: List[float] = []
    has_width = False
    if et == "LWPOLYLINE":
        try:
            has_width = _width_nonzero(getattr(entity.dxf, "const_width", 0.0))
        except Exception:
            has_width = True
        for p in entity.get_points():
            pts.append([float(p[0]), float(p[1])])
            bulges.append(float(p[4]) if len(p) >= 5 else 0.0)
            if len(p) >= 4 and (_width_nonzero(p[2]) or _width_nonzero(p[3])):
                has_width = True
        return pts, bulges, has_width
    try:
        has_width = _width_nonzero(
            getattr(entity.dxf, "default_start_width", 0.0)
        ) or _width_nonzero(getattr(entity.dxf, "default_end_width", 0.0))
    except Exception:
        has_width = True
    for v in entity.vertices:
        loc = v.dxf.location
        pts.append([float(loc.x), float(loc.y)])
        try:
            bulges.append(float(getattr(v.dxf, "bulge", 0.0) or 0.0))
        except Exception:
            bulges.append(0.0)
        try:
            if _width_nonzero(getattr(v.dxf, "start_width", 0.0)) or _width_nonzero(
                getattr(v.dxf, "end_width", 0.0)
            ):
                has_width = True
        except Exception:
            has_width = True
    return pts, bulges, has_width


def extract_dxf(path: str, *, use_cache: bool = True) -> Dict[str, Any]:
    if ezdxf is None:
        raise RuntimeError("ezdxf not installed: pip install ezdxf")
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    insunits = _header_insunits(doc)
    if _raw_insunits_non_integral(path):
        insunits = 0

    layers = {}
    for layer in doc.layers:
        layers[layer.dxf.name] = {
            "color": getattr(layer.dxf, "color", None),
            "linetype": getattr(layer.dxf, "linetype", None),
        }

    # Block definitions (extended metadata and signature)
    blocks: Dict[str, Any] = {}
    # Optional in-process cache to avoid re-hashing identical block defs
    _bh_cache: Dict[str, str] = {}
    try:
        for bdef in doc.blocks:
            bname = str(getattr(bdef, "name", "") or "")
            if bname.startswith("*"):
                continue
            type_counts: Dict[str, int] = {}
            ent_count = 0
            ents_simple: List[Dict[str, Any]] = []
            try:
                for be in bdef:
                    ent_count += 1
                    t = getattr(be, "dxftype")() if hasattr(be, "dxftype") else "UNKNOWN"
                    type_counts[t] = type_counts.get(t, 0) + 1
                    # capture minimal inner entity for matching
                    ie: Dict[str, Any] = {"type": t}
                    if t == "POLYLINE" and _classic_polyline_is_non_2d(be):
                        ie["type"] = "POLYLINE3D"
                    elif _dxf_primitive_non_planar(be, t):
                        ie["type"] = "NONPLANAR"
                    elif t == "LINE":
                        try:
                            ie.update(
                                {
                                    "start": [float(be.dxf.start.x), float(be.dxf.start.y)],
                                    "end": [float(be.dxf.end.x), float(be.dxf.end.y)],
                                }
                            )
                        except Exception:
                            pass
                    elif t == "CIRCLE":
                        try:
                            ie.update(
                                {
                                    "center": [float(be.dxf.center.x), float(be.dxf.center.y)],
                                    "radius": float(be.dxf.radius),
                                }
                            )
                        except Exception:
                            pass
                    elif t in ("LWPOLYLINE", "POLYLINE"):
                        pts: List[List[float]] = []
                        bulges: List[float] = []
                        has_width = False
                        try:
                            pts, bulges, has_width = _polyline_xy_and_bulges(be, t)
                        except Exception:
                            pass
                        if pts:
                            ie.update(
                                {
                                    "points": pts,
                                    "closed": _dxf_polyline_closed(be, t),
                                }
                            )
                            if any(b != 0.0 for b in bulges):
                                ie["bulges"] = bulges
                            if has_width:
                                ie["has_width"] = True
                    elif t == "ELLIPSE":
                        try:
                            center = [float(be.dxf.center.x), float(be.dxf.center.y)]
                            major = be.dxf.major_axis
                            ratio = float(be.dxf.ratio)
                            ie.update(
                                {
                                    "center": center,
                                    "major": [float(major.x), float(major.y)],
                                    "ratio": ratio,
                                    "start_param": float(getattr(be.dxf, "start_param", 0.0)),
                                    "end_param": float(getattr(be.dxf, "end_param", 0.0)),
                                }
                            )
                        except Exception:
                            pass
                    elif t == "HATCH":
                        try:
                            pattern = str(getattr(be.dxf, "pattern_name", "") or "")
                            color = int(getattr(be.dxf, "color", 0) or 0)
                            loops = int(
                                getattr(be, "paths", []) and len(getattr(be, "paths").paths) or 0
                            )
                            ie.update({"pattern": pattern, "color": color, "loops": loops})
                        except Exception:
                            pass
                    elif t in ("TEXT", "MTEXT"):
                        tx = getattr(be.dxf, "text", None) or getattr(be, "text", "")
                        ie.update({"text": str(tx)})
                    # limit to core types (now include ELLIPSE/HATCH proxies)
                    if t in {
                        "LINE",
                        "CIRCLE",
                        "LWPOLYLINE",
                        "POLYLINE",
                        "TEXT",
                        "MTEXT",
                        "ELLIPSE",
                        "HATCH",
                    }:
                        ents_simple.append(ie)
            except Exception:
                pass
            # Simple signature and coarse area proxy from type:count pairs
            parts = [f"{k}:{type_counts[k]}" for k in sorted(type_counts.keys())]
            sig = "|".join(parts)

            # Build enriched v2 signature with basic geometric histograms
            def _build_sig2(ents: List[Dict[str, Any]]) -> str:
                import math as _m

                lens: List[float] = []
                angs: List[float] = []  # 0..180
                rads: List[float] = []
                for ie in ents:
                    t = ie.get("type")
                    if t == "LINE" and "start" in ie and "end" in ie:
                        x1, y1 = ie["start"]
                        x2, y2 = ie["end"]
                        dx, dy = float(x2) - float(x1), float(y2) - float(y1)
                        lens.append(_m.hypot(dx, dy))
                        a = abs(_m.degrees(_m.atan2(dy, dx))) % 180.0
                        angs.append(a)
                    elif t == "ARC" and "center" in ie and "radius" in ie:
                        cx, cy = ie.get("center", [0.0, 0.0])
                        r = float(ie.get("radius") or 0.0)
                        a1 = float(ie.get("start_angle", 0.0) or 0.0)
                        a2 = float(ie.get("end_angle", 0.0) or 0.0)
                        rads.append(r)
                        # approximate arc length and dominant direction via chord
                        a1r = _m.radians(a1)
                        a2r = _m.radians(a2)
                        sx, sy = float(cx) + r * _m.cos(a1r), float(cy) + r * _m.sin(a1r)
                        ex, ey = float(cx) + r * _m.cos(a2r), float(cy) + r * _m.sin(a2r)
                        dx, dy = ex - sx, ey - sy
                        # chord direction
                        angs.append(abs(_m.degrees(_m.atan2(dy, dx))) % 180.0)
                        # arc length contribution
                        da = abs((a2 - a1) % 360.0)
                        if da > 180.0:
                            da = 360.0 - da
                        lens.append(abs(r) * _m.radians(da))
                    elif t in ("LWPOLYLINE", "POLYLINE") and "points" in ie:
                        pts = ie["points"] or []
                        if pts:
                            # include closing segment if near-closed
                            closed = False
                            if len(pts) >= 3:
                                x1, y1 = pts[0]
                                xN, yN = pts[-1]
                                if _m.hypot(float(xN) - float(x1), float(yN) - float(y1)) <= 1e-3:
                                    closed = True
                            for i in range(len(pts) - 1):
                                x1, y1 = pts[i]
                                x2, y2 = pts[i + 1]
                                dx, dy = float(x2) - float(x1), float(y2) - float(y1)
                                lens.append(_m.hypot(dx, dy))
                                a = abs(_m.degrees(_m.atan2(dy, dx))) % 180.0
                                angs.append(a)
                            if closed:
                                x1, y1 = pts[-1]
                                x2, y2 = pts[0]
                                dx, dy = float(x2) - float(x1), float(y2) - float(y1)
                                lens.append(_m.hypot(dx, dy))
                                a = abs(_m.degrees(_m.atan2(dy, dx))) % 180.0
                                angs.append(a)
                    elif t == "CIRCLE" and "radius" in ie:
                        rads.append(float(ie.get("radius") or 0.0))

                def _hist(vals: List[float], bins: int, vmin: float, vmax: float) -> List[float]:
                    if not vals:
                        return [0.0] * bins
                    h = [0] * bins
                    rng = max(vmax - vmin, 1e-9)
                    for v in vals:
                        u = (float(v) - vmin) / rng
                        k = int(u * bins)
                        if k < 0:
                            k = 0
                        if k >= bins:
                            k = bins - 1
                        h[k] += 1
                    s = float(sum(h)) or 1.0
                    return [round(x / s, 3) for x in h]

                # Normalize lengths by median to get scale-robust histogram
                if lens:
                    sl = sorted(lens)
                    med = sl[len(sl) // 2]
                    if med > 1e-9:
                        lens_n = [l / med for l in lens]
                    else:
                        lens_n = lens
                else:
                    lens_n = []
                lh = _hist(lens_n, 8, 0.0, 4.0)
                ah = _hist(angs, 8, 0.0, 180.0)
                ch = _hist(rads, 6, 0.0, max(rads) if rads else 1.0)
                return (
                    "v2|"
                    + sig
                    + "|LH:"
                    + ",".join(f"{v:.3f}" for v in lh)
                    + "|AH:"
                    + ",".join(f"{v:.3f}" for v in ah)
                    + "|CH:"
                    + ",".join(f"{v:.3f}" for v in ch)
                )

            sig2 = _build_sig2(ents_simple)

            # Approximate area from inner entities (very coarse):
            # - CIRCLE: pi*r^2
            # - POLYLINE/LWPOLYLINE closed: polygon area
            # - others ignored
            def _poly_area(pts: List[List[float]]) -> float:
                if len(pts) < 3:
                    return 0.0
                s = 0.0
                for i in range(len(pts)):
                    x1, y1 = pts[i]
                    x2, y2 = pts[(i + 1) % len(pts)]
                    s += x1 * y2 - x2 * y1
                return abs(s) * 0.5

            approx_area = 0.0
            try:
                import math as _math

                for ie in ents_simple:
                    t = ie.get("type")
                    if t == "CIRCLE" and "radius" in ie:
                        r = float(ie.get("radius") or 0.0)
                        approx_area += _math.pi * r * r
                    elif t in ("LWPOLYLINE", "POLYLINE") and "points" in ie:
                        approx_area += _poly_area(ie.get("points") or [])
            except Exception:
                pass
            # Compute stable block hash from minimal inner entities
            try:
                try:
                    from .modules.block_hash import HASH_VERSION, compute_block_hash  # type: ignore
                except Exception:  # pragma: no cover - script run fallback
                    try:
                        from modules.block_hash import (  # type: ignore
                            HASH_VERSION,
                            compute_block_hash,
                        )
                    except Exception:  # pragma: no cover
                        from modules.block_hash import compute_block_hash  # type: ignore

                        HASH_VERSION = None  # type: ignore
                cache_key = f"{bname}|{ent_count}|{sig}"
                block_hash = _bh_cache.get(cache_key)
                if not block_hash:
                    # Use configurable params
                    cfg = get_settings()
                    q = float(getattr(cfg, "block_hash_quant_step", 0.001))
                    ena = bool(getattr(cfg, "block_hash_enable_arc_spline", True))
                    mmax = int(getattr(cfg, "block_hash_max_entities", 0))
                    pca = bool(getattr(cfg, "block_hash_pca_align", False))
                    mir = bool(getattr(cfg, "block_hash_mirror_invariant", False))
                    rs = float(getattr(cfg, "block_hash_resample_step", 0.0))
                    block_hash = compute_block_hash(
                        ents_simple,
                        quant_step=q,
                        enable_arc_spline=ena,
                        max_entities=mmax,
                        pca_align=pca,
                        mirror_invariant=mir,
                        resample_step=rs,
                    )
                    _bh_cache[cache_key] = block_hash
            except Exception:
                block_hash = None
            blocks[bname] = {
                "entity_count": ent_count,
                "types": type_counts,
                "sig": sig,
                "sig2": sig2,
                "hash": block_hash,
                "hash_version": HASH_VERSION if "HASH_VERSION" in locals() else None,
                "quant_step": getattr(get_settings(), "block_hash_quant_step", 0.001),
                "entities": ents_simple,
                "approx_area": approx_area,
            }
    except Exception:
        pass

    # ---- Simple signature cache (file-level) ----
    # Use sha256 of the DXF file to cache extracted entities + block hash results
    sig_cache_dir: Optional[Path] = None
    file_hash = None
    if use_cache:
        sig_cache_dir = (
            Path(
                getattr(get_settings(), "cache_dir", "standalone-product/dedupcad2/cache")
            )
            / "extract_sig"
        )
        try:
            sig_cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            import hashlib

            h = hashlib.sha256()
            with open(path, "rb") as fbin:
                while True:
                    chunk = fbin.read(65536)
                    if not chunk:
                        break
                    h.update(chunk)
            file_hash = h.hexdigest()
        except Exception:
            pass
    entities: List[Dict[str, Any]] = []
    if use_cache and file_hash and sig_cache_dir is not None:
        cache_file = sig_cache_dir / f"{file_hash}.json"
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                if (
                    isinstance(cached, dict)
                    and cached.get("extract_cache_version") == _EXTRACT_CACHE_VERSION
                    and "entities" in cached
                    and "blocks" in cached
                ):
                    return {
                        "file_info": {
                            "dxf_version": doc.dxfversion,
                            "cache_hit": True,
                            "insunits": insunits,
                        },
                        "layers": layers,
                        "entities": cached["entities"],
                        "blocks": cached["blocks"],
                    }
            except Exception:
                pass
    for e in msp:
        et = e.dxftype()
        layer = getattr(e.dxf, "layer", None)
        item = {"type": et, "layer": layer}
        # Minimal geometry capture (can be expanded later)
        if et == "POLYLINE" and _classic_polyline_is_non_2d(e):
            # Do not flatten Z/mesh/polyface vertices into a 2D POLYLINE.
            item["type"] = "POLYLINE3D"
        elif _dxf_primitive_non_planar(e, et):
            # Nonzero Z/elevation or a non-default extrusion is not 2D L4.
            item["type"] = "NONPLANAR"
        elif et == "LINE":
            item.update(
                {
                    "start": [float(e.dxf.start.x), float(e.dxf.start.y)],
                    "end": [float(e.dxf.end.x), float(e.dxf.end.y)],
                }
            )
        elif et == "CIRCLE":
            item.update(
                {
                    "center": [float(e.dxf.center.x), float(e.dxf.center.y)],
                    "radius": float(e.dxf.radius),
                }
            )
        elif et == "ARC":
            item.update(
                {
                    "center": [float(e.dxf.center.x), float(e.dxf.center.y)],
                    "radius": float(e.dxf.radius),
                    "start_angle": float(e.dxf.start_angle),
                    "end_angle": float(e.dxf.end_angle),
                }
            )
        elif et in ("LWPOLYLINE", "POLYLINE"):
            pts: List[List[float]] = []
            bulges: List[float] = []
            has_width = False
            try:
                # LWPOLYLINE .points() -> (x,y[,start_width,end_width,bulge])
                pts, bulges, has_width = _polyline_xy_and_bulges(e, et)
            except Exception:
                pass
            if pts:
                item.update(
                    {"points": pts, "closed": _dxf_polyline_closed(e, et)}
                )
                if any(b != 0.0 for b in bulges):
                    item["bulges"] = bulges
                if has_width:
                    item["has_width"] = True
        elif et == "ELLIPSE":
            # Represent by center and radii (approx)
            try:
                center = [float(e.dxf.center.x), float(e.dxf.center.y)]
                major = e.dxf.major_axis
                ratio = float(e.dxf.ratio)
                item.update(
                    {
                        "center": center,
                        "major": [float(major.x), float(major.y)],
                        "ratio": ratio,
                        "start_param": float(e.dxf.start_param),
                        "end_param": float(e.dxf.end_param),
                    }
                )
            except Exception:
                pass
        elif et == "SPLINE":
            # Use control points as minimal representation
            try:
                cps = [[float(p.x), float(p.y)] for p in e.control_points]
                if cps:
                    item.update({"control_points": cps, "degree": int(getattr(e.dxf, "degree", 3))})
            except Exception:
                pass
        elif et in ("TEXT", "MTEXT"):
            text = getattr(e.dxf, "text", None) or getattr(e, "text", "")
            item.update({"text": str(text)})
        elif et == "DIMENSION":
            try:
                # Some fields may be missing depending on DXF
                txt = str(getattr(e.dxf, "text", "") or "")
                dimstyle = str(getattr(e.dxf, "dimstyle", "") or "")
                # Parse numeric value (+/- tolerance) with unit; convert to mm
                import re

                unit_map = {
                    "mm": 1.0,
                    "millimeter": 1.0,
                    "cm": 10.0,
                    "m": 1000.0,
                    "in": 25.4,
                    "inch": 25.4,
                    "inches": 25.4,
                }
                # Patterns like: 12.3±0.1mm, 12.3 mm, 0.5in
                m = re.search(
                    r"([-+]?\d*\.?\d+)\s*(mm|millimeter|cm|m|in|inch|inches)?\s*(?:[±\+\-]\s*([-+]?\d*\.?\d+))?",
                    txt,
                    re.IGNORECASE,
                )
                val = tol = None
                unit = "mm"
                if m:
                    val = float(m.group(1))
                    if m.group(2):
                        unit = m.group(2).lower()
                    if m.group(3):
                        tol = float(m.group(3))
                    factor = unit_map.get(unit, 1.0)
                    val = val * factor
                    tol = tol * factor if tol is not None else None
                item.update(
                    {
                        "text": txt,
                        "dimstyle": dimstyle,
                        "value": val,
                        "tol": tol,
                        "unit": unit,
                    }
                )
            except Exception:
                pass
        elif et == "LEADER":
            try:
                verts = getattr(e, "vertices", [])
                pts = []
                for v in verts:
                    pts.append([float(getattr(v, "x", 0.0)), float(getattr(v, "y", 0.0))])
                if pts:
                    item.update({"vertices": pts})
            except Exception:
                pass
        elif et in ("HATCH",):
            try:
                pattern = str(getattr(e.dxf, "pattern_name", "") or "")
                color = int(getattr(e.dxf, "color", 0) or 0)
                item.update(
                    {
                        "pattern": pattern,
                        "color": color,
                        "loops": int(
                            getattr(e, "paths", []) and len(getattr(e, "paths").paths) or 0
                        ),
                    }
                )
            except Exception:
                pass
        elif et == "INSERT":
            try:
                name = str(getattr(e.dxf, "name", "") or "")
                inspt = getattr(e.dxf, "insert", None)
                sx = float(getattr(e.dxf, "xscale", 1.0) or 1.0)
                sy = float(getattr(e.dxf, "yscale", 1.0) or 1.0)
                rot = float(getattr(e.dxf, "rotation", 0.0) or 0.0)
                ip = (
                    [float(getattr(inspt, "x", 0.0)), float(getattr(inspt, "y", 0.0))]
                    if inspt is not None
                    else [0.0, 0.0]
                )
                item.update(
                    {
                        "block": name,
                        "insert": ip,
                        "scale": [sx, sy],
                        "rotation": rot,
                        "block_sig": (blocks.get(name) or {}).get("sig"),
                        "block_sig2": (blocks.get(name) or {}).get("sig2"),
                        "block_area": (blocks.get(name) or {}).get("approx_area"),
                        "block_hash": (blocks.get(name) or {}).get("hash"),
                        "block_types": (blocks.get(name) or {}).get("types"),
                        "block_entities": (blocks.get(name) or {}).get("entities"),
                    }
                )
            except Exception:
                pass
        entities.append(item)

    result = {
        "file_info": {
            "dxf_version": doc.dxfversion,
            "cache_hit": False,
            "insunits": insunits,
        },
        "layers": layers,
        "entities": entities,
        "blocks": blocks,
    }
    if use_cache and file_hash and sig_cache_dir is not None:
        try:
            (sig_cache_dir / f"{file_hash}.json").write_text(
                json.dumps(
                    {
                        "extract_cache_version": _EXTRACT_CACHE_VERSION,
                        "entities": entities,
                        "blocks": blocks,
                        "insunits": insunits,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass
    return result
