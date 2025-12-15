"""
DXF extractor for DedupCAD 2.0 standalone.
Relies on ezdxf to parse DXF and produce a simple JSON structure.
"""

import json
from pathlib import Path  # ensure Path available for cache directory logic
from typing import Any, Dict, List

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


def extract_dxf(path: str) -> Dict[str, Any]:
    if ezdxf is None:
        raise RuntimeError("ezdxf not installed: pip install ezdxf")
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()

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
                    if t == "LINE":
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
                        try:
                            if t == "LWPOLYLINE":
                                for p in be.get_points():
                                    pts.append([float(p[0]), float(p[1])])
                            else:
                                for v in be.vertices:
                                    pts.append([float(v.dxf.location.x), float(v.dxf.location.y)])
                        except Exception:
                            pass
                        if pts:
                            ie.update({"points": pts})
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
    sig_cache_dir = (
        Path(getattr(get_settings(), "cache_dir", "standalone-product/dedupcad2/cache"))
        / "extract_sig"
    )
    try:
        sig_cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    file_hash = None
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
    if file_hash:
        cache_file = sig_cache_dir / f"{file_hash}.json"
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                if isinstance(cached, dict) and "entities" in cached and "blocks" in cached:
                    return {
                        "file_info": {
                            "dxf_version": doc.dxfversion,
                            "cache_hit": True,
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
        if et == "LINE":
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
            try:
                # LWPOLYLINE .points() -> (x,y[,start_width,end_width,bulge])
                if et == "LWPOLYLINE":
                    for p in e.get_points():
                        pts.append([float(p[0]), float(p[1])])
                else:
                    for v in e.vertices:
                        pts.append([float(v.dxf.location.x), float(v.dxf.location.y)])
            except Exception:
                pass
            if pts:
                item.update({"points": pts, "closed": bool(getattr(e, "closed", False))})
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
        },
        "layers": layers,
        "entities": entities,
        "blocks": blocks,
    }
    if file_hash:
        try:
            (sig_cache_dir / f"{file_hash}.json").write_text(
                json.dumps({"entities": entities, "blocks": blocks}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass
    return result
