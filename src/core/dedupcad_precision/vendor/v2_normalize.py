"""
Schema v2 normalization for DedupCAD 2.0.

Converts extractor output (or raw JSON) into a normalized structure with:
- unified units/rounding
- basic bbox and signatures (placeholders for future)
"""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from .config import Settings, get_settings
except ImportError:  # pragma: no cover - direct script run
    from config import Settings, get_settings


def _round(x: float, p: int = 3) -> float:
    try:
        return round(float(x), p)
    except Exception:
        return 0.0


def _maybe_round(x: Any, p: int = 3) -> float | None:
    if x is None:
        return None
    try:
        return round(float(x), p)
    except Exception:
        return None


def normalize_v2(data: Dict[str, Any], cfg: Settings | None = None) -> Dict[str, Any]:
    cfg = cfg or get_settings()
    layers_in = data.get("layers", {}) or {}
    entities_in = data.get("entities", []) or []

    layers: Dict[str, Dict[str, Any]] = {}
    for name, meta in layers_in.items():
        layers[str(name)] = {
            "color": meta.get("color"),
            "linetype": meta.get("linetype"),
        }

    entities: List[Dict[str, Any]] = []
    for e in entities_in:
        et = str(e.get("type", "UNKNOWN"))
        layer = str(e.get("layer", ""))
        item: Dict[str, Any] = {"type": et, "layer": layer}
        if et == "LINE":
            s = e.get("start", [0, 0])
            t = e.get("end", [0, 0])
            item.update(
                {
                    "start": [_round(s[0]), _round(s[1])],
                    "end": [_round(t[0]), _round(t[1])],
                }
            )
        elif et == "CIRCLE":
            c = e.get("center", [0, 0])
            r = e.get("radius", 0)
            item.update(
                {
                    "center": [_round(c[0]), _round(c[1])],
                    "radius": _round(r),
                }
            )
        elif et == "ARC":
            c = e.get("center", [0, 0])
            item.update(
                {
                    "center": [_round(c[0]), _round(c[1])],
                    "radius": _round(e.get("radius", 0)),
                    "start_angle": _round(e.get("start_angle", 0)),
                    "end_angle": _round(e.get("end_angle", 0)),
                }
            )
        elif et in ("LWPOLYLINE", "POLYLINE"):
            pts = e.get("points") or []
            norm_pts: List[List[float]] = []
            for p in pts:
                try:
                    if p is None or len(p) < 2:  # type: ignore[arg-type]
                        continue
                    norm_pts.append([_round(p[0]), _round(p[1])])  # type: ignore[index]
                except Exception:
                    continue
            if norm_pts:
                item["points"] = norm_pts
            if "closed" in e:
                item["closed"] = bool(e.get("closed"))
        elif et == "ELLIPSE":
            c = e.get("center", [0, 0])
            maj = e.get("major", [1, 0])
            item.update(
                {
                    "center": [_round(c[0]), _round(c[1])],
                    "major": [_round(maj[0]), _round(maj[1])],
                    "ratio": _round(e.get("ratio", 1.0)),
                    "start_param": _round(e.get("start_param", 0.0)),
                    "end_param": _round(e.get("end_param", 0.0)),
                }
            )
        elif et == "SPLINE":
            cps = e.get("control_points") or []
            norm_cps: List[List[float]] = []
            for p in cps:
                try:
                    if p is None or len(p) < 2:  # type: ignore[arg-type]
                        continue
                    norm_cps.append([_round(p[0]), _round(p[1])])  # type: ignore[index]
                except Exception:
                    continue
            if norm_cps:
                item["control_points"] = norm_cps
            try:
                item["degree"] = int(e.get("degree") or 0)
            except Exception:
                item["degree"] = 0
        elif et in ("TEXT", "MTEXT"):
            text = str(e.get("text", ""))
            if cfg.tol_text_case_insensitive:
                text = text.lower()
            if cfg.tol_text_ignore_ws:
                text = " ".join(text.split())
            item.update({"text": text})
        elif et == "DIMENSION":
            txt = str(e.get("text", "") or "")
            if cfg.tol_text_case_insensitive:
                txt = txt.lower()
            if cfg.tol_text_ignore_ws:
                txt = " ".join(txt.split())
            item.update(
                {
                    "text": txt,
                    "dimstyle": str(e.get("dimstyle", "") or ""),
                    "value": _maybe_round(e.get("value")),
                    "tol": _maybe_round(e.get("tol")),
                    "unit": str(e.get("unit", "") or ""),
                }
            )
        elif et == "HATCH":
            item.update(
                {
                    "pattern": str(e.get("pattern", "") or ""),
                    "loops": int(e.get("loops") or 0),
                    "color": int(e.get("color") or 0),
                }
            )
        elif et == "LEADER":
            pts = e.get("vertices") or []
            norm_pts2: List[List[float]] = []
            for p in pts:
                try:
                    if p is None or len(p) < 2:  # type: ignore[arg-type]
                        continue
                    norm_pts2.append([_round(p[0]), _round(p[1])])  # type: ignore[index]
                except Exception:
                    continue
            if norm_pts2:
                item["vertices"] = norm_pts2
        elif et == "INSERT":
            # Preserve block hash and pose for blocks scoring
            name = str(e.get("block", ""))
            ip = e.get("insert", [0.0, 0.0])
            sc = e.get("scale", [1.0, 1.0])
            rot = float(e.get("rotation", 0.0) or 0.0)
            item.update(
                {
                    "block": name,
                    "insert": [_round(ip[0]), _round(ip[1])],
                    "scale": [float(sc[0] or 1.0), float(sc[1] or 1.0)],
                    "rotation": _round(rot),
                    "block_hash": e.get("block_hash"),
                    # Attach block signature if available to enable near-hash equivalence
                    "block_sig": e.get("block_sig"),
                    "block_sig2": e.get("block_sig2"),
                    # Carry approximate area if present (for area-weighted Jaccard)
                    "block_area": e.get("block_area"),
                }
            )
        # Other entities pass-through minimally
        entities.append(item)

    # Normalize blocks map: keep hash and counts
    blocks_in = data.get("blocks", {}) or {}
    blocks_out: Dict[str, Any] = {}
    for name, meta in blocks_in.items():
        blocks_out[str(name)] = {
            "entity_count": meta.get("entity_count"),
            "sig": meta.get("sig"),
            "sig2": meta.get("sig2"),
            "hash": meta.get("hash"),
            # optional metadata for traceability
            "hash_version": meta.get("hash_version"),
            "quant_step": meta.get("quant_step"),
        }

    # Backfill INSERT.block_sig from blocks map when possible
    if blocks_out and entities:
        for e in entities:
            if e.get("type") == "INSERT":
                bname = e.get("block")
                if bname and bname in blocks_out:
                    if e.get("block_sig") is None:
                        e["block_sig"] = blocks_out[bname].get("sig")
                    if e.get("block_sig2") is None:
                        e["block_sig2"] = blocks_out[bname].get("sig2")

    return {
        "layers": layers,
        "entities": entities,
        # Placeholders for future weighted scoring sections
        "dimensions": data.get("dimensions", {}),
        "hatches": data.get("hatches", {}),
        "blocks": blocks_out,
        "text_content": data.get("text_content", []),
        "statistics": data.get("statistics", {}),
    }
