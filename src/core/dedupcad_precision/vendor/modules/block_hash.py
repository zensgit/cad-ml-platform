"""
Block hashing utilities for DedupCAD2.

Generates a stable hash for a block definition from its minimal entity
features. The goal is to be order- and translation-invariant, and robust
to tiny numeric noise via quantization. Rotation/scale invariance is not
enforced here; instance-level verification handles pose tolerances.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Tuple


def _collect_points(ent: Dict[str, Any]) -> List[Tuple[float, float]]:
    t = ent.get("type")
    pts: List[Tuple[float, float]] = []
    if t == "LINE":
        s = ent.get("start", [0.0, 0.0])
        e = ent.get("end", [0.0, 0.0])
        pts.extend(
            [(float(s[0] or 0.0), float(s[1] or 0.0)), (float(e[0] or 0.0), float(e[1] or 0.0))]
        )
    elif t == "CIRCLE":
        c = ent.get("center", [0.0, 0.0])
        pts.append((float(c[0] or 0.0), float(c[1] or 0.0)))
    elif t == "ARC":
        c = ent.get("center", [0.0, 0.0])
        pts.append((float(c[0] or 0.0), float(c[1] or 0.0)))
    elif t in ("LWPOLYLINE", "POLYLINE"):
        for p in ent.get("points", []) or []:
            pts.append((float(p[0] or 0.0), float(p[1] or 0.0)))
    elif t == "SPLINE":
        for p in ent.get("control_points", []) or []:
            pts.append((float(p[0] or 0.0), float(p[1] or 0.0)))
    elif t == "ELLIPSE":
        c = ent.get("center", [0.0, 0.0])
        pts.append((float(c[0] or 0.0), float(c[1] or 0.0)))
    elif t in ("TEXT", "MTEXT"):
        # no geometry points; hash will rely on text content
        pass
    return pts


def _centroid(all_pts: Iterable[Tuple[float, float]]) -> Tuple[float, float]:
    xs: float = 0.0
    ys: float = 0.0
    n = 0
    for x, y in all_pts:
        xs += float(x)
        ys += float(y)
        n += 1
    if n == 0:
        return (0.0, 0.0)
    return (xs / n, ys / n)


def _q(x: float, step: float) -> float:
    try:
        if step <= 0:
            return float(x)
        return round(float(x) / step) * step
    except Exception:
        return 0.0


def _qa(deg: float, step_deg: float = 0.1) -> float:
    try:
        a = float(deg) % 360.0
        return round(a / step_deg) * step_deg
    except Exception:
        return 0.0


def _canon_entity(ent: Dict[str, Any], cx: float, cy: float, q: float) -> str:
    t = str(ent.get("type", "?"))
    if t == "LINE":
        s = ent.get("start", [0.0, 0.0])
        e = ent.get("end", [0.0, 0.0])
        sx, sy = _q(float(s[0]) - cx, q), _q(float(s[1]) - cy, q)
        ex, ey = _q(float(e[0]) - cx, q), _q(float(e[1]) - cy, q)
        # order-invariance for the segment ends
        a = (sx, sy)
        b = (ex, ey)
        if a > b:
            a, b = b, a
        return f"L|{a[0]:.6f},{a[1]:.6f}|{b[0]:.6f},{b[1]:.6f}"
    if t == "CIRCLE":
        c = ent.get("center", [0.0, 0.0])
        r = float(ent.get("radius", 0.0) or 0.0)
        cxq, cyq = _q(float(c[0]) - cx, q), _q(float(c[1]) - cy, q)
        rq = _q(r, q)
        return f"C|{cxq:.6f},{cyq:.6f}|{rq:.6f}"
    if t == "ARC":
        c = ent.get("center", [0.0, 0.0])
        r = float(ent.get("radius", 0.0) or 0.0)
        a1 = float(ent.get("start_angle", 0.0) or 0.0)
        a2 = float(ent.get("end_angle", 0.0) or 0.0)
        cxq, cyq = _q(float(c[0]) - cx, q), _q(float(c[1]) - cy, q)
        rq = _q(r, q)
        a1q, a2q = _qa(a1), _qa(a2)
        if a1q > a2q:
            a1q, a2q = a2q, a1q
        return f"A|{cxq:.6f},{cyq:.6f}|{rq:.6f}|{a1q:.3f}-{a2q:.3f}"
    if t in ("LWPOLYLINE", "POLYLINE"):
        pts = [
            (_q(float(p[0]) - cx, q), _q(float(p[1]) - cy, q)) for p in (ent.get("points") or [])
        ]
        # Make polyline invariant to direction by choosing lexicographically smaller of (forward, reversed)
        if pts and tuple(pts[::-1]) < tuple(pts):
            pts = pts[::-1]
        pts_s = ";".join(f"{x:.6f},{y:.6f}" for x, y in pts)
        return f"P|{pts_s}"
    if t == "SPLINE":
        cps = [
            (_q(float(p[0]) - cx, q), _q(float(p[1]) - cy, q))
            for p in (ent.get("control_points") or [])
        ]
        if cps and tuple(cps[::-1]) < tuple(cps):
            cps = cps[::-1]
        cps_s = ";".join(f"{x:.6f},{y:.6f}" for x, y in cps)
        deg = int(ent.get("degree", 3) or 3)
        return f"S|d{deg}|{cps_s}"
    if t in ("TEXT", "MTEXT"):
        tx = str(ent.get("text", ""))
        tx_norm = " ".join(tx.split())[:64].lower()
        return f"T|{tx_norm}"
    if t == "ELLIPSE":
        c = ent.get("center", [0.0, 0.0])
        major = ent.get("major", [0.0, 0.0])
        ratio = float(ent.get("ratio", 0.0) or 0.0)
        cxq, cyq = _q(float(c[0]) - cx, q), _q(float(c[1]) - cy, q)
        mxq, myq = _q(float(major[0]), q), _q(float(major[1]), q)
        return f"E|{cxq:.6f},{cyq:.6f}|{mxq:.6f},{myq:.6f}|r{ratio:.3f}"
    if t == "HATCH":
        # Hash by pattern/color/loops only; geometry already represented by boundary entities
        pat = str(ent.get("pattern", "")).lower()
        col = int(ent.get("color", 0) or 0)
        loops = int(ent.get("loops", 0) or 0)
        return f"H|{pat}|c{col}|l{loops}"
    # Fallback: type only
    return f"X|{t}"


HASH_VERSION = "v1.2"


def compute_block_hash(
    block_entities: List[Dict[str, Any]],
    quant_step: float = 0.001,
    enable_arc_spline: bool = True,
    max_entities: int = 0,
    pca_align: bool = False,
    mirror_invariant: bool = False,
    resample_step: float = 0.0,
) -> str:
    """Compute a stable SHA256 hash for the block based on minimal entities.

    - Translation invariance via centroid removal
    - Optional PCA-based rotation normalization
    - Quantization by `quant_step` for numeric stability
    - Order invariance by sorting canonical entity strings
    - Optional mirror invariance by evaluating mirrored variants and picking the minimal signature
    """
    # 1) Pre-scan to determine centroid and (optionally) PCA angle
    all_pts: List[Tuple[float, float]] = []
    count = 0
    for ent in block_entities or []:
        if max_entities and count >= max_entities:
            break
        t = ent.get("type")
        if not enable_arc_spline and t in ("ARC", "SPLINE"):
            continue
        all_pts.extend(_collect_points(ent))
        count += 1
    cx, cy = _centroid(all_pts)

    cos_t, sin_t = 1.0, 0.0
    ang_deg = 0.0
    if pca_align and all_pts:
        # First principal axis of covariance for coarse rotation normalization
        mx, my = cx, cy
        sxx = syy = sxy = 0.0
        n = 0
        for x, y in all_pts:
            dx = x - mx
            dy = y - my
            sxx += dx * dx
            syy += dy * dy
            sxy += dx * dy
            n += 1
        if n > 0:
            import math

            ang = 0.5 * math.atan2(2.0 * sxy, (sxx - syy) if (sxx != syy) else 1e-9)
            cos_t = math.cos(ang)
            sin_t = math.sin(ang)
            ang_deg = ang * 180.0 / math.pi

    def _rot(x: float, y: float) -> Tuple[float, float]:
        if sin_t == 0.0 and cos_t == 1.0:
            return x, y
        rx = x * cos_t - y * sin_t
        ry = x * sin_t + y * cos_t
        return rx, ry

    def _normalize_and_canon(axis: str) -> List[str]:
        """Build canonical strings under a specific mirror axis.

        axis: 'none' | 'mx' (mirror X: y->-y) | 'my' (mirror Y: x->-x)
        """
        out: List[str] = []
        processed = 0
        for ent in block_entities or []:
            if max_entities and processed >= max_entities:
                break
            et = ent.get("type")
            if not enable_arc_spline and et in ("ARC", "SPLINE"):
                continue
            # Optional resampling for polylines (improves robustness under different vertex sampling)
            ent_work = ent
            if resample_step and resample_step > 0 and et in ("LWPOLYLINE", "POLYLINE"):
                pts0 = ent.get("points") or []
                if len(pts0) >= 2:
                    new_pts: List[Tuple[float, float]] = []
                    import math

                    acc = 0.0
                    step = float(resample_step)
                    for i in range(len(pts0) - 1):
                        x1, y1 = float(pts0[i][0]), float(pts0[i][1])
                        x2, y2 = float(pts0[i + 1][0]), float(pts0[i + 1][1])
                        dx, dy = x2 - x1, y2 - y1
                        seg = math.hypot(dx, dy)
                        if seg <= 1e-9:
                            continue
                        tcur = 0.0
                        while acc + (seg - tcur) >= step:
                            need = step - acc
                            tcur = tcur + need
                            u = tcur / seg
                            nx = x1 + u * dx
                            ny = y1 + u * dy
                            new_pts.append([nx, ny])
                            acc = 0.0
                        acc = acc + (seg - tcur)
                    if new_pts:
                        ent_work = dict(ent)
                        ent_work["points"] = new_pts

            # Translate to centroid and apply PCA rotation
            t_ent = dict(ent_work)
            if et == "LINE":
                s = ent_work.get("start", [0.0, 0.0])
                e = ent_work.get("end", [0.0, 0.0])
                sx, sy = _rot(float(s[0]) - cx, float(s[1]) - cy)
                ex, ey = _rot(float(e[0]) - cx, float(e[1]) - cy)
                t_ent["start"] = [sx, sy]
                t_ent["end"] = [ex, ey]
            elif et in ("LWPOLYLINE", "POLYLINE"):
                pts = []
                for p in ent_work.get("points") or []:
                    rx, ry = _rot(float(p[0]) - cx, float(p[1]) - cy)
                    pts.append([rx, ry])
                t_ent["points"] = pts
            elif et == "SPLINE":
                cps = []
                for p in ent_work.get("control_points") or []:
                    rx, ry = _rot(float(p[0]) - cx, float(p[1]) - cy)
                    cps.append([rx, ry])
                t_ent["control_points"] = cps
            elif et in ("CIRCLE", "ARC"):
                c = ent_work.get("center", [0.0, 0.0])
                rx, ry = _rot(float(c[0]) - cx, float(c[1]) - cy)
                t_ent["center"] = [rx, ry]
                if et == "ARC":
                    # Adjust start/end angles for PCA rotation
                    try:
                        sa = float(ent_work.get("start_angle", 0.0) or 0.0) - ang_deg
                        ea = float(ent_work.get("end_angle", 0.0) or 0.0) - ang_deg
                        while sa < 0.0:
                            sa += 360.0
                        while ea < 0.0:
                            ea += 360.0
                        while sa >= 360.0:
                            sa -= 360.0
                        while ea >= 360.0:
                            ea -= 360.0
                        t_ent["start_angle"] = sa
                        t_ent["end_angle"] = ea
                    except Exception:
                        pass

            # Apply mirror if requested (in the rotated, centered space)
            if axis in ("mx", "my"):
                et = t_ent.get("type")
                if et == "LINE":
                    s = t_ent.get("start", [0.0, 0.0])
                    e = t_ent.get("end", [0.0, 0.0])
                    if axis == "mx":
                        s = [s[0], -s[1]]
                        e = [e[0], -e[1]]
                    else:
                        s = [-s[0], s[1]]
                        e = [-e[0], e[1]]
                    t_ent["start"], t_ent["end"] = s, e
                elif et in ("LWPOLYLINE", "POLYLINE"):
                    pts = []
                    for x, y in t_ent.get("points") or []:
                        if axis == "mx":
                            pts.append([x, -y])
                        else:
                            pts.append([-x, y])
                    t_ent["points"] = pts
                elif et == "SPLINE":
                    cps = []
                    for x, y in t_ent.get("control_points") or []:
                        if axis == "mx":
                            cps.append([x, -y])
                        else:
                            cps.append([-x, y])
                    t_ent["control_points"] = cps
                elif et in ("CIRCLE", "ARC"):
                    c = t_ent.get("center", [0.0, 0.0])
                    if axis == "mx":
                        c = [c[0], -c[1]]
                    else:
                        c = [-c[0], c[1]]
                    t_ent["center"] = c
                    if et == "ARC":
                        try:
                            sa = float(t_ent.get("start_angle", 0.0) or 0.0)
                            ea = float(t_ent.get("end_angle", 0.0) or 0.0)
                            if axis == "mx":
                                # Mirror X: y -> -y, angle -> -angle
                                sa = (-sa) % 360.0
                                ea = (-ea) % 360.0
                            else:
                                # Mirror Y: x -> -x, angle -> 180 - angle
                                sa = (180.0 - sa) % 360.0
                                ea = (180.0 - ea) % 360.0
                            t_ent["start_angle"] = sa
                            t_ent["end_angle"] = ea
                        except Exception:
                            pass
                elif et == "ELLIPSE":
                    c = t_ent.get("center", [0.0, 0.0])
                    if axis == "mx":
                        c = [c[0], -c[1]]
                    else:
                        c = [-c[0], c[1]]
                    t_ent["center"] = c
                    major = t_ent.get("major", [0.0, 0.0])
                    if isinstance(major, list) and len(major) >= 2:
                        if axis == "mx":
                            major = [major[0], -float(major[1])]
                        else:
                            major = [-float(major[0]), major[1]]
                        t_ent["major"] = major

            # Canonicalize at origin with quantization
            out.append(_canon_entity(t_ent, 0.0, 0.0, quant_step))
            processed += 1

        out.sort()
        return out

    # 2) Build candidate canonical lists and pick lexicographically minimal when mirror_invariant
    variants = ["none"]
    if mirror_invariant:
        variants.extend(["mx", "my"])  # mirror over X and Y axes
    payload_lists = [_normalize_and_canon(v) for v in variants]
    canon = min(payload_lists)

    payload = ("#" + HASH_VERSION + "\n" + "\n".join(canon)).encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()
