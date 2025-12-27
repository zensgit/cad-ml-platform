"""
Advanced entities matching (Hungarian) with geometric/text costs.

This is a lightweight implementation intended for improving the
entities section similarity beyond raw JSON flattening. For speed,
it uses a simple O(n^3) Hungarian algorithm on small subsets.
"""

from __future__ import annotations

import datetime
import math
from typing import Any, Dict, List, Tuple

try:
    from .config import Settings
except ImportError:  # pragma: no cover
    from config import Settings

import statistics
from functools import lru_cache


def _dist2(p: List[float], q: List[float]) -> float:
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def _deg_delta(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _polyline_length(pts: List[List[float]]) -> float:
    d = 0.0
    for i in range(1, len(pts)):
        d += math.dist(pts[i - 1], pts[i])
    return d


def _rdp(points: List[List[float]], eps: float) -> List[List[float]]:
    """Douglas–Peucker simplification for polyline points."""
    if eps <= 0.0 or len(points) < 3:
        return points

    def perp_dist(p, a, b):
        ax, ay = a
        bx, by = b
        px, py = p
        dx, dy = bx - ax, by - ay
        if dx == 0 and dy == 0:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
        cx, cy = ax + t * dx, ay + t * dy
        return math.hypot(px - cx, py - cy)

    dmax, idx = 0.0, 0
    for i in range(1, len(points) - 1):
        d = perp_dist(points[i], points[0], points[-1])
        if d > dmax:
            dmax, idx = d, i
    if dmax > eps:
        left = _rdp(points[: idx + 1], eps)
        right = _rdp(points[idx:], eps)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]


def _polyline_sample(pts: List[List[float]], k: int = 32, eps: float = 0.0) -> List[List[float]]:
    """RDP simplify then uniform down-sample to k points."""
    if not pts:
        return []
    if eps > 0.0:
        pts = _rdp(pts, eps)
    if len(pts) <= k:
        return pts
    step = (len(pts) - 1) / (k - 1)
    out = []
    for i in range(k):
        idx = int(round(i * step))
        out.append(pts[idx])
    return out


def _flip_180(pts: List[List[float]]) -> List[List[float]]:
    """Flip by 180 degrees around origin (x,y)->(-x,-y) in canonical space."""
    return [[-p[0], -p[1]] for p in pts]


def _procrustes_adjust(
    A: List[List[float]], B: List[List[float]], max_deg: float, steps: int
) -> float:
    """Try small rotation adjustments on B to minimize symmetric Hausdorff distance.

    Returns the best symmetric Hausdorff distance after rotating B in [-max_deg, max_deg].
    """
    if not A or not B:
        return 1.0
    max_deg = float(max_deg)
    steps = max(1, int(steps))

    def rot(pts, ang):
        ca, sa = math.cos(ang), math.sin(ang)
        return [[p[0] * ca + p[1] * sa, -p[0] * sa + p[1] * ca] for p in pts]

    def haus_sym(X, Y):
        def mind(a, BB):
            return min(math.dist(a, b) for b in BB) if BB else 1.0

        d1 = max(mind(a, Y) for a in X) if X else 1.0
        d2 = max(mind(b, X) for b in Y) if Y else 1.0
        return max(d1, d2)

    best = float("inf")
    rad = math.radians(max_deg)
    for i in range(-steps, steps + 1):
        ang = (rad * i) / steps
        Br = rot(B, ang)
        d = haus_sym(A, Br)
        if d < best:
            best = d
    return best


def _cost(e1: Dict[str, Any], e2: Dict[str, Any], cfg: Settings) -> float:
    t1, t2 = e1.get("type"), e2.get("type")
    if t1 != t2:
        return 1.0
    layer_penalty = 0.0 if e1.get("layer") == e2.get("layer") else cfg.layer_mismatch_penalty
    t = t1
    if t == "LINE":
        s1, e1p = e1.get("start", [0, 0]), e1.get("end", [0, 0])
        s2, e2p = e2.get("start", [0, 0]), e2.get("end", [0, 0])
        d = math.sqrt(min(_dist2(s1, s2) + _dist2(e1p, e2p), _dist2(s1, e2p) + _dist2(e1p, s2)))
        return min(1.0, d / max(cfg.tol_line_pos, 1e-9) + layer_penalty)
    if t == "CIRCLE":
        c1, r1 = e1.get("center", [0, 0]), float(e1.get("radius", 0))
        c2, r2 = e2.get("center", [0, 0]), float(e2.get("radius", 0))
        dc = math.sqrt(_dist2(c1, c2)) / max(cfg.tol_circle_center, 1e-9)
        dr = abs(r1 - r2) / max(cfg.tol_circle_radius, 1e-9)
        return min(1.0, dc + dr + layer_penalty)
    if t == "ARC":
        c1, r1, a1, b1 = (
            e1.get("center", [0, 0]),
            float(e1.get("radius", 0)),
            float(e1.get("start_angle", 0)),
            float(e1.get("end_angle", 0)),
        )
        c2, r2, a2, b2 = (
            e2.get("center", [0, 0]),
            float(e2.get("radius", 0)),
            float(e2.get("start_angle", 0)),
            float(e2.get("end_angle", 0)),
        )
        dc = math.sqrt(_dist2(c1, c2)) / max(cfg.tol_circle_center, 1e-9)
        dr = abs(r1 - r2) / max(cfg.tol_circle_radius, 1e-9)
        da = _deg_delta(a1, a2) / cfg.tol_arc_angle_deg + _deg_delta(b1, b2) / cfg.tol_arc_angle_deg
        return min(1.0, dc + dr + da + layer_penalty)
    if t in ("TEXT", "MTEXT"):
        tx1 = str(e1.get("text", "") or "").strip()
        tx2 = str(e2.get("text", "") or "").strip()
        if getattr(cfg, "tol_text_case_insensitive", True):
            tx1 = tx1.lower()
            tx2 = tx2.lower()
        if getattr(cfg, "tol_text_ignore_ws", True):
            import re

            tx1 = re.sub(r"\s+", " ", tx1)
            tx2 = re.sub(r"\s+", " ", tx2)
        if tx1 == tx2:
            return 0.0
        if getattr(cfg, "text_fuzzy_enable", True) or getattr(
            cfg, "text_fuzzy_shadow_collect", False
        ):
            # Normalized Levenshtein distance
            def _lev(a: str, b: str) -> float:
                if a == b:
                    return 0.0
                la, lb = len(a), len(b)
                if la == 0 or lb == 0:
                    return float(max(la, lb))
                # DP row optimization
                prev = list(range(lb + 1))
                for i in range(1, la + 1):
                    cur = [i] + [0] * lb
                    for j in range(1, lb + 1):
                        cost = 0 if a[i - 1] == b[j - 1] else 1
                        cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
                    prev = cur
                return prev[lb]

            dist = _lev(tx1, tx2)
            norm = dist / float(max(len(tx1), len(tx2), 1))
            if getattr(cfg, "text_fuzzy_enable", True) and norm <= getattr(
                cfg, "text_fuzzy_threshold", 0.12
            ):
                return 0.0
            # Shadow collection: attach debug distance for later stats if enabled
            if getattr(cfg, "text_fuzzy_shadow_collect", False):
                # Store on entity objects (non-persistent) for external aggregation
                try:
                    e1.setdefault("_shadow_text_fuzzy", []).append(norm)
                    e2.setdefault("_shadow_text_fuzzy", []).append(norm)
                except Exception:
                    pass
        # Semantic embedding similarity (optional)
        if getattr(cfg, "text_embed_enable", False):
            try:
                from .app import (  # type: ignore
                    EMBED_HEAVY_CALLS,
                    EMBED_HEAVY_SIM,
                    EMBED_LIGHT_CALLS,
                    EMBED_LIGHT_SIM,
                )
                from .text_embed import cosine, get_embedding, heavy_embed_batch
            except Exception:
                try:
                    from text_embed import cosine, get_embedding, heavy_embed_batch  # type: ignore

                    EMBED_LIGHT_SIM = EMBED_LIGHT_CALLS = EMBED_HEAVY_SIM = EMBED_HEAVY_CALLS = None  # type: ignore
                except Exception:
                    get_embedding = cosine = heavy_embed_batch = None  # type: ignore
            if get_embedding and cosine:
                # Early stop: configurable short text penalty
                if min(len(tx1), len(tx2)) < getattr(cfg, "text_short_len", 3):
                    return min(1.0, getattr(cfg, "text_short_penalty", 1.0) + layer_penalty)
                emb1 = get_embedding(tx1)
                emb2 = get_embedding(tx2)
                sim = cosine(emb1, emb2)
                try:
                    from .app import _STATS  # type: ignore

                    bucket = f"{(int(sim*20)/20):.2f}"  # 0.05 bins
                    hist = _STATS["embed"].setdefault("hist_light", {})
                    hist[bucket] = hist.get(bucket, 0) + 1
                except Exception:
                    pass
                try:
                    from .app import _STATS  # type: ignore

                    _STATS["embed"]["light_calls"] += 1
                    _STATS["embed"]["light_similarity_sum"] += sim
                    _STATS["embed"]["light_similarity_count"] += 1
                    if EMBED_LIGHT_SIM is not None:
                        EMBED_LIGHT_SIM.observe(sim)
                    if EMBED_LIGHT_CALLS is not None:
                        EMBED_LIGHT_CALLS.inc()
                except Exception:
                    pass
                if sim >= getattr(cfg, "text_embed_similarity_threshold", 0.85):
                    return 0.0
                if (
                    getattr(cfg, "text_embed_heavy_enable", False)
                    and heavy_embed_batch
                    and sim < getattr(cfg, "text_embed_heavy_threshold", 0.90)
                ):
                    if not hasattr(cfg, "_heavy_text_embeddings") or cfg._heavy_text_embeddings is None:  # type: ignore
                        try:
                            # Concurrency guard (double-checked locking)
                            import threading

                            lock = getattr(cfg, "_heavy_text_lock", None)
                            if lock is None:
                                lock = threading.Lock()
                                setattr(cfg, "_heavy_text_lock", lock)
                            if lock.acquire(blocking=False):
                                try:
                                    if (
                                        not hasattr(cfg, "_heavy_text_embeddings")
                                        or cfg._heavy_text_embeddings is None
                                    ):  # recheck after lock
                                        texts_unique = []
                                        seen = set()
                                        for ent in getattr(cfg, "_entities_cache_A", []) + getattr(cfg, "_entities_cache_B", []):  # type: ignore
                                            if ent.get("type") in ("TEXT", "MTEXT"):
                                                tval = str(ent.get("text", "")).strip()
                                                if tval and tval not in seen:
                                                    seen.add(tval)
                                                    texts_unique.append(tval)
                                        vecs = heavy_embed_batch(texts_unique)
                                        cfg._heavy_text_embeddings = {t: v for t, v in zip(texts_unique, vecs)}  # type: ignore
                                        # Optional memory clear after recording stats if batch too large
                                        try:
                                            th_clear = int(
                                                getattr(cfg, "heavy_batch_clear_threshold", 0) or 0
                                            )
                                            if th_clear > 0 and len(texts_unique) > th_clear:
                                                # Keep map for this call then schedule clear
                                                cfg._heavy_text_embeddings = None  # type: ignore
                                        except Exception:
                                            pass
                                        # Record batch size last + histogram bucket
                                        try:
                                            from .app import _STATS  # type: ignore

                                            size = len(texts_unique)
                                            _STATS["embed"]["heavy_batch_last_size"] = size
                                            # Memory estimate (vector length from first embedding, assume all same)
                                            try:
                                                dim = len(vecs[0]) if vecs else 0
                                                bytes_est = size * dim * 4  # float32
                                                _STATS["embed"]["heavy_mem_bytes"] = bytes_est
                                                # Prometheus counter (monotonic) records last observed value
                                                try:
                                                    from .app import (
                                                        EMBED_HEAVY_MEM_BYTES,  # type: ignore
                                                    )

                                                    if EMBED_HEAVY_MEM_BYTES is not None:
                                                        EMBED_HEAVY_MEM_BYTES.inc(bytes_est)
                                                except Exception:
                                                    pass
                                                warn_mb = int(
                                                    getattr(cfg, "heavy_mem_warn_mb", 0) or 0
                                                )
                                                if (
                                                    warn_mb > 0
                                                    and bytes_est / (1024 * 1024) > warn_mb
                                                ):
                                                    _STATS["recent_errors"].append(
                                                        {
                                                            "timestamp": datetime.datetime.utcnow().isoformat()
                                                            + "Z",
                                                            "endpoint": "heavy_batch",
                                                            "error": f"heavy_mem_bytes>{warn_mb}MB",
                                                            "request_id": None,
                                                        }
                                                    )
                                                    _STATS["recent_errors"] = _STATS[
                                                        "recent_errors"
                                                    ][-20:]
                                            except Exception:
                                                pass
                                            b = 1
                                            buckets = [1, 2, 4, 8, 16, 32, 64, 128]
                                            for th in buckets:
                                                if size <= th:
                                                    b = th
                                                    break
                                            hist_b = _STATS["embed"].setdefault(
                                                "heavy_batch_size_hist", {}
                                            )
                                            key = (
                                                f"<= {b}"
                                                if size <= buckets[-1]
                                                else f"> {buckets[-1]}"
                                            )
                                            hist_b[key] = hist_b.get(key, 0) + 1
                                            # Efficiency metric approximate benefit over light similarity
                                            try:
                                                if (
                                                    size > 0
                                                    and _STATS["embed"].get(
                                                        "light_similarity_count", 0
                                                    )
                                                    > 0
                                                ):
                                                    h_avg = (
                                                        (
                                                            _STATS["embed"].get(
                                                                "heavy_similarity_sum", 0.0
                                                            )
                                                            / max(
                                                                1,
                                                                _STATS["embed"].get(
                                                                    "heavy_similarity_count", 0
                                                                ),
                                                            )
                                                        )
                                                        if _STATS["embed"].get(
                                                            "heavy_similarity_count", 0
                                                        )
                                                        > 0
                                                        else 0.0
                                                    )
                                                    l_avg = _STATS["embed"].get(
                                                        "light_similarity_sum", 0.0
                                                    ) / max(
                                                        1,
                                                        _STATS["embed"].get(
                                                            "light_similarity_count", 0
                                                        ),
                                                    )
                                                    _STATS["embed"]["heavy_efficiency"] = (
                                                        h_avg - l_avg
                                                    )
                                            except Exception:
                                                pass
                                            # Clear counter
                                            if th_clear > 0 and size > th_clear:
                                                _STATS["embed"]["heavy_batch_clears"] = (
                                                    _STATS["embed"].get("heavy_batch_clears", 0) + 1
                                                )
                                        except Exception:
                                            pass
                                finally:
                                    try:
                                        lock.release()
                                    except Exception:
                                        pass
                            # If lock not acquired, another thread is building; fall through to use existing map later
                            else:
                                try:
                                    from .app import _STATS  # type: ignore

                                    _STATS["embed"]["heavy_lock_miss"] = (
                                        _STATS["embed"].get("heavy_lock_miss", 0) + 1
                                    )
                                except Exception:
                                    pass
                            texts_unique = []
                        except Exception:
                            cfg._heavy_text_embeddings = {}  # type: ignore
                    hv_map = getattr(cfg, "_heavy_text_embeddings", {})  # type: ignore
                    v1 = hv_map.get(tx1)
                    v2 = hv_map.get(tx2)
                    if v1 and v2:
                        h_sim = cosine(v1, v2)
                        try:
                            bucket_h = f"{(int(h_sim*20)/20):.2f}"
                            hist_h = _STATS["embed"].setdefault("hist_heavy", {})
                            hist_h[bucket_h] = hist_h.get(bucket_h, 0) + 1
                        except Exception:
                            pass
                        try:
                            from .app import _STATS  # type: ignore

                            _STATS["embed"]["heavy_calls"] += 1
                            _STATS["embed"]["heavy_similarity_sum"] += h_sim
                            _STATS["embed"]["heavy_similarity_count"] += 1
                            if EMBED_HEAVY_SIM is not None:
                                EMBED_HEAVY_SIM.observe(h_sim)
                            if EMBED_HEAVY_CALLS is not None:
                                EMBED_HEAVY_CALLS.inc()
                        except Exception:
                            pass
                        if h_sim >= getattr(cfg, "text_embed_heavy_threshold", 0.90):
                            return 0.0
                        if h_sim > sim:
                            # Record efficiency gain histogram bucket (heavy - light)
                            try:
                                from .app import _STATS  # type: ignore

                                gain = max(0.0, h_sim - sim)
                                b = f"{(int(gain*100)/100):.2f}"  # 0.01 bins
                                eff_hist = _STATS["embed"].setdefault("heavy_eff_hist", {})
                                eff_hist[b] = eff_hist.get(b, 0) + 1
                            except Exception:
                                pass
                            sim = h_sim
                return min(1.0, (1.0 - sim) + layer_penalty)
        return 1.0
    if t == "DIMENSION":
        v1 = e1.get("value")
        v2 = e2.get("value")
        if v1 is None and v2 is None:
            # Fallback: compare displayed text when numeric value extraction failed on both sides.
            tx1 = str(e1.get("text", "") or "").strip()
            tx2 = str(e2.get("text", "") or "").strip()
            if getattr(cfg, "tol_text_case_insensitive", True):
                tx1 = tx1.lower()
                tx2 = tx2.lower()
            if getattr(cfg, "tol_text_ignore_ws", True):
                import re

                tx1 = re.sub(r"\s+", " ", tx1)
                tx2 = re.sub(r"\s+", " ", tx2)
            if tx1 == tx2:
                return min(1.0, layer_penalty)
            return 1.0
        if v1 is None or v2 is None:
            return 1.0
        dv = abs(float(v1) - float(v2)) / max(1e-9, cfg.tol_dimension_value)
        # consider tolerance if both present
        t1 = e1.get("tol")
        t2 = e2.get("tol")
        if t1 is not None and t2 is not None:
            dv = max(0.0, dv - min(float(t1), float(t2)) / max(1e-9, cfg.tol_dimension_value))
        return min(1.0, dv + layer_penalty)
    if t == "HATCH":
        pat1 = str(e1.get("pattern", "") or "")
        pat2 = str(e2.get("pattern", "") or "")
        loops1 = int(e1.get("loops") or 0)
        loops2 = int(e2.get("loops") or 0)
        pen = 0.0 if pat1 == pat2 else cfg.hatch_pattern_penalty
        dl = abs(loops1 - loops2) / max(cfg.tol_hatch_loops, 1e-9)
        return min(1.0, pen + dl + layer_penalty)
    if t in ("LWPOLYLINE", "POLYLINE"):
        p1 = e1.get("points") or []
        p2 = e2.get("points") or []
        if not p1 or not p2:
            return 1.0
        # simplify and sample
        s1 = _polyline_sample(p1, k=32, eps=cfg.rdp_eps if getattr(cfg, "use_rdp", True) else 0.0)
        s2 = _polyline_sample(p2, k=32, eps=cfg.rdp_eps if getattr(cfg, "use_rdp", True) else 0.0)
        c1, o1, sc1, th1 = _fit_canonical(s1)
        c2, o2, sc2, th2 = _fit_canonical(s2)

        # approximate distances (Hausdorff or discrete Fréchet)
        def hausdorff(A: List[List[float]], B: List[List[float]]) -> float:
            def mind(pt, BB):
                return min(math.dist(pt, b) for b in BB) if BB else 1.0

            return max((mind(pt, B) for pt in A), default=0.0) if A else 1.0

        def frechet(A: List[List[float]], B: List[List[float]]) -> float:
            if not A or not B:
                return 1.0
            n, m = len(A), len(B)
            ca = [[-1.0] * m for _ in range(n)]

            def dist(i, j):
                return math.dist(A[i], B[j])

            def rec(i, j):
                if ca[i][j] > -0.5:
                    return ca[i][j]
                if i == 0 and j == 0:
                    ca[i][j] = dist(0, 0)
                elif i == 0 and j > 0:
                    ca[i][j] = max(rec(0, j - 1), dist(0, j))
                elif i > 0 and j == 0:
                    ca[i][j] = max(rec(i - 1, 0), dist(i, 0))
                else:
                    ca[i][j] = max(min(rec(i - 1, j), rec(i - 1, j - 1), rec(i, j - 1)), dist(i, j))
                return ca[i][j]

            return rec(n - 1, m - 1)

        if getattr(cfg, "use_frechet", True):
            # resample to configured size for frechet
            s1f = _polyline_sample(c1, k=getattr(cfg, "frechet_samples", 64))
            s2f = _polyline_sample(c2, k=getattr(cfg, "frechet_samples", 64))
            d1 = frechet(s1f, s2f)
            d2 = frechet(s1f, _flip_180(s2f))
            d = min(d1, d2)
        else:
            if getattr(cfg, "use_procrustes", True):
                d = min(
                    _procrustes_adjust(
                        c1,
                        c2,
                        getattr(cfg, "procrustes_deg", 5.0),
                        getattr(cfg, "procrustes_steps", 3),
                    ),
                    _procrustes_adjust(
                        c1,
                        _flip_180(c2),
                        getattr(cfg, "procrustes_deg", 5.0),
                        getattr(cfg, "procrustes_steps", 3),
                    ),
                )
            else:
                d1 = max(hausdorff(c1, c2), hausdorff(c2, c1))
                d2 = max(hausdorff(c1, _flip_180(c2)), hausdorff(_flip_180(c2), c1))
                d = min(d1, d2)
        dl = abs(_polyline_length(p1) - _polyline_length(p2)) / max(cfg.tol_polyline_len, 1e-9)
        return min(1.0, d + dl + layer_penalty)
    if t == "SPLINE":
        c1 = e1.get("control_points") or []
        c2 = e2.get("control_points") or []
        if not c1 or not c2:
            return 1.0
        k = min(len(c1), len(c2), 16)
        acc = 0.0
        for i in range(k):
            acc += math.dist(c1[i], c2[i])
        acc /= k
        return min(1.0, acc / max(cfg.tol_spline_ctrl, 1e-9) + layer_penalty)
    if t == "INSERT":
        ip1 = e1.get("insert") or [0, 0]
        ip2 = e2.get("insert") or [0, 0]
        sc1 = e1.get("scale") or [1, 1]
        sc2 = e2.get("scale") or [1, 1]
        rot1 = float(e1.get("rotation", 0.0) or 0.0)
        rot2 = float(e2.get("rotation", 0.0) or 0.0)
        bn1 = str(e1.get("block", "") or "")
        bn2 = str(e2.get("block", "") or "")
        if bn1 != bn2:
            return 1.0
        # block content signature quick check (if available)
        sig1 = e1.get("block_sig")
        sig2 = e2.get("block_sig")
        if sig1 and sig2 and sig1 != sig2:
            return 1.0
        # type distribution distance (light-weight inner check)
        types1 = e1.get("block_types") or {}
        types2 = e2.get("block_types") or {}
        if types1 and types2:
            keys = set(types1.keys()) | set(types2.keys())
            sum1 = sum(types1.values()) or 1
            sum2 = sum(types2.values()) or 1
            diff = 0.0
            for k in keys:
                diff += abs((types1.get(k, 0) / sum1) - (types2.get(k, 0) / sum2))
            # clamp contribution
            layer_penalty += min(0.5, diff)
        # optional inner matching: compare simplified inner entities in local space
        if getattr(cfg, "block_inner_match", True):
            inner1 = e1.get("block_entities") or []
            inner2 = e2.get("block_entities") or []
            if inner1 and inner2:
                # reuse polyline/line/text/circle costs by constructing temp v2
                v2a = {"entities": inner1}
                v2b = {"entities": inner2}
                sim_inner = entities_similarity(v2a, v2b, cfg)
                # convert to penalty (1-sim_inner) scaled
                layer_penalty += min(0.5, 1.0 - sim_inner)
        dp = math.dist(ip1, ip2) / max(cfg.tol_insert_pos, 1e-9)
        ds = abs(sc1[0] - sc2[0]) + abs(sc1[1] - sc2[1])
        ds /= max(cfg.tol_insert_scale, 1e-9)
        dr = _deg_delta(rot1, rot2) / max(cfg.tol_insert_rot_deg, 1e-9)
        return min(1.0, dp + ds + dr + layer_penalty)
    if t == "ELLIPSE" and getattr(cfg, "enable_entity_ellipse", True):
        c1 = e1.get("center", [0, 0])
        c2 = e2.get("center", [0, 0])
        maj1 = e1.get("major", [1, 0])
        maj2 = e2.get("major", [1, 0])
        r1 = float(e1.get("ratio", 1.0) or 1.0)
        r2 = float(e2.get("ratio", 1.0) or 1.0)
        sp1 = float(e1.get("start_param", 0.0) or 0.0)
        sp2 = float(e2.get("start_param", 0.0) or 0.0)
        ep1 = float(e1.get("end_param", 0.0) or 0.0)
        ep2 = float(e2.get("end_param", 0.0) or 0.0)
        dc = math.sqrt(_dist2(c1, c2)) / max(cfg.tol_ellipse_center, 1e-9)

        # Major axis length + angle
        def _len(v):
            return math.hypot(float(v[0]), float(v[1]))

        L1 = _len(maj1)
        L2 = _len(maj2)
        dL = abs(L1 - L2) / max(cfg.tol_ellipse_major_len, 1e-9)
        ang1 = math.degrees(math.atan2(float(maj1[1]), float(maj1[0]) or 1e-9)) % 180.0
        ang2 = math.degrees(math.atan2(float(maj2[1]), float(maj2[0]) or 1e-9)) % 180.0
        dang = _deg_delta(ang1, ang2) / max(cfg.tol_ellipse_major_angle_deg, 1e-9)
        dratio = abs(r1 - r2) / max(cfg.tol_ellipse_ratio, 1e-9)
        # Parameter span difference (capture trimmed ellipse arcs)
        span1 = abs(ep1 - sp1)
        span2 = abs(ep2 - sp2)
        dspan = abs(span1 - span2) / max(cfg.tol_ellipse_param, 1e-9)
        return min(1.0, dc + dL + dang + dratio + dspan + layer_penalty)
    if t == "LEADER" and getattr(cfg, "enable_entity_leader", True):
        pts1 = e1.get("vertices") or []
        pts2 = e2.get("vertices") or []
        if not pts1 or not pts2:
            return 1.0
        # Compare first/last anchor positions + polyline length difference
        pA1, pZ1 = pts1[0], pts1[-1]
        pA2, pZ2 = pts2[0], pts2[-1]
        d_head = math.sqrt(_dist2(pA1, pA2)) / max(cfg.tol_leader_pos, 1e-9)
        d_tail = math.sqrt(_dist2(pZ1, pZ2)) / max(cfg.tol_leader_pos, 1e-9)

        def _plen(P):
            return sum(math.dist(P[i - 1], P[i]) for i in range(1, len(P)))

        len1 = _plen(pts1)
        len2 = _plen(pts2)
        d_len = abs(len1 - len2) / max(cfg.tol_leader_len, 1e-9)
        # Optional simplified shape comparison via sampled points
        s1 = _polyline_sample(pts1, k=16)
        s2 = _polyline_sample(pts2, k=16)

        # Align via translation of first point
        def _norm(P, ref):
            return [[p[0] - ref[0], p[1] - ref[1]] for p in P]

        n1 = _norm(s1, s1[0])
        n2 = _norm(s2, s2[0])

        # Hausdorff distance scaled
        def _haus(A, B):
            if not A or not B:
                return 1.0

            def md(a, BB):
                return min(math.dist(a, b) for b in BB)

            return max(max(md(a, B) for a in A), max(md(b, A) for b in B))

        d_shape = _haus(n1, n2) / max(cfg.tol_leader_pos, 1e-9)
        return min(1.0, d_head + d_tail + d_len + d_shape + layer_penalty)
    # fallback
    return 1.0


def _hungarian(cost: List[List[float]]) -> Tuple[List[int], float]:
    """Classic Hungarian algorithm (O(n^3)) for square matrices.
    Returns (assignment, total_cost).
    """
    n = len(cost)
    if n == 0:
        return [], 0.0
    # Convert to mutable copy
    a = [row[:] for row in cost]
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = a[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assign = [-1] * n
    for j in range(1, n + 1):
        if p[j] > 0:
            assign[p[j] - 1] = j - 1
    total = sum(cost[i][assign[i]] for i in range(n) if assign[i] != -1)
    return assign, total


def _salience(e: Dict[str, Any], cfg: Settings) -> float:
    t = e.get("type", "")
    return {
        "LINE": cfg.sal_w_line,
        "CIRCLE": cfg.sal_w_circle,
        "ARC": cfg.sal_w_arc,
        "LWPOLYLINE": cfg.sal_w_polyline,
        "POLYLINE": cfg.sal_w_polyline,
        "TEXT": cfg.sal_w_text,
        "MTEXT": cfg.sal_w_text,
        "INSERT": cfg.sal_w_insert,
        "ELLIPSE": getattr(cfg, "sal_w_ellipse", 1.0),
        "LEADER": getattr(cfg, "sal_w_leader", 1.0),
    }.get(t, 1.0)


def entities_similarity(left_v2: Dict[str, Any], right_v2: Dict[str, Any], cfg: Settings) -> float:
    A = sorted(list(left_v2.get("entities") or []), key=lambda e: _salience(e, cfg), reverse=True)
    B = sorted(list(right_v2.get("entities") or []), key=lambda e: _salience(e, cfg), reverse=True)
    # Cache entity lists for heavy batch reuse; rebuild only if set changes.
    try:
        prev_sig = getattr(cfg, "_heavy_entities_sig", None)
        # Signature: sizes + sha256 of sorted unique text values for TEXT/MTEXT
        import hashlib

        def _text_sig(lst):
            vals = []
            for e in lst:
                if e.get("type") in ("TEXT", "MTEXT"):
                    t = str(e.get("text", "")).strip()
                    if t:
                        vals.append(t)
            if not vals:
                return "0:"
            s_vals = sorted(set(vals))
            h = hashlib.sha256(
                ("\u0001".join(s_vals)).encode("utf-8", errors="ignore")
            ).hexdigest()[:16]
            return f"{len(s_vals)}:{h}"

        sig = f"{len(A)}|{len(B)}|{_text_sig(A)}|{_text_sig(B)}"
        cfg._entities_cache_A = A  # type: ignore
        cfg._entities_cache_B = B  # type: ignore
        if prev_sig != sig:
            setattr(cfg, "_heavy_entities_sig", sig)  # type: ignore
            setattr(cfg, "_heavy_text_embeddings", None)  # invalidate for rebuild
    except Exception:
        pass
    if not A and not B:
        return 1.0
    n = min(len(A), len(B), max(1, cfg.max_match_entities))  # cap for performance
    A = A[:n]
    B = B[:n]
    # Square cost matrix by padding with max cost
    size = max(len(A), len(B))
    padA = A + [{}] * (size - len(A))
    padB = B + [{}] * (size - len(B))
    cost = [[1.0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if i < len(A) and j < len(B):
                cost[i][j] = _cost(padA[i], padB[j], cfg)
    # Guard against very large matrices
    if cfg.use_hungarian and size <= 128:
        assign, total = _hungarian(cost)
    else:
        # Fallback greedy
        used_j = set()
        assign = [-1] * size
        total = 0.0
        for i in range(size):
            j_min, v_min = -1, 1e9
            for j in range(size):
                if j in used_j:
                    continue
                if cost[i][j] < v_min:
                    v_min = cost[i][j]
                    j_min = j
            if j_min >= 0:
                used_j.add(j_min)
                assign[i] = j_min
                total += v_min
    # Normalize: 0 cost → 1 sim, 1 cost → 0 sim
    avg_cost = total / float(size)
    sim = max(0.0, 1.0 - min(1.0, avg_cost))
    return sim


def _normalize_point(
    p: List[float], origin: List[float], scale: float, cos_t: float, sin_t: float
) -> List[float]:
    x, y = p[0] - origin[0], p[1] - origin[1]
    xr = x * cos_t + y * sin_t
    yr = -x * sin_t + y * cos_t
    return [xr / scale, yr / scale]


def _fit_canonical(pts: List[List[float]]) -> Tuple[List[List[float]], List[float], float, float]:
    """Fit canonical transform: origin=center, scale=max(|x|,|y|), rotation via PCA major axis."""
    if not pts:
        return [], [0.0, 0.0], 1.0, 0.0
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    centered = [[p[0] - cx, p[1] - cy] for p in pts]
    # 2x2 covariance
    if len(centered) >= 2:
        xs = [c[0] for c in centered]
        ys = [c[1] for c in centered]
        sxx = statistics.pvariance(xs) if len(xs) > 1 else 0.0
        syy = statistics.pvariance(ys) if len(ys) > 1 else 0.0
        sxy = sum(
            (xs[i] - statistics.mean(xs)) * (ys[i] - statistics.mean(ys)) for i in range(len(xs))
        ) / max(1, len(xs) - 1)
        # eigenvectors for [[sxx, sxy],[sxy, syy]]
        tr = sxx + syy
        det = sxx * syy - sxy * sxy
        disc = max(0.0, tr * tr - 4 * det)
        l1 = 0.5 * (tr + math.sqrt(disc))
        # eigenvector of l1
        vx, vy = (1.0, 0.0)
        if abs(sxy) > 1e-9 or abs(sxx - l1) > 1e-9:
            vx = sxy
            vy = l1 - sxx
        # normalize
        norm = math.hypot(vx, vy) or 1.0
        vx /= norm
        vy /= norm
        theta = math.atan2(vy, vx)
    else:
        theta = 0.0
    # scale
    max_abs = max(1e-6, max(max(abs(c[0]), abs(c[1])) for c in centered))
    scale = max_abs
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    canon = [
        [(c[0] * cos_t + c[1] * sin_t) / scale, (-c[0] * sin_t + c[1] * cos_t) / scale]
        for c in centered
    ]
    return canon, [cx, cy], scale, theta


def _boff_vector(ents: List[Dict[str, Any]], cfg: Settings) -> List[float]:
    """Compute a lightweight bag-of-features vector for entities.

    Bins:
      - LINE length (normalized by median): 8 bins in [0,4]
      - LINE angle (0..180): 8 bins
      - CIRCLE radius: 6 bins (0..max)
      - POLYLINE perimeter: 6 bins (0..max over doc)
    Returns a concatenated normalized histogram vector.
    """
    import math

    lines_len = []
    lines_ang = []
    circles_r = []
    polys_p = []
    ell_axes = []  # major axis lengths
    ell_ratios = []
    leader_len = []
    leader_ang = []
    for e in ents or []:
        t = e.get("type")
        if t == "LINE":
            s = e.get("start", [0, 0])
            ep = e.get("end", [0, 0])
            dx, dy = float(ep[0]) - float(s[0]), float(ep[1]) - float(s[1])
            lines_len.append(math.hypot(dx, dy))
            a = abs(math.degrees(math.atan2(dy, dx))) % 180.0
            lines_ang.append(a)
        elif t == "CIRCLE":
            r = float(e.get("radius", 0.0) or 0.0)
            circles_r.append(r)
        elif t in ("LWPOLYLINE", "POLYLINE"):
            pts = e.get("points") or []
            p = 0.0
            for i in range(1, len(pts)):
                p += math.dist(pts[i - 1], pts[i])
            polys_p.append(p)
        elif t == "ELLIPSE" and getattr(cfg, "enable_entity_ellipse", True):
            maj = e.get("major") or [0.0, 0.0]
            L = math.hypot(float(maj[0]), float(maj[1]))
            if L > 0:
                ell_axes.append(L)
            r = float(e.get("ratio", 0.0) or 0.0)
            if r > 0:
                ell_ratios.append(r)
        elif t == "LEADER" and getattr(cfg, "enable_entity_leader", True):
            pts = e.get("vertices") or []
            if len(pts) >= 2:
                Ld = 0.0
                for i in range(1, len(pts)):
                    Ld += math.dist(pts[i - 1], pts[i])
                leader_len.append(Ld)
                dx = float(pts[-1][0]) - float(pts[0][0])
                dy = float(pts[-1][1]) - float(pts[0][1])
                ang = abs(math.degrees(math.atan2(dy, dx))) % 180.0
                leader_ang.append(ang)

    def hist(vals, bins, vmin, vmax):
        if not vals:
            return [0.0] * bins
        H = [0] * bins
        rng = max(vmax - vmin, 1e-9)
        for v in vals:
            u = (float(v) - vmin) / rng
            k = int(u * bins)
            if k < 0:
                k = 0
            if k >= bins:
                k = bins - 1
            H[k] += 1
        s = float(sum(H)) or 1.0
        return [h / s for h in H]

    # Normalize lengths by median for scale robustness
    def norm_by_median(vals):
        if not vals:
            return []
        srt = sorted(vals)
        med = srt[len(srt) // 2]
        if med > 1e-9:
            return [v / med for v in vals]
        return vals

    v = []
    v.extend(hist(norm_by_median(lines_len), 8, 0.0, 4.0))
    v.extend(hist(lines_ang, 8, 0.0, 180.0))
    vmax_r = max(circles_r) if circles_r else 1.0
    v.extend(hist(circles_r, 6, 0.0, vmax_r))
    vmax_p = max(polys_p) if polys_p else 1.0
    v.extend(hist(polys_p, 6, 0.0, vmax_p))
    # Ellipse major axis length normalized by median
    v.extend(hist(norm_by_median(ell_axes), 6, 0.0, 4.0))
    # Ellipse ratio (0..1) bucketed
    v.extend(hist(ell_ratios, 5, 0.0, 1.0))
    # Leader length normalized by median
    v.extend(hist(norm_by_median(leader_len), 6, 0.0, 4.0))
    # Leader angle 0..180
    v.extend(hist(leader_ang, 6, 0.0, 180.0))
    return v


def _percentile_sorted(values_sorted: List[float], q: float) -> float:
    """Return an approximate percentile from a sorted list (q in [0,100])."""
    if not values_sorted:
        return 0.0
    if q <= 0.0:
        return float(values_sorted[0])
    if q >= 100.0:
        return float(values_sorted[-1])
    idx = int(round((len(values_sorted) - 1) * (q / 100.0)))
    idx = max(0, min(len(values_sorted) - 1, idx))
    return float(values_sorted[idx])


def _entity_rep_points(ents: List[Dict[str, Any]], cfg: Settings) -> List[List[float]]:
    """Representative points for spatial signatures (translation/scale normalized later)."""
    pts: List[List[float]] = []
    for e in ents or []:
        t = e.get("type")
        if t == "LINE":
            s = e.get("start", [0.0, 0.0])
            ep = e.get("end", [0.0, 0.0])
            try:
                pts.append([(float(s[0]) + float(ep[0])) * 0.5, (float(s[1]) + float(ep[1])) * 0.5])
            except Exception:
                continue
        elif t in ("CIRCLE", "ARC"):
            c = e.get("center", [0.0, 0.0])
            try:
                pts.append([float(c[0]), float(c[1])])
            except Exception:
                continue
        elif t in ("LWPOLYLINE", "POLYLINE"):
            ps = e.get("points") or []
            if not ps:
                continue
            try:
                sx = sum(float(p[0]) for p in ps) / len(ps)
                sy = sum(float(p[1]) for p in ps) / len(ps)
                pts.append([sx, sy])
            except Exception:
                continue
        elif t == "INSERT":
            ip = e.get("insert", [0.0, 0.0])
            try:
                pts.append([float(ip[0]), float(ip[1])])
            except Exception:
                continue
        elif t == "ELLIPSE" and getattr(cfg, "enable_entity_ellipse", True):
            c = e.get("center", [0.0, 0.0])
            try:
                pts.append([float(c[0]), float(c[1])])
            except Exception:
                continue
        elif t == "SPLINE":
            cps = e.get("control_points") or []
            if not cps:
                continue
            try:
                sx = sum(float(p[0]) for p in cps) / len(cps)
                sy = sum(float(p[1]) for p in cps) / len(cps)
                pts.append([sx, sy])
            except Exception:
                continue
    return pts


def _spatial_hist_vector(
    pts: List[List[float]],
    *,
    grid: int,
    q_low: float,
    q_high: float,
) -> Tuple[List[float], int]:
    """2D histogram of representative points in a robust bbox, normalized to sum=1."""
    if not pts:
        return [0.0] * (grid * grid), 0

    xs = sorted(float(p[0]) for p in pts)
    ys = sorted(float(p[1]) for p in pts)
    x0 = _percentile_sorted(xs, q_low)
    x1 = _percentile_sorted(xs, q_high)
    y0 = _percentile_sorted(ys, q_low)
    y1 = _percentile_sorted(ys, q_high)

    if not math.isfinite(x0) or not math.isfinite(x1) or x1 <= x0:
        x0, x1 = float(xs[0]), float(xs[-1])
    if not math.isfinite(y0) or not math.isfinite(y1) or y1 <= y0:
        y0, y1 = float(ys[0]), float(ys[-1])

    w = float(x1) - float(x0)
    h = float(y1) - float(y0)
    if w <= 1e-9:
        w = 1.0
    if h <= 1e-9:
        h = 1.0

    grid = max(2, int(grid))
    hist = [0.0] * (grid * grid)
    for p in pts:
        try:
            x = (float(p[0]) - float(x0)) / w
            y = (float(p[1]) - float(y0)) / h
        except Exception:
            continue
        if x < 0.0:
            x = 0.0
        if y < 0.0:
            y = 0.0
        if x >= 1.0:
            x = 1.0 - 1e-9
        if y >= 1.0:
            y = 1.0 - 1e-9
        ix = int(x * grid)
        iy = int(y * grid)
        idx = iy * grid + ix
        if 0 <= idx < len(hist):
            hist[idx] += 1.0

    s = float(sum(hist)) or 1.0
    return [v / s for v in hist], len(pts)


def _cosine(a: List[float], b: List[float]) -> float:
    import math

    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def entities_geom_similarity(
    left_v2: Dict[str, Any], right_v2: Dict[str, Any], cfg: Settings
) -> float:
    if not getattr(cfg, "entities_geom_hash", True):
        return 0.0
    le = left_v2.get("entities") or []
    re = right_v2.get("entities") or []
    sim_boff = _cosine(_boff_vector(le, cfg), _boff_vector(re, cfg))

    if not getattr(cfg, "entities_spatial_enable", False):
        return sim_boff

    try:
        grid = int(getattr(cfg, "entities_spatial_grid", 8) or 8)
    except Exception:
        grid = 8
    try:
        q_low = float(getattr(cfg, "entities_spatial_bbox_q_low", 1.0) or 1.0)
        q_high = float(getattr(cfg, "entities_spatial_bbox_q_high", 99.0) or 99.0)
    except Exception:
        q_low, q_high = 1.0, 99.0
    q_low = max(0.0, min(100.0, q_low))
    q_high = max(0.0, min(100.0, q_high))
    if q_high < q_low:
        q_low, q_high = q_high, q_low

    pts_l = _entity_rep_points(le, cfg)
    pts_r = _entity_rep_points(re, cfg)
    hist_l, n_l = _spatial_hist_vector(pts_l, grid=grid, q_low=q_low, q_high=q_high)
    hist_r, n_r = _spatial_hist_vector(pts_r, grid=grid, q_low=q_low, q_high=q_high)
    sim_spatial = _cosine(hist_l, hist_r)

    try:
        min_pts = int(getattr(cfg, "entities_spatial_min_points", 20) or 0)
    except Exception:
        min_pts = 20
    if min(n_l, n_r) < max(0, min_pts):
        return sim_boff

    try:
        w_max = float(getattr(cfg, "entities_spatial_w_max", 0.6) or 0.0)
    except Exception:
        w_max = 0.0
    w_max = max(0.0, min(1.0, w_max))
    if w_max <= 0.0:
        return sim_boff

    # Weight spatial contribution by how similar the point counts are.
    n_max = max(n_l, n_r, 1)
    count_sim = 1.0 - abs(n_l - n_r) / float(n_max)
    count_sim = max(0.0, min(1.0, count_sim))
    try:
        gamma = float(getattr(cfg, "entities_spatial_count_sim_gamma", 2.0) or 1.0)
    except Exception:
        gamma = 2.0
    if gamma <= 0.0:
        gamma = 1.0
    count_factor = count_sim**gamma
    w_spatial = w_max * count_factor
    return (1.0 - w_spatial) * sim_boff + w_spatial * sim_spatial
