"""
Weighted similarity scoring for DedupCAD 2.0 v2 JSON.

Computes per-section similarity and fuses using configured weights.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

try:
    from .config import Settings
    from .entities_match import entities_geom_similarity, entities_similarity
    from .json_diff import compare_json
except ImportError:  # pragma: no cover - direct script run
    from json_diff import compare_json  # type: ignore

    from config import Settings  # type: ignore

    try:
        from entities_match import entities_geom_similarity, entities_similarity  # type: ignore
    except Exception:  # Fallback stub to avoid NameError in script mode
        from entities_match import entities_similarity  # type: ignore

        def entities_geom_similarity(left_v2, right_v2, cfg):  # type: ignore
            return 0.0


def _safe_section(d: Dict[str, Any], key: str, default: Any) -> Any:
    v = d.get(key)
    return v if v is not None else default


def weighted_similarity(
    left_v2: Dict[str, Any], right_v2: Dict[str, Any], cfg: Settings
) -> Tuple[float, Dict[str, float]]:
    """Return (fused_similarity, breakdown) using section-wise compare.

    Sections: entities, layers, dimensions, text_content, blocks
    """
    # Build sections adaptively; include blocks only when enabled and data available
    sections_list: List[Tuple[str, float, Any, Any]] = []
    sections_list.append(
        (
            "entities",
            cfg.w_entities,
            _safe_section(left_v2, "entities", []),
            _safe_section(right_v2, "entities", []),
        )
    )
    sections_list.append(
        (
            "layers",
            cfg.w_layers,
            _safe_section(left_v2, "layers", {}),
            _safe_section(right_v2, "layers", {}),
        )
    )
    sections_list.append(
        (
            "dimensions",
            cfg.w_dimensions,
            _safe_section(left_v2, "dimensions", {}),
            _safe_section(right_v2, "dimensions", {}),
        )
    )
    sections_list.append(
        (
            "text",
            cfg.w_text,
            _safe_section(left_v2, "text_content", []),
            _safe_section(right_v2, "text_content", []),
        )
    )

    # Decide if blocks section applies; optionally auto down-weight when unavailable
    left_ents = _safe_section(left_v2, "entities", []) or []
    right_ents = _safe_section(right_v2, "entities", []) or []
    has_hash_l = any((e.get("type") == "INSERT" and e.get("block_hash")) for e in left_ents)
    has_hash_r = any((e.get("type") == "INSERT" and e.get("block_hash")) for e in right_ents)
    blocks_unavailable = not (has_hash_l and has_hash_r)
    if cfg.w_blocks > 0:
        if blocks_unavailable and getattr(cfg, "blocks_auto_downweight_when_unavailable", True):
            # Skip blocks section weight for this pair to avoid penalizing missing data
            pass
        else:
            # Respect blocks_strict: require both sides to have hashes when strict
            if (not getattr(cfg, "blocks_strict", True)) or (has_hash_l and has_hash_r):
                sections_list.append(("blocks", cfg.w_blocks, left_ents, right_ents))

    total_w = 0.0
    weighted_sum = 0.0
    breakdown: Dict[str, float] = {}

    for name, w, lsec, rsec in sections_list:
        # Skip zero-weight sections
        if w <= 0:
            continue
        if name == "entities" and cfg.use_entities_matching:
            # Optionally boost entities when blocks unavailable and boost flag set
            boost = 0.0
            if getattr(cfg, "entities_boost_when_no_blocks", False):
                if "blocks" not in [sec[0] for sec in sections_list]:
                    boost = cfg.w_blocks
            s = entities_similarity(left_v2, right_v2, cfg)
            # Geometric bag-of-features fallback; take max to avoid hurting exact matches
            if getattr(cfg, "entities_geom_hash", True):
                sg = entities_geom_similarity(left_v2, right_v2, cfg)
                if sg > s:
                    s = sg
            w = w + boost
        elif name == "blocks":
            s = _blocks_similarity(lsec, rsec, cfg)
        else:
            _, s = compare_json(lsec if lsec is not None else {}, rsec if rsec is not None else {})
        breakdown[name] = s
        total_w += w
        weighted_sum += w * s

    fused = (weighted_sum / total_w) if total_w > 0 else 0.0
    # Optional dimension extra section (text/value consistency + bbox presence)
    if getattr(cfg, "w_dim_extra", 0.0) > 0.0:
        try:
            # Expect dimensions embedded under v2['dimensions'] as list or dict
            dims_left = left_v2.get("dimensions", {}) or {}
            dims_right = right_v2.get("dimensions", {}) or {}
            # Accept list form
            if isinstance(dims_left, list):
                dl = dims_left
            else:
                dl = dims_left.get("left") or dims_left.get("items") or []
            if isinstance(dims_right, list):
                dr = dims_right
            else:
                dr = dims_right.get("right") or dims_right.get("items") or []
            # Only score when both sides non-empty
            if not dl and not dr:
                pass
            elif not dl or not dr:
                breakdown["dim_extra"] = 0.0
                bd_w = float(getattr(cfg, "w_dim_extra", 0.0) or 0.0)
                fused = (fused * total_w + 0.0 * bd_w) / (total_w + bd_w)
                total_w += bd_w
            else:

                def _score_dim(dims_a, dims_b):
                    if not dims_a and not dims_b:
                        return 1.0
                    if not dims_a or not dims_b:
                        return 0.0
                    # Quick ratio on count similarity
                    ca = len(dims_a)
                    cb = len(dims_b)
                    cnt_sim = 1.0 - abs(ca - cb) / float(max(ca, cb, 1))

                    # Text/value match rate heuristic
                    def _tv_rate(ds):
                        if not ds:
                            return 1.0
                        good = 0
                        for d in ds:
                            if d.get("text_matches_value"):
                                good += 1
                        return good / len(ds)

                    tv_a = _tv_rate(dims_a)
                    tv_b = _tv_rate(dims_b)
                    tv_sim = 1.0 - abs(tv_a - tv_b)
                    return max(0.0, min(1.0, (cnt_sim * 0.6 + tv_sim * 0.4)))

                s_dim = _score_dim(dl, dr)
                bd_w = float(getattr(cfg, "w_dim_extra", 0.0) or 0.0)
                breakdown["dim_extra"] = s_dim
                fused = (fused * total_w + s_dim * bd_w) / (total_w + bd_w)
                total_w += bd_w
        except Exception:
            pass
    # Optional hatch extra section (pattern distribution + area similarity)
    if getattr(cfg, "w_hatch_extra", 0.0) > 0.0:
        try:
            h_left = left_v2.get("hatches", {}) or {}
            h_right = right_v2.get("hatches", {}) or {}
            if isinstance(h_left, list):
                hl = h_left
            else:
                hl = h_left.get("left") or h_left.get("items") or []
            if isinstance(h_right, list):
                hr = h_right
            else:
                hr = h_right.get("right") or h_right.get("items") or []
            if not hl and not hr:
                pass
            elif not hl or not hr:
                s_hatch = 0.0
                bh_w = float(getattr(cfg, "w_hatch_extra", 0.0) or 0.0)
                breakdown["hatch_extra"] = s_hatch
                fused = (fused * total_w + s_hatch * bh_w) / (total_w + bh_w)
                total_w += bh_w
            else:

                def _score_hatch(ha, hb):
                    if not ha and not hb:
                        return 1.0
                    if not ha or not hb:
                        return 0.0
                    # Pattern distribution cosine
                    from collections import Counter

                    pa = Counter([h.get("pattern_name") or h.get("hatch_type") for h in ha])
                    pb = Counter([h.get("pattern_name") or h.get("hatch_type") for h in hb])
                    keys = sorted(set(pa) | set(pb))
                    va = [pa.get(k, 0) for k in keys]
                    vb = [pb.get(k, 0) for k in keys]

                    def _cos(a, b):
                        import math

                        dot = sum(x * y for x, y in zip(a, b))
                        na = math.sqrt(sum(x * x for x in a))
                        nb = math.sqrt(sum(y * y for y in b))
                        if na <= 1e-9 or nb <= 1e-9:
                            return 0.0
                        return max(0.0, min(1.0, dot / (na * nb)))

                    pat_sim = _cos(va, vb)
                    # Area similarity (approx via area field if present)
                    import math

                    area_a = sum(float(h.get("area", 0.0) or 0.0) for h in ha)
                    area_b = sum(float(h.get("area", 0.0) or 0.0) for h in hb)
                    if area_a <= 1e-9 and area_b <= 1e-9:
                        area_sim = 1.0
                    else:
                        area_sim = 1.0 - abs(area_a - area_b) / float(max(area_a, area_b, 1e-9))
                    return max(0.0, min(1.0, pat_sim * 0.7 + area_sim * 0.3))

                s_hatch = _score_hatch(hl, hr)
                bh_w = float(getattr(cfg, "w_hatch_extra", 0.0) or 0.0)
                breakdown["hatch_extra"] = s_hatch
                fused = (fused * total_w + s_hatch * bh_w) / (total_w + bh_w)
                total_w += bh_w
        except Exception:
            pass
    # Optionally fuse enhanced similarity (distribution + count + layer structure)
    if getattr(cfg, "w_enhanced", 0.0) > 0.0 and getattr(cfg, "enhanced_similarity_enable", False):
        try:
            from .enhanced_similarity import compute_enhanced_similarity  # type: ignore
        except Exception:
            try:
                from enhanced_similarity import compute_enhanced_similarity  # type: ignore
            except Exception:
                compute_enhanced_similarity = None  # type: ignore
        if compute_enhanced_similarity is not None:
            enh_score, enh_breakdown = compute_enhanced_similarity(left_v2, right_v2, cfg)
            # Safeguard: if critical sections fail thresholds, cap enhanced contribution
            cap = 1.0
            try:
                if getattr(cfg, "enforce_section_thresholds", False):
                    if "entities" in breakdown and breakdown["entities"] < getattr(
                        cfg, "th_entities", 0.0
                    ):
                        cap = min(cap, 0.5)
                    if "layers" in breakdown and breakdown["layers"] < getattr(
                        cfg, "th_layers", 0.0
                    ):
                        cap = min(cap, 0.7)
            except Exception:
                pass
            w_enh = float(getattr(cfg, "w_enhanced", 0.0) or 0.0)
            fused = (fused * total_w + min(1.0, enh_score) * w_enh * cap) / (total_w + w_enh)
            breakdown["enhanced"] = enh_score
    # Optional enforcement of section thresholds
    if cfg.enforce_section_thresholds:
        th_ok = True
        if "entities" in breakdown:
            th_ok = th_ok and (breakdown["entities"] >= cfg.th_entities)
        if "layers" in breakdown:
            th_ok = th_ok and (breakdown["layers"] >= cfg.th_layers)
        if "dimensions" in breakdown:
            th_ok = th_ok and (breakdown["dimensions"] >= cfg.th_dimensions)
        if "text" in breakdown:
            th_ok = th_ok and (breakdown["text"] >= cfg.th_text)
        if "blocks" in breakdown:
            th_ok = th_ok and (breakdown["blocks"] >= cfg.th_blocks)
        if not th_ok:
            fused = min(fused, 0.0)
    return fused, breakdown


def _pose_close(a: Dict[str, Any], b: Dict[str, Any], cfg: Settings) -> bool:
    ia = a.get("insert", [0.0, 0.0]) or [0.0, 0.0]
    ib = b.get("insert", [0.0, 0.0]) or [0.0, 0.0]
    sa = a.get("scale", [1.0, 1.0]) or [1.0, 1.0]
    sb = b.get("scale", [1.0, 1.0]) or [1.0, 1.0]
    ra = float(a.get("rotation", 0.0) or 0.0)
    rb = float(b.get("rotation", 0.0) or 0.0)
    dx = abs(float(ia[0]) - float(ib[0]))
    dy = abs(float(ia[1]) - float(ib[1]))
    dsx = abs(float(sa[0]) - float(sb[0]))
    dsy = abs(float(sa[1]) - float(sb[1]))
    drot = abs(ra - rb)
    return (
        dx <= cfg.tol_insert_pos
        and dy <= cfg.tol_insert_pos
        and dsx <= cfg.tol_insert_scale
        and dsy <= cfg.tol_insert_scale
        and drot <= cfg.tol_insert_rot_deg
    )


def _blocks_similarity(
    left_entities: List[Dict[str, Any]], right_entities: List[Dict[str, Any]], cfg: Settings
) -> float:
    """Compare INSERT references by block_hash and pose tolerance.

    Returns Jaccard-like similarity on matched INSERTs.
    """
    la = [e for e in left_entities or [] if e.get("type") == "INSERT" and e.get("block_hash")]
    rb = [e for e in right_entities or [] if e.get("type") == "INSERT" and e.get("block_hash")]
    if not la or not rb:
        return 0.0
    # Optional Hungarian matching for optimal 1-1 assignment by pose cost
    if getattr(cfg, "use_hungarian", True):
        return _blocks_similarity_hungarian(la, rb, cfg)
    # Greedy fallback
    used_r = set()
    match = 0
    for i, a in enumerate(la):
        for j, b in enumerate(rb):
            if j in used_r:
                continue
            if a.get("block_hash") != b.get("block_hash"):
                continue
            if _pose_close(a, b, cfg):
                used_r.add(j)
                match += 1
                break
    union = len(la) + len(rb) - match
    return match / union if union else 0.0


def _angle_diff_deg(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return d if d <= 180.0 else 360.0 - d


def _sig_similarity(sa: Optional[str], sb: Optional[str]) -> float:
    """Compute a simple token Jaccard similarity between block signatures like
    'LINE:3|CIRCLE:1|TEXT:2'. Returns 0.0..1.0.
    """
    if not sa or not sb:
        return 0.0
    try:

        def parse(sig: str) -> Dict[str, int]:
            out: Dict[str, int] = {}
            for tok in (sig or "").split("|"):
                if not tok:
                    continue
                parts = tok.split(":")
                if len(parts) == 2:
                    k, v = parts[0].strip(), int(parts[1] or "0")
                else:
                    k, v = parts[0].strip(), 1
                if k:
                    out[k] = out.get(k, 0) + max(0, v)
            return out

        A = parse(sa)
        B = parse(sb)
        if not A or not B:
            return 0.0
        keys = set(A.keys()) | set(B.keys())
        inter = 0.0
        uni = 0.0
        for k in keys:
            av = float(A.get(k, 0))
            bv = float(B.get(k, 0))
            inter += min(av, bv)
            uni += max(av, bv)
        return (inter / uni) if uni > 0 else 0.0
    except Exception:
        return 0.0


def _cosine_sim(a: List[float], b: List[float]) -> float:
    try:
        import math

        if len(a) != len(b) or not a:
            return 0.0
        dot = sum((ai * bi) for ai, bi in zip(a, b))
        na = math.sqrt(sum(ai * ai for ai in a))
        nb = math.sqrt(sum(bi * bi for bi in b))
        if na <= 1e-9 or nb <= 1e-9:
            return 0.0
        return max(0.0, min(1.0, dot / (na * nb)))
    except Exception:
        return 0.0


def _sig2_similarity(sa: Optional[str], sb: Optional[str]) -> float:
    """Cosine similarity over concatenated histograms in sig2 (LH, AH, CH)."""
    if not sa or not sb or not sa.startswith("v2|") or not sb.startswith("v2|"):
        return 0.0
    try:

        def parse(sig2: str) -> List[float]:
            # v2|<sig>|LH:a,b,c|AH:a,b,c|CH:a,b,c
            parts = sig2.split("|")
            vec: List[float] = []
            for tag in ("LH:", "AH:", "CH:"):
                found = [p for p in parts if p.startswith(tag)]
                if found:
                    nums = found[0][len(tag) :].split(",")
                    vec.extend([float(x) for x in nums if x != ""])
            return vec

        va = parse(sa)
        vb = parse(sb)
        if not va or not vb:
            return 0.0
        # Optional TF-IDF-like weighting: emphasize rare bins across the pair
        return _cosine_sim(_tfidf_weight(va, vb), _tfidf_weight(vb, va))
    except Exception:
        return 0.0


def _tfidf_weight(a: List[float], b: List[float]) -> List[float]:
    # Build a simple idf per bin from the pair: idf = log( (sum + eps) / (bin + eps) )
    try:
        import math

        eps = 1e-6
        s = sum(a) + sum(b) + eps
        return [ai * math.log(s / (ai + bi + eps)) for ai, bi in zip(a, b)]
    except Exception:
        return a


def _hash_compatible(a: Dict[str, Any], b: Dict[str, Any], cfg: Settings) -> bool:
    """Return True when block hashes match, or when near-hash is enabled and
    block signatures are sufficiently similar."""
    ha = a.get("block_hash")
    hb = b.get("block_hash")
    if ha and hb and ha == hb:
        return True
    if getattr(cfg, "blocks_near_hash", False):
        sim = _sig_similarity(a.get("block_sig"), b.get("block_sig"))
        return sim >= float(getattr(cfg, "blocks_near_hash_sig_threshold", 0.8) or 0.8)
    if getattr(cfg, "blocks_near_hash_v2", False):
        if getattr(cfg, "blocks_near_hash_v2_tfidf", False):
            sim2 = _sig2_similarity(a.get("block_sig2"), b.get("block_sig2"))
        else:
            # Raw cosine without TF-IDF adjustment
            try:

                def _parse(sig2: str) -> List[float]:
                    parts = sig2.split("|")
                    vec: List[float] = []
                    for tag in ("LH:", "AH:", "CH:"):
                        found = [p for p in parts if p.startswith(tag)]
                        if found:
                            nums = found[0][len(tag) :].split(",")
                            vec.extend([float(x) for x in nums if x != ""])
                    return vec

                va = _parse(a.get("block_sig2") or "")
                vb = _parse(b.get("block_sig2") or "")
                sim2 = _cosine_sim(va, vb)
            except Exception:
                sim2 = 0.0
        return sim2 >= float(getattr(cfg, "blocks_near_hash_v2_threshold", 0.85) or 0.85)
    return False


def _pose_cost(a: Dict[str, Any], b: Dict[str, Any], cfg: Settings) -> float:
    # Infinite cost if block_hash not compatible
    if not _hash_compatible(a, b, cfg):
        return 1e6
    ia = a.get("insert", [0.0, 0.0]) or [0.0, 0.0]
    ib = b.get("insert", [0.0, 0.0]) or [0.0, 0.0]
    sa = a.get("scale", [1.0, 1.0]) or [1.0, 1.0]
    sb = b.get("scale", [1.0, 1.0]) or [1.0, 1.0]
    ra = float(a.get("rotation", 0.0) or 0.0)
    rb = float(b.get("rotation", 0.0) or 0.0)
    # Position normalized by tol (Euclidean)
    dx = float(ia[0]) - float(ib[0])
    dy = float(ia[1]) - float(ib[1])
    pos = math.hypot(dx, dy) / max(cfg.tol_insert_pos, 1e-9)
    # Scale normalized by tol
    dsx = abs(float(sa[0]) - float(sb[0])) / max(cfg.tol_insert_scale, 1e-9)
    dsy = abs(float(sa[1]) - float(sb[1])) / max(cfg.tol_insert_scale, 1e-9)
    sc = max(dsx, dsy)
    # Rotation normalized by tol (shortest angle)
    drot = _angle_diff_deg(ra, rb) / max(cfg.tol_insert_rot_deg, 1e-9)
    # If any exceeds 1.0 by a large factor, mark as infeasible
    if pos > 4.0 or sc > 4.0 or drot > 4.0:
        return 1e6
    # Weighted average cost in [0, +)
    return (pos + sc + drot) / 3.0


def _hungarian(cost: List[List[float]]) -> List[Optional[int]]:
    """Hungarian algorithm for minimum cost assignment. Returns list mapping rows->col index or None.
    Pads to square with zeros on diagonal costs left as given.
    """
    n = max(len(cost), max((len(r) for r in cost), default=0))
    INF = 1e9
    # Build square matrix
    a = [[INF] * n for _ in range(n)]
    for i, row in enumerate(cost):
        for j, v in enumerate(row):
            a[i][j] = v
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = a[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, n + 1):
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
    ans = [None] * n
    for j in range(1, n + 1):
        if p[j] != 0 and a[p[j] - 1][j - 1] < INF / 2:
            ans[p[j] - 1] = j - 1
    return ans


def _centroid_inserts(items: List[Dict[str, Any]]) -> Tuple[float, float]:
    sx = sy = 0.0
    n = 0
    for e in items:
        if e.get("type") != "INSERT":
            continue
        ip = e.get("insert") or [0.0, 0.0]
        sx += float(ip[0])
        sy += float(ip[1])
        n += 1
    return (sx / n, sy / n) if n else (0.0, 0.0)


def _rot_about(x: float, y: float, cx: float, cy: float, ang_deg: float) -> Tuple[float, float]:
    ang = math.radians(ang_deg)
    dx, dy = x - cx, y - cy
    rx = dx * math.cos(ang) - dy * math.sin(ang)
    ry = dx * math.sin(ang) + dy * math.cos(ang)
    return (rx + cx, ry + cy)


def _bbox_diag_inserts(items: List[Dict[str, Any]]) -> float:
    xs: List[float] = []
    ys: List[float] = []
    for e in items:
        if e.get("type") != "INSERT":
            continue
        ip = e.get("insert") or [0.0, 0.0]
        xs.append(float(ip[0] or 0.0))
        ys.append(float(ip[1] or 0.0))
    if not xs or not ys:
        return 0.0
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def _blocks_assign_count(
    la: List[Dict[str, Any]], rb: List[Dict[str, Any]], cfg: Settings
) -> Tuple[int, int, int]:
    # Group by block_hash to avoid cross-hash matches
    from collections import defaultdict

    left_by = defaultdict(list)
    right_by = defaultdict(list)
    for e in la:
        left_by[e.get("block_hash")].append(e)
    for e in rb:
        right_by[e.get("block_hash")].append(e)
    total_match = 0
    total_left = 0
    total_right = 0

    # Estimate global rigid transform (rotation + translation)
    def _centroid(items: List[Tuple[float, float]]) -> Tuple[float, float]:
        sx = sy = 0.0
        n = len(items)
        for x, y in items:
            sx += x
            sy += y
        return (sx / n, sy / n) if n else (0.0, 0.0)

    pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for h in set(left_by.keys()) & set(right_by.keys()):
        L = left_by[h]
        R = right_by[h]
        if not L or not R:
            continue
        cl = _centroid(
            [
                (
                    float(e.get("insert", [0.0, 0.0])[0] or 0.0),
                    float(e.get("insert", [0.0, 0.0])[1] or 0.0),
                )
                for e in L
            ]
        )
        cr = _centroid(
            [
                (
                    float(e.get("insert", [0.0, 0.0])[0] or 0.0),
                    float(e.get("insert", [0.0, 0.0])[1] or 0.0),
                )
                for e in R
            ]
        )
        pairs.append((cl, cr))

    def _estimate_rt_centroids(
        pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    ) -> Tuple[float, float, float, float, float]:
        # Returns (ang_deg, cx_l, cy_l, tx, ty) where rotation around (cx_l,cy_l) then translate by (tx,ty)
        if not pairs:
            # Fallback: no rotation, translate by overall centroid delta
            cx_l, cy_l = _centroid_inserts(la)
            cx_r, cy_r = _centroid_inserts(rb)
            return (0.0, cx_l, cy_l, cx_r - cx_l, cy_r - cy_l)
        cl_pts = [p[0] for p in pairs]
        cr_pts = [p[1] for p in pairs]
        cx_l, cy_l = _centroid(cl_pts)
        cx_r, cy_r = _centroid(cr_pts)
        # Procrustes angle estimate
        sxx = sxy = 0.0
        for (xl, yl), (xr, yr) in zip(cl_pts, cr_pts):
            xl -= cx_l
            yl -= cy_l
            xr -= cx_r
            yr -= cy_r
            sxx += xl * xr + yl * yr
            sxy += xl * yr - yl * xr
        ang = math.atan2(sxy, max(sxx, 1e-12))
        ang_deg = ang * 180.0 / math.pi
        # Translation so that R * [cx_l,cy_l] + t = [cx_r,cy_r]
        tx = cx_r - (cx_l * math.cos(ang) - cy_l * math.sin(ang))
        ty = cy_r - (cx_l * math.sin(ang) + cy_l * math.cos(ang))
        return (ang_deg, cx_l, cy_l, tx, ty)

    # RANSAC-based estimate using a Kabsch-like 2-point hypothesis on INSERT positions
    def _estimate_rt_ransac() -> Tuple[float, float, float, float, float]:
        # Build per-hash point lists (INSERT positions)
        left_pts_by = {
            h: [
                (
                    (float(e.get("insert", [0.0, 0.0])[0] or 0.0)),
                    (float(e.get("insert", [0.0, 0.0])[1] or 0.0)),
                )
                for e in left_by[h]
            ]
            for h in left_by
        }
        right_pts_by = {
            h: [
                (
                    (float(e.get("insert", [0.0, 0.0])[0] or 0.0)),
                    (float(e.get("insert", [0.0, 0.0])[1] or 0.0)),
                )
                for e in right_by[h]
            ]
            for h in right_by
        }
        common = [h for h in left_pts_by.keys() if h in right_pts_by]
        if not common:
            return _estimate_rt_centroids(pairs)
        # Adaptive position tolerance
        diag = max(_bbox_diag_inserts(la), _bbox_diag_inserts(rb))
        eff_pos_tol = max(
            getattr(cfg, "tol_insert_pos", 1.0),
            diag * float(getattr(cfg, "blocks_ransac_pos_scale_fraction", 0.0) or 0.0),
        )
        iters = max(8, int(getattr(cfg, "blocks_ransac_iters", 64) or 64))
        best_inliers = -1
        best = (0.0, 0.0, 0.0, 0.0, 0.0)  # (ang_deg, cx, cy, tx, ty)
        # Pivot around overall left centroid improves numerical stability
        cxl, cyl = _centroid_inserts(la)
        import math as _m
        import random as _r

        for _ in range(iters):
            # Sample two (possibly distinct) hashes; ensure each side has at least one point
            h1 = _r.choice(common)
            lp1 = left_pts_by[h1]
            rp1 = right_pts_by[h1]
            if not lp1 or not rp1:
                continue
            a1 = _r.choice(lp1)
            b1 = _r.choice(rp1)
            h2 = _r.choice(common)
            lp2 = left_pts_by[h2]
            rp2 = right_pts_by[h2]
            if not lp2 or not rp2:
                continue
            a2 = _r.choice(lp2)
            b2 = _r.choice(rp2)
            # Compute rotation that aligns vector a1->a2 to b1->b2
            vLx, vLy = (a2[0] - a1[0], a2[1] - a1[1])
            vRx, vRy = (b2[0] - b1[0], b2[1] - b1[1])
            nL = _m.hypot(vLx, vLy)
            nR = _m.hypot(vRx, vRy)
            if nL < 1e-6 or nR < 1e-6:
                continue
            cos_t = (vLx * vRx + vLy * vRy) / max(nL * nR, 1e-9)
            sin_t = (vLx * vRy - vLy * vRx) / max(nL * nR, 1e-9)
            ang_deg = _m.degrees(_m.atan2(sin_t, cos_t))
            # Hypothesis translation: rotate a1 about (cxl,cyl), then translate to b1
            axr, ayr = _rot_about(a1[0], a1[1], cxl, cyl, ang_deg)
            tx_h = b1[0] - axr
            ty_h = b1[1] - ayr
            # Count inliers: transform each left point and check proximity to any right point of same hash
            inliers = 0
            for h in common:
                Rpts = right_pts_by[h]
                if not Rpts:
                    continue
                for lx, ly in left_pts_by[h]:
                    gx, gy = _rot_about(lx, ly, cxl, cyl, ang_deg)
                    gx += tx_h
                    gy += ty_h
                    for rx, ry in Rpts:
                        if _m.hypot(gx - rx, gy - ry) <= eff_pos_tol:
                            inliers += 1
                            break
            if inliers > best_inliers:
                best_inliers = inliers
                best = (ang_deg, cxl, cyl, tx_h, ty_h)
        return best

    if getattr(cfg, "blocks_ransac", True):
        ang_deg_gl, cx_l_gl, cy_l_gl, tx_gl, ty_gl = _estimate_rt_ransac()
    else:
        ang_deg_gl, cx_l_gl, cy_l_gl, tx_gl, ty_gl = _estimate_rt_centroids(pairs)

    # Optional: local clustering by spatial proximity on left INSERTs
    clusters: List[List[int]] = []
    if getattr(cfg, "blocks_local_align", True) and la:
        # Simple grid-based clustering (approx DBSCAN): bucket by eps derived from bbox diag
        diag = _bbox_diag_inserts(la)
        eps = max(
            1e-6,
            float(getattr(cfg, "blocks_local_eps_frac", 0.02) or 0.02)
            * (diag if diag > 0 else 1.0),
        )
        grid = {}
        coords = []
        for idx, e in enumerate(la):
            ip = e.get("insert") or [0.0, 0.0]
            x, y = float(ip[0] or 0.0), float(ip[1] or 0.0)
            coords.append((x, y))
            gx = int(x // eps)
            gy = int(y // eps)
            grid.setdefault((gx, gy), []).append(idx)
        visited = [False] * len(la)

        def neighbors(i: int) -> List[int]:
            x, y = coords[i]
            gx = int(x // eps)
            gy = int(y // eps)
            out = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    out.extend(grid.get((gx + dx, gy + dy), []))
            # radius filter
            res = []
            for j in out:
                if (coords[j][0] - x) ** 2 + (coords[j][1] - y) ** 2 <= (eps * eps):
                    res.append(j)
            return res

        min_samples = max(1, int(getattr(cfg, "blocks_local_min_samples", 3) or 3))
        for i in range(len(la)):
            if visited[i]:
                continue
            visited[i] = True
            nbrs = neighbors(i)
            if len(nbrs) < min_samples:
                continue
            # expand cluster
            cluster = [i]
            seeds = list(nbrs)
            for s in seeds:
                if not visited[s]:
                    visited[s] = True
                    nbrs2 = neighbors(s)
                    if len(nbrs2) >= min_samples:
                        seeds.extend(nbrs2)
                if s not in cluster:
                    cluster.append(s)
            clusters.append(cluster)
    else:
        clusters = [list(range(len(la)))] if la else []

    # Precompute per-cluster transforms via RANSAC (same approach as global), map left-index -> (ang_deg, cx, cy, tx, ty)
    left_idx_by_id = {id(e): i for i, e in enumerate(la)}
    cluster_rt: List[Tuple[float, float, float, float, float]] = []
    if clusters:
        for cl in clusters:
            if not cl:
                cluster_rt.append((ang_deg_gl, cx_l_gl, cy_l_gl, tx_gl, ty_gl))
                continue
            # Candidate pairs constrained to left indices in this cluster
            cands: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []
            for h in set(left_by.keys()) & set(right_by.keys()):
                for idx in cl:
                    a = la[idx]
                    ia = a.get("insert") or [0.0, 0.0]
                    ra = float(a.get("rotation", 0.0) or 0.0)
                    for b in right_by[h]:
                        if a.get("block_hash") != b.get("block_hash"):
                            continue
                        ib = b.get("insert") or [0.0, 0.0]
                        rb_ang = float(b.get("rotation", 0.0) or 0.0)
                        cands.append(
                            ((float(ia[0]), float(ia[1]), ra), (float(ib[0]), float(ib[1]), rb_ang))
                        )
            if not cands:
                cluster_rt.append((ang_deg_gl, cx_l_gl, cy_l_gl, tx_gl, ty_gl))
                continue
            # RANSAC within cluster
            diag_cl = _bbox_diag_inserts([la[i] for i in cl])
            eff_pos_tol_cl = max(
                getattr(cfg, "tol_insert_pos", 1.0),
                diag_cl * float(getattr(cfg, "blocks_ransac_pos_scale_fraction", 0.0) or 0.0),
            )
            iters = max(8, int(getattr(cfg, "blocks_ransac_iters", 64) or 64))
            best_in = -1
            best = (ang_deg_gl, cx_l_gl, cy_l_gl, tx_gl, ty_gl)
            cx_lc, cy_lc = _centroid(
                [
                    (
                        float(la[i].get("insert", [0.0, 0.0])[0] or 0.0),
                        float(la[i].get("insert", [0.0, 0.0])[1] or 0.0),
                    )
                    for i in cl
                ]
            )
            for _ in range(iters):
                import math as _m
                import random as _r

                # Pick two pairs (possibly from different hashes) using only left indices in cluster
                hc = list(set(left_by.keys()) & set(right_by.keys()))
                if not hc:
                    break
                h1 = _r.choice(hc)
                L1 = [
                    (
                        (float(la[i].get("insert", [0.0, 0.0])[0] or 0.0)),
                        (float(la[i].get("insert", [0.0, 0.0])[1] or 0.0)),
                    )
                    for i in cl
                    if la[i].get("block_hash") == h1
                ]
                R1 = [
                    (
                        (float(e.get("insert", [0.0, 0.0])[0] or 0.0)),
                        (float(e.get("insert", [0.0, 0.0])[1] or 0.0)),
                    )
                    for e in right_by[h1]
                ]
                if not L1 or not R1:
                    continue
                a1 = _r.choice(L1)
                b1 = _r.choice(R1)
                h2 = _r.choice(hc)
                L2 = [
                    (
                        (float(la[i].get("insert", [0.0, 0.0])[0] or 0.0)),
                        (float(la[i].get("insert", [0.0, 0.0])[1] or 0.0)),
                    )
                    for i in cl
                    if la[i].get("block_hash") == h2
                ]
                R2 = [
                    (
                        (float(e.get("insert", [0.0, 0.0])[0] or 0.0)),
                        (float(e.get("insert", [0.0, 0.0])[1] or 0.0)),
                    )
                    for e in right_by[h2]
                ]
                if not L2 or not R2:
                    continue
                a2 = _r.choice(L2)
                b2 = _r.choice(R2)
                vLx, vLy = (a2[0] - a1[0], a2[1] - a1[1])
                vRx, vRy = (b2[0] - b1[0], b2[1] - b1[1])
                nL = _m.hypot(vLx, vLy)
                nR = _m.hypot(vRx, vRy)
                if nL < 1e-6 or nR < 1e-6:
                    continue
                cos_t = (vLx * vRx + vLy * vRy) / max(nL * nR, 1e-9)
                sin_t = (vLx * vRy - vLy * vRx) / max(nL * nR, 1e-9)
                ang = _m.atan2(sin_t, cos_t) * 180.0 / _m.pi
                xr, yr = _rot_about(a1[0], a1[1], cx_lc, cy_lc, ang)
                tx_h = b1[0] - xr
                ty_h = b1[1] - yr
                inliers = 0
                for h in hc:
                    Rh = [
                        (
                            (float(e.get("insert", [0.0, 0.0])[0] or 0.0)),
                            (float(e.get("insert", [0.0, 0.0])[1] or 0.0)),
                        )
                        for e in right_by[h]
                    ]
                    Lh = [
                        (
                            (float(la[i].get("insert", [0.0, 0.0])[0] or 0.0)),
                            (float(la[i].get("insert", [0.0, 0.0])[1] or 0.0)),
                        )
                        for i in cl
                        if la[i].get("block_hash") == h
                    ]
                    for lx, ly in Lh:
                        gx, gy = _rot_about(lx, ly, cx_lc, cy_lc, ang)
                        gx += tx_h
                        gy += ty_h
                        for rx, ry in Rh:
                            if _m.hypot(gx - rx, gy - ry) <= eff_pos_tol_cl:
                                inliers += 1
                                break
                if inliers > best_in:
                    best_in = inliers
                    best = (ang, cx_lc, cy_lc, tx_h, ty_h)
            cluster_rt.append(best)
    else:
        cluster_rt = [(ang_deg_gl, cx_l_gl, cy_l_gl, tx_gl, ty_gl)]

    # Precompute cluster membership map to avoid O(C) scans per cell
    cluster_id_of: Dict[int, int] = {}
    for ci, cl in enumerate(clusters or []):
        for idx in cl:
            cluster_id_of[idx] = ci

    for h in set(left_by.keys()) | set(right_by.keys()):
        L = left_by.get(h, [])
        R = right_by.get(h, [])
        if not L or not R:
            total_left += len(L)
            total_right += len(R)
            continue
        # Build cost matrix with global rotation + translation alignment
        bias = ang_deg_gl if getattr(cfg, "blocks_global_align", True) else 0.0
        cx_l, cy_l = cx_l_gl, cy_l_gl
        tx_g, ty_g = tx_gl, ty_gl

        # If in local clusters mode, expand cost by selecting better of (global, local) transform per left index
        def cost_cell(a: Dict[str, Any], b: Dict[str, Any]) -> float:
            if not _hash_compatible(a, b, cfg):
                return 1e6
            ia = a.get("insert", [0.0, 0.0]) or [0.0, 0.0]
            ib = b.get("insert", [0.0, 0.0]) or [0.0, 0.0]
            sa = a.get("scale", [1.0, 1.0]) or [1.0, 1.0]
            sb = b.get("scale", [1.0, 1.0]) or [1.0, 1.0]
            ra = float(a.get("rotation", 0.0) or 0.0)
            rb = float(b.get("rotation", 0.0) or 0.0)
            # Global hypothesis
            gx, gy = _rot_about(float(ia[0]), float(ia[1]), cx_l, cy_l, bias)
            gx += tx_g
            gy += ty_g
            gdx = gx - float(ib[0])
            gdy = gy - float(ib[1])
            pos = math.hypot(gdx, gdy) / max(cfg.tol_insert_pos, 1e-9)
            dsx = abs(float(sa[0]) - float(sb[0])) / max(cfg.tol_insert_scale, 1e-9)
            dsy = abs(float(sa[1]) - float(sb[1])) / max(cfg.tol_insert_scale, 1e-9)
            sc = max(dsx, dsy)
            drot = _angle_diff_deg(ra + bias, rb) / max(cfg.tol_insert_rot_deg, 1e-9)
            g_cost = (pos + sc + drot) / 3.0
            # Local hypothesis: use cluster transform if available
            cost_val = g_cost
            if clusters and left_idx_by_id:
                idx = left_idx_by_id.get(id(a))
                if idx is not None:
                    cl_id = cluster_id_of.get(idx)
                    if cl_id is not None and cl_id < len(cluster_rt):
                        ang_c, cx_c, cy_c, tx_c, ty_c = cluster_rt[cl_id]
                        lx, ly = _rot_about(float(ia[0]), float(ia[1]), cx_c, cy_c, ang_c)
                        lx += tx_c
                        ly += ty_c
                        ldx = lx - float(ib[0])
                        ldy = ly - float(ib[1])
                        lpos = math.hypot(ldx, ldy) / max(cfg.tol_insert_pos, 1e-9)
                        ldrot = _angle_diff_deg(ra + ang_c, rb) / max(cfg.tol_insert_rot_deg, 1e-9)
                        l_cost = (lpos + sc + ldrot) / 3.0
                        if l_cost < cost_val:
                            cost_val = l_cost
            if pos > 4.0 or sc > 4.0 or drot > 4.0:
                return 1e6
            return cost_val

        cost = [[cost_cell(a, b) for b in R] for a in L]
        assign = _hungarian(cost)
        # Count feasible matches (cost <= 1.0 threshold)
        match = 0
        for i, j in enumerate(assign[: len(L)]):
            if j is None or j >= len(R):
                continue
            if cost[i][j] <= 1.0:
                match += 1
        total_match += match
        total_left += len(L)
        total_right += len(R)
    return total_match, total_left, total_right


def _blocks_similarity_hungarian(
    la: List[Dict[str, Any]], rb: List[Dict[str, Any]], cfg: Settings
) -> float:
    # Fast path: unweighted Jaccard on counts
    if not getattr(cfg, "blocks_weighted_jaccard", False) and not getattr(
        cfg, "blocks_area_weighted_jaccard", False
    ):
        total_match, total_left, total_right = _blocks_assign_count(la, rb, cfg)
        union = total_left + total_right - total_match
        return total_match / union if union else 0.0

    # Weighted Jaccard based on inverse frequency of block occurrences.
    # 1) Compute pair assignments (accepted matches)
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    from collections import defaultdict

    left_by = defaultdict(list)
    right_by = defaultdict(list)
    for i, e in enumerate(la):
        left_by[e.get("block_hash")].append((i, e))
    for j, e in enumerate(rb):
        right_by[e.get("block_hash")].append((j, e))

    # Reuse global/cluster alignment estimation from _blocks_assign_count by calling it to set globals
    # We can't directly access internals; re-estimate minimal parameters here.
    # Global RT estimate using centroid method across same-hash groups
    def _centroid(items: List[Tuple[float, float]]) -> Tuple[float, float]:
        sx = sy = 0.0
        n = len(items)
        for x, y in items:
            sx += x
            sy += y
        return (sx / n, sy / n) if n else (0.0, 0.0)

    # Pair of centroids (left, right) for global transform estimation
    pairs_ct: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for h in set(left_by.keys()) & set(right_by.keys()):
        L = [e for _, e in left_by[h]]
        R = [e for _, e in right_by[h]]
        if not L or not R:
            continue
        cl = _centroid(
            [
                (
                    float(e.get("insert", [0.0, 0.0])[0] or 0.0),
                    float(e.get("insert", [0.0, 0.0])[1] or 0.0),
                )
                for e in L
            ]
        )
        cr = _centroid(
            [
                (
                    float(e.get("insert", [0.0, 0.0])[0] or 0.0),
                    float(e.get("insert", [0.0, 0.0])[1] or 0.0),
                )
                for e in R
            ]
        )
        pairs_ct.append((cl, cr))

    if pairs_ct:
        cl_pts = [p[0] for p in pairs_ct]
        cr_pts = [p[1] for p in pairs_ct]
        cx_l = sum(x for x, _ in cl_pts) / len(cl_pts)
        cy_l = sum(y for _, y in cl_pts) / len(cl_pts)
        cx_r = sum(x for x, _ in cr_pts) / len(cr_pts)
        cy_r = sum(y for _, y in cr_pts) / len(cr_pts)
        sxx = sxy = 0.0
        for (xl, yl), (xr, yr) in zip(cl_pts, cr_pts):
            xl -= cx_l
            yl -= cy_l
            xr -= cx_r
            yr -= cy_r
            sxx += xl * xr + yl * yr
            sxy += xl * yr - yl * xr
        ang = math.atan2(sxy, max(sxx, 1e-12))
        ang_deg = ang * 180.0 / math.pi
        tx_g = cx_r - (cx_l * math.cos(ang) - cy_l * math.sin(ang))
        ty_g = cy_r - (cx_l * math.sin(ang) + cy_l * math.cos(ang))
    else:
        ang_deg = 0.0
        cx_l, cy_l = _centroid_inserts(la)
        cx_r, cy_r = _centroid_inserts(rb)
        tx_g = cx_r - cx_l
        ty_g = cy_r - cy_l

    bias = ang_deg if getattr(cfg, "blocks_global_align", True) else 0.0

    # Build matches per hash group with Hungarian and pose gating
    for h in set(left_by.keys()) | set(right_by.keys()):
        L = left_by.get(h, [])
        R = right_by.get(h, [])
        if not L or not R:
            continue
        # Optional neighbor index to reduce candidate rights per left
        use_nbr = bool(getattr(cfg, "blocks_neighbor_index", True))
        cand_right_idx: List[int] = list(range(len(R)))
        # Only enable neighbor index for sufficiently large matrices to avoid recall loss on tiny sets
        if use_nbr and (
            len(L) * len(R) >= int(getattr(cfg, "blocks_nbr_min_matrix", 20000) or 20000)
        ):
            # Grid over right INSERT positions
            diag = _bbox_diag_inserts([e for _, e in R])
            eps = max(
                1e-6,
                float(getattr(cfg, "blocks_local_eps_frac", 0.02) or 0.02)
                * (diag if diag > 0 else 1.0),
            )
            eff_pos_tol = max(
                getattr(cfg, "tol_insert_pos", 1.0),
                diag * float(getattr(cfg, "blocks_ransac_pos_scale_fraction", 0.0) or 0.0),
            )
            mul = float(getattr(cfg, "blocks_nbr_radius_mul", 3.0) or 3.0)
            radius = max(eps, eff_pos_tol) * mul
            # KD-tree over right INSERT positions for radius search
            try:
                from .neighbor_index import KDTree2D  # type: ignore
            except Exception:
                from neighbor_index import KDTree2D  # type: ignore
            rcoords = []
            for j, (_, b) in enumerate(R):
                ip = b.get("insert") or [0.0, 0.0]
                x, y = float(ip[0] or 0.0), float(ip[1] or 0.0)
                rcoords.append((x, y))
            kdt = KDTree2D(rcoords)
            cand = set()
            # Use global transform to predict positions
            for _, a in L:
                ia = a.get("insert") or [0.0, 0.0]
                px, py = _rot_about(float(ia[0]), float(ia[1]), cx_l, cy_l, ang_deg)
                px += tx_g
                py += ty_g
                for j in kdt.radius_search((px, py), radius):
                    cand.add(j)
            # If too few candidates (e.g., <5% of R) fall back to full R to avoid recall loss
            if len(cand) >= max(1, int(0.05 * len(R))):
                cand_right_idx = sorted(list(cand))
        # Remap rights to candidate subset
        R_sub = [R[j] for j in cand_right_idx]

        def cost_cell(a: Dict[str, Any], b: Dict[str, Any]) -> float:
            if not _hash_compatible(a, b, cfg):
                return 1e6
            ia = a.get("insert", [0.0, 0.0]) or [0.0, 0.0]
            ib = b.get("insert", [0.0, 0.0]) or [0.0, 0.0]
            sa = a.get("scale", [1.0, 1.0]) or [1.0, 1.0]
            sb = b.get("scale", [1.0, 1.0]) or [1.0, 1.0]
            ra = float(a.get("rotation", 0.0) or 0.0)
            rb_ang = float(b.get("rotation", 0.0) or 0.0)
            gx, gy = _rot_about(float(ia[0]), float(ia[1]), cx_l, cy_l, bias)
            gx += tx_g
            gy += ty_g
            gdx = gx - float(ib[0])
            gdy = gy - float(ib[1])
            pos = math.hypot(gdx, gdy) / max(cfg.tol_insert_pos, 1e-9)
            dsx = abs(float(sa[0]) - float(sb[0])) / max(cfg.tol_insert_scale, 1e-9)
            dsy = abs(float(sa[1]) - float(sb[1])) / max(cfg.tol_insert_scale, 1e-9)
            sc = max(dsx, dsy)
            drot = _angle_diff_deg(ra + bias, rb_ang) / max(cfg.tol_insert_rot_deg, 1e-9)
            if pos > 4.0 or sc > 4.0 or drot > 4.0:
                return 1e6
            return (pos + sc + drot) / 3.0

        cost = [[cost_cell(a, b) for _, b in R_sub] for _, a in L]
        assign = _hungarian(cost)
        for i, j in enumerate(assign[: len(L)]):
            if j is None or j >= len(R):
                continue
            if j is not None and j < len(R_sub) and cost[i][j] <= 1.0:
                pairs.append((L[i][1], R_sub[j][1]))

    # 2) Compute weights
    from collections import Counter

    freq_l = Counter([e.get("block_hash") for e in la])
    freq_r = Counter([e.get("block_hash") for e in rb])

    def w_invfreq_for_pair(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        ha = a.get("block_hash")
        hb = b.get("block_hash")
        fl = max(1, int(freq_l.get(ha, 1)))
        fr = max(1, int(freq_r.get(hb, 1)))
        return 1.0 / float(max(fl, fr))

    def area_of(e: Dict[str, Any]) -> float:
        # Prefer per-INSERT embedded block_area populated by extractor; fallback 1.0
        try:
            v = float(e.get("block_area") or 0.0)
            return v if v > 0 else 1.0
        except Exception:
            return 1.0

    def w_area_for_pair(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        # Area-weighted: use geometric mean of area proxies from both sides (when available)
        return max(1e-6, math.sqrt(area_of(a) * area_of(b)))

    # sum weights
    use_area = getattr(cfg, "blocks_area_weighted_jaccard", False)
    alpha = float(getattr(cfg, "blocks_weight_alpha", 0.0) or 0.0)
    beta = float(getattr(cfg, "blocks_weight_beta", 1.0) or 1.0)

    def w_hybrid(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        # area^alpha / freq^beta
        area_g = max(1e-6, math.sqrt(area_of(a) * area_of(b)))
        ha = a.get("block_hash")
        hb = b.get("block_hash")
        fl = max(1, int(freq_l.get(ha, 1)))
        fr = max(1, int(freq_r.get(hb, 1)))
        freq_g = float(max(fl, fr))
        return (area_g**alpha) / (freq_g**beta)

    if use_area and alpha > 0.0:
        inter = sum(w_hybrid(a, b) for (a, b) in pairs)
        # approximate union with per-side sums under hybrid weights
        sum_left = sum(
            (max(1e-6, area_of(e)) ** alpha)
            / (float(max(1, freq_l.get(e.get("block_hash"), 1))) ** beta)
            for e in la
        )
        sum_right = sum(
            (max(1e-6, area_of(e)) ** alpha)
            / (float(max(1, freq_r.get(e.get("block_hash"), 1))) ** beta)
            for e in rb
        )
    elif use_area:
        inter = sum(w_area_for_pair(a, b) for (a, b) in pairs)
        sum_left = float(len(la))
        sum_right = float(len(rb))
    else:
        inter = sum(w_invfreq_for_pair(a, b) for (a, b) in pairs)
        sum_left = sum(1.0 / float(max(1, freq_l.get(e.get("block_hash"), 1))) for e in la)
        sum_right = sum(1.0 / float(max(1, freq_r.get(e.get("block_hash"), 1))) for e in rb)
    union = sum_left + sum_right - inter
    return (inter / union) if union > 0 else 0.0


def blocks_match_stats(
    left_entities: List[Dict[str, Any]], right_entities: List[Dict[str, Any]], cfg: Settings
) -> Tuple[int, int, int]:
    """Return (matched_inserts, left_inserts, right_inserts) using the same matching as similarity.

    Only INSERTs with non-empty block_hash are considered.
    """
    la = [e for e in left_entities or [] if e.get("type") == "INSERT" and e.get("block_hash")]
    rb = [e for e in right_entities or [] if e.get("type") == "INSERT" and e.get("block_hash")]
    if not la or not rb:
        return (0, len(la), len(rb))
    if getattr(cfg, "use_hungarian", True):
        m, l, r = _blocks_assign_count(la, rb, cfg)
        return (m, l, r)
    # Greedy fallback
    used_r = set()
    match = 0
    for i, a in enumerate(la):
        for j, b in enumerate(rb):
            if j in used_r:
                continue
            if a.get("block_hash") != b.get("block_hash"):
                continue
            if _pose_close(a, b, cfg):
                used_r.add(j)
                match += 1
                break
    return (match, len(la), len(rb))
