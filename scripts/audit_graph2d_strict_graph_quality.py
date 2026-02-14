#!/usr/bin/env python3
"""Audit Graph2D strict-mode graph build quality on a DXF directory.

This script inspects the *graph construction* stage (nodes/edges) under
production-like strict settings:
  - DXF text entities stripped (optional)
  - importance sampling applied (DXF_MAX_NODES / ratios)
  - epsilon adjacency edges computed from entity keypoints
  - empty-edge fallback applied when epsilon adjacency yields no edges

It produces:
  - per_file.csv: per-DXF stats
  - summary.json: aggregated stats and configuration snapshot

The output directory is intended for local iteration (contains local paths).
Do not commit the artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    q = max(0.0, min(1.0, float(q)))
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    idx = q * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = idx - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _iter_dxf_paths(dxf_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for ext in ("*.dxf", "*.DXF"):
        paths.extend(sorted(dxf_dir.glob(ext)))
    return sorted(set(paths))


def _touching(
    pts_a: Sequence[Tuple[float, float]],
    pts_b: Sequence[Tuple[float, float]],
    eps: float,
) -> bool:
    eps2 = eps * eps
    for ax, ay in pts_a:
        for bx, by in pts_b:
            dx = ax - bx
            dy = ay - by
            if (dx * dx + dy * dy) <= eps2:
                return True
    return False


def _enhanced_keypoints_enabled() -> bool:
    raw = os.getenv("DXF_ENHANCED_KEYPOINTS", "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _entity_keypoints_and_center(entity: Any) -> Tuple[str, List[Tuple[float, float]], Tuple[float, float]]:
    dtype = str(entity.dxftype())
    pts: List[Tuple[float, float]] = []
    center = (0.0, 0.0)

    if dtype == "LINE":
        start = entity.dxf.start
        end = entity.dxf.end
        sx = float(start.x)
        sy = float(start.y)
        ex = float(end.x)
        ey = float(end.y)
        pts = [(sx, sy), (ex, ey)]
        center = ((sx + ex) * 0.5, (sy + ey) * 0.5)
    elif dtype == "CIRCLE":
        c = entity.dxf.center
        cx = float(c.x)
        cy = float(c.y)
        pts = [(cx, cy)]
        if _enhanced_keypoints_enabled():
            radius = float(entity.dxf.radius)
            if radius > 0:
                pts.extend(
                    [
                        (cx + radius, cy),
                        (cx - radius, cy),
                        (cx, cy + radius),
                        (cx, cy - radius),
                    ]
                )
        center = (cx, cy)
    elif dtype == "ARC":
        c = entity.dxf.center
        cx = float(c.x)
        cy = float(c.y)
        radius = float(entity.dxf.radius)
        start_angle = math.radians(float(entity.dxf.start_angle))
        end_angle = math.radians(float(entity.dxf.end_angle))
        delta = end_angle - start_angle
        if delta < 0:
            delta += 2 * math.pi
        sx = cx + radius * math.cos(start_angle)
        sy = cy + radius * math.sin(start_angle)
        ex = cx + radius * math.cos(end_angle)
        ey = cy + radius * math.sin(end_angle)
        pts = [(sx, sy), (ex, ey)]
        if _enhanced_keypoints_enabled() and radius > 0 and delta > 1e-9:
            mid_angle = start_angle + delta * 0.5
            mx = cx + radius * math.cos(mid_angle)
            my = cy + radius * math.sin(mid_angle)
            pts.append((mx, my))
        center = (cx, cy)
    elif dtype == "LWPOLYLINE":
        try:
            pts = [(float(p[0]), float(p[1])) for p in entity.get_points()]
        except Exception:
            pts = []
        if pts:
            center = (
                sum(p[0] for p in pts) / len(pts),
                sum(p[1] for p in pts) / len(pts),
            )
    elif dtype in {"TEXT", "MTEXT"}:
        insert = getattr(entity.dxf, "insert", None) or getattr(entity.dxf, "location", None)
        if insert is not None:
            center = (float(insert.x), float(insert.y))
        pts = [center]
    elif dtype == "DIMENSION":
        point = (
            getattr(entity.dxf, "text_midpoint", None)
            or getattr(entity.dxf, "defpoint", None)
            or getattr(entity.dxf, "insert", None)
        )
        if point is not None:
            center = (float(point.x), float(point.y))
        pts = [center]
    elif dtype == "INSERT":
        insert = getattr(entity.dxf, "insert", None)
        if insert is not None:
            center = (float(insert.x), float(insert.y))
        pts = [center]

    if not pts:
        pts = [center]
    return dtype, pts, center


def _compute_bbox(points: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    if not points:
        return (0.0, 0.0, 100.0, 100.0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _max_dim(bbox: Tuple[float, float, float, float]) -> float:
    min_x, min_y, max_x, max_y = bbox
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    return max(width, height, 1.0)


def _knn_edge_count(centers: Sequence[Tuple[float, float]], k: int) -> int:
    n = len(centers)
    if n <= 1:
        return 0
    k = max(1, min(int(k), n - 1))

    edge_set: set[Tuple[int, int]] = set()
    for i in range(n):
        cx_i, cy_i = centers[i]
        dists: List[Tuple[float, int]] = []
        for j in range(n):
            if i == j:
                continue
            cx_j, cy_j = centers[j]
            dx = cx_j - cx_i
            dy = cy_j - cy_i
            dists.append((dx * dx + dy * dy, j))
        dists.sort(key=lambda t: t[0])
        for _dist2, j in dists[:k]:
            edge_set.add((i, j))
            edge_set.add((j, i))
    return len(edge_set)


def _connected_components(n: int, edges_directed: Sequence[Tuple[int, int]]) -> int:
    if n <= 0:
        return 0
    if n == 1:
        return 1

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges_directed:
        if 0 <= a < n and 0 <= b < n:
            union(a, b)

    roots = {find(i) for i in range(n)}
    return len(roots)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class AuditConfig:
    strip_text_entities: bool
    max_files: int
    seed: int


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Graph2D strict-mode graph build quality.")
    parser.add_argument("--dxf-dir", required=True, help="DXF directory to inspect.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory (default: /tmp/graph2d_graph_audit_<ts>).",
    )
    parser.add_argument(
        "--strip-text-entities",
        action="store_true",
        help="Strip TEXT/MTEXT/DIMENSION entities from DXF before graph build.",
    )
    parser.add_argument("--max-files", type=int, default=0, help="Optional cap (0=all).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dxf_dir = Path(args.dxf_dir)
    if not dxf_dir.exists():
        print(f"DXF dir not found: {dxf_dir}")
        return 2

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path("/tmp") / f"graph2d_graph_audit_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")

    config = AuditConfig(
        strip_text_entities=bool(args.strip_text_entities),
        max_files=int(args.max_files),
        seed=int(args.seed),
    )

    paths = _iter_dxf_paths(dxf_dir)
    rng = random.Random(config.seed)
    rng.shuffle(paths)
    if config.max_files > 0:
        paths = paths[: config.max_files]

    # Resolve sampling/fallback config from env (same knobs used by training).
    sampler_env = {
        "DXF_MAX_NODES": os.getenv("DXF_MAX_NODES", ""),
        "DXF_SAMPLING_STRATEGY": os.getenv("DXF_SAMPLING_STRATEGY", ""),
        "DXF_SAMPLING_SEED": os.getenv("DXF_SAMPLING_SEED", ""),
        "DXF_TEXT_PRIORITY_RATIO": os.getenv("DXF_TEXT_PRIORITY_RATIO", ""),
        "DXF_FRAME_PRIORITY_RATIO": os.getenv("DXF_FRAME_PRIORITY_RATIO", ""),
        "DXF_LONG_LINE_RATIO": os.getenv("DXF_LONG_LINE_RATIO", ""),
        "DXF_EDGE_AUGMENT_KNN_K": os.getenv("DXF_EDGE_AUGMENT_KNN_K", ""),
        "DXF_EDGE_AUGMENT_STRATEGY": os.getenv("DXF_EDGE_AUGMENT_STRATEGY", ""),
        "DXF_EMPTY_EDGE_FALLBACK": os.getenv("DXF_EMPTY_EDGE_FALLBACK", "fully_connected"),
        "DXF_EMPTY_EDGE_K": os.getenv("DXF_EMPTY_EDGE_K", "8"),
        "DXF_ENHANCED_KEYPOINTS": os.getenv("DXF_ENHANCED_KEYPOINTS", ""),
    }

    per_file: List[Dict[str, Any]] = []
    node_counts: List[float] = []
    adj_edge_counts: List[float] = []
    final_edge_counts: List[float] = []
    fallback_used = 0

    import ezdxf  # noqa: PLC0415

    from src.ml.importance_sampler import ImportanceSampler  # noqa: PLC0415
    from src.utils.dxf_io import (  # noqa: PLC0415
        read_dxf_document_from_bytes,
        strip_dxf_text_entities_from_bytes,
    )

    sampler = ImportanceSampler()
    valid_types = {
        "LINE",
        "CIRCLE",
        "ARC",
        "LWPOLYLINE",
        "TEXT",
        "MTEXT",
        "DIMENSION",
        "INSERT",
    }

    started_at = time.time()

    for path in paths:
        file_started = time.time()
        status = "ok"
        err = ""

        sampled_entities: List[Any] = []
        dtype_counts: Dict[str, int] = {}
        adj_edges_directed: List[Tuple[int, int]] = []
        final_edges_directed: List[Tuple[int, int]] = []

        try:
            if config.strip_text_entities:
                raw_bytes = path.read_bytes()
                stripped = strip_dxf_text_entities_from_bytes(raw_bytes, strip_blocks=True)
                doc = read_dxf_document_from_bytes(stripped)
            else:
                doc = ezdxf.readfile(str(path))
            msp = doc.modelspace()
            entities = [e for e in list(msp) if e.dxftype() in valid_types]
            if entities:
                sampled_entities = sampler.sample(entities).sampled_entities
            else:
                sampled_entities = []
        except Exception as exc:  # noqa: BLE001
            status = "error"
            err = f"load_error: {exc}"
            sampled_entities = []

        keypoints: List[List[Tuple[float, float]]] = []
        centers: List[Tuple[float, float]] = []
        dtypes: List[str] = []
        all_points: List[Tuple[float, float]] = []

        if status == "ok" and sampled_entities:
            for ent in sampled_entities:
                try:
                    dtype, pts, center = _entity_keypoints_and_center(ent)
                except Exception:
                    dtype = "UNKNOWN"
                    pts = [(0.0, 0.0)]
                    center = (0.0, 0.0)
                dtypes.append(dtype)
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
                keypoints.append(list(pts))
                centers.append(center)
                all_points.extend(pts)

            bbox = _compute_bbox(all_points)
            max_dim = _max_dim(bbox)
            eps = max(1e-3, max_dim * 1e-3)

            n = len(sampled_entities)
            for i in range(n):
                for j in range(i + 1, n):
                    if _touching(keypoints[i], keypoints[j], eps):
                        adj_edges_directed.append((i, j))
                        adj_edges_directed.append((j, i))

            final_edge_mode = "epsilon"
            final_edges_directed = list(adj_edges_directed)
            final_edge_count = len(final_edges_directed)

            # Optional kNN augmentation (union with epsilon-adjacency).
            augment_k = _safe_int(sampler_env.get("DXF_EDGE_AUGMENT_KNN_K", ""), 0)
            augment_strategy = str(
                sampler_env.get("DXF_EDGE_AUGMENT_STRATEGY", "union_all") or "union_all"
            ).strip().lower()
            if augment_strategy not in {"union_all", "isolates_only"}:
                augment_strategy = "union_all"

            if augment_k > 0 and final_edge_count > 0 and n > 1:
                augment_nodes: List[int]
                if augment_strategy == "isolates_only":
                    degrees = [0] * n
                    for src, _dst in final_edges_directed:
                        degrees[int(src)] += 1
                    augment_nodes = [idx for idx, deg in enumerate(degrees) if deg == 0]
                else:
                    augment_nodes = list(range(n))
                if not augment_nodes:
                    augment_k = 0

            if augment_k > 0 and final_edge_count > 0 and n > 1:
                edge_set: set[Tuple[int, int]] = set(final_edges_directed)
                k = max(1, min(int(augment_k), n - 1))
                for a in augment_nodes:
                    cx_a, cy_a = centers[a]
                    dist_list: List[Tuple[float, int]] = []
                    for b in range(n):
                        if a == b:
                            continue
                        cx_b, cy_b = centers[b]
                        dx = cx_b - cx_a
                        dy = cy_b - cy_a
                        dist_list.append((dx * dx + dy * dy, b))
                    dist_list.sort(key=lambda t: t[0])
                    for _dist2, b in dist_list[:k]:
                        edge_set.add((a, b))
                        edge_set.add((b, a))
                final_edges_directed = list(edge_set)
                final_edge_count = len(final_edges_directed)
                final_edge_mode = f"epsilon+knn:{k}:{augment_strategy}"

            empty_fallback_mode = str(sampler_env["DXF_EMPTY_EDGE_FALLBACK"]).strip().lower()
            empty_fallback_k = _safe_int(sampler_env["DXF_EMPTY_EDGE_K"], 8)
            if final_edge_count == 0 and n > 1:
                fallback_used += 1
                final_edge_mode = f"fallback:{empty_fallback_mode}"
                if empty_fallback_mode in {"knn", "k_nn", "nearest"}:
                    final_edge_count = _knn_edge_count(centers, empty_fallback_k)
                    # Directed edges are not material for CC calculation; construct
                    # an approximate directed list for CC by adding kNN edges.
                    # Keep it cheap by reusing the exact counter path for density.
                    # CC will be computed on an undirected union of those edges.
                    # (This is sufficient for audit purposes.)
                    # Build a small set of edges for CC.
                    edge_set: set[Tuple[int, int]] = set()
                    k = max(1, min(int(empty_fallback_k), n - 1))
                    for a in range(n):
                        cx_a, cy_a = centers[a]
                        dist_list: List[Tuple[float, int]] = []
                        for b in range(n):
                            if a == b:
                                continue
                            cx_b, cy_b = centers[b]
                            dx = cx_b - cx_a
                            dy = cy_b - cy_a
                            dist_list.append((dx * dx + dy * dy, b))
                        dist_list.sort(key=lambda t: t[0])
                        for _dist2, b in dist_list[:k]:
                            edge_set.add((a, b))
                            edge_set.add((b, a))
                    final_edges_directed = list(edge_set)
                else:
                    final_edge_count = n * (n - 1)
                    final_edges_directed = [
                        (a, b) for a in range(n) for b in range(n) if a != b
                    ]

            density_den = max(1, n * (n - 1))
            adj_density = len(adj_edges_directed) / float(density_den)
            final_density = len(final_edges_directed) / float(density_den)
            cc_count = _connected_components(n, final_edges_directed)

            row: Dict[str, Any] = {
                "file_name": path.name,
                "file_path": str(path),
                "status": status,
                "error": err,
                "nodes": n,
                "adj_edges": len(adj_edges_directed),
                "adj_density": round(adj_density, 6),
                "final_edges": len(final_edges_directed),
                "final_density": round(final_density, 6),
                "final_edge_mode": final_edge_mode,
                "connected_components": cc_count,
                "eps": round(eps, 6),
                "max_dim": round(max_dim, 3),
                "elapsed_seconds": round(time.time() - file_started, 3),
            }
            for dtype, count in sorted(dtype_counts.items()):
                row[f"dtype_{dtype.lower()}"] = int(count)

            per_file.append(row)
            node_counts.append(float(n))
            adj_edge_counts.append(float(len(adj_edges_directed)))
            final_edge_counts.append(float(len(final_edges_directed)))
        else:
            per_file.append(
                {
                    "file_name": path.name,
                    "file_path": str(path),
                    "status": status,
                    "error": err,
                    "nodes": 0,
                    "adj_edges": 0,
                    "adj_density": 0.0,
                    "final_edges": 0,
                    "final_density": 0.0,
                    "final_edge_mode": "",
                    "connected_components": 0,
                    "eps": 0.0,
                    "max_dim": 0.0,
                    "elapsed_seconds": round(time.time() - file_started, 3),
                }
            )

    elapsed = time.time() - started_at
    summary: Dict[str, Any] = {
        "status": "ok",
        "dxf_dir": str(dxf_dir),
        "files": len(paths),
        "strip_text_entities": bool(config.strip_text_entities),
        "sampler_env": sampler_env,
        "fallback_used": int(fallback_used),
        "fallback_rate": float(fallback_used) / float(len(paths) or 1),
        "nodes": {
            "min": int(min(node_counts) if node_counts else 0),
            "p50": round(_percentile(node_counts, 0.5), 3),
            "p90": round(_percentile(node_counts, 0.9), 3),
            "max": int(max(node_counts) if node_counts else 0),
        },
        "adj_edges": {
            "p50": round(_percentile(adj_edge_counts, 0.5), 3),
            "p90": round(_percentile(adj_edge_counts, 0.9), 3),
        },
        "final_edges": {
            "p50": round(_percentile(final_edge_counts, 0.5), 3),
            "p90": round(_percentile(final_edge_counts, 0.9), 3),
        },
        "elapsed_seconds": round(elapsed, 3),
        "output": {
            "per_file_csv": str(out_dir / "per_file.csv"),
            "summary_json": str(out_dir / "summary.json"),
        },
    }

    _write_csv(out_dir / "per_file.csv", per_file)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
