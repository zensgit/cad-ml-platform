#!/usr/bin/env python3
from __future__ import annotations

"""
Batch search + dedup report generator for 2D CAD.

Typical workflow:
  1) Export/prepare PNG + *.v2.json (plugin/accoreconsole recommended)
  2) (Optional) index them into cad-ml-platform
  3) Run this script to search each item and write:
     - matches.csv (per-query ranked matches)
     - groups.json / groups.csv (duplicate clusters within the input set)
     - summary.json
"""

import argparse
import base64
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.dedupcad_precision.verifier import PrecisionVerifier  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".pdf"}


@dataclass(frozen=True)
class DatasetItem:
    image_path: Path
    geom_json_path: Optional[Path]
    file_hash: str


def _iter_images(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _find_geom_json(image_path: Path) -> Optional[Path]:
    candidates = [
        image_path.with_suffix(".json"),
        image_path.with_name(f"{image_path.stem}.v2.json"),
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class _UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent: Dict[str, str] = {x: x for x in items}
        self.rank: Dict[str, int] = {x: 0 for x in items}

    def find(self, x: str) -> str:
        p = self.parent.get(x, x)
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent.get(x, x)

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank.get(ra, 0) < self.rank.get(rb, 0):
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank.get(ra, 0) == self.rank.get(rb, 0):
            self.rank[ra] = self.rank.get(ra, 0) + 1


def _ranked_matches(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    for key in ("duplicates", "similar"):
        block = response.get(key)
        if isinstance(block, list):
            matches.extend([m for m in block if isinstance(m, dict)])
    matches.sort(key=lambda m: float(m.get("similarity") or 0.0), reverse=True)
    return matches


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def index_dataset(
    items: Sequence[DatasetItem],
    *,
    base_url: str,
    api_key: str,
    user_name: str,
    upload_to_s3: bool,
    require_json: bool,
    rebuild_index: bool,
) -> None:
    index_endpoint = base_url.rstrip("/") + "/api/v1/dedup/2d/index/add"
    rebuild_endpoint = base_url.rstrip("/") + "/api/v1/dedup/2d/index/rebuild"
    headers = {"X-API-Key": api_key}
    params = {"user_name": user_name, "upload_to_s3": "true" if upload_to_s3 else "false"}

    ok = 0
    for it in items:
        if it.geom_json_path is None:
            if require_json:
                raise SystemExit(f"Missing geom_json for {it.image_path}")
            continue
        with open(it.image_path, "rb") as f_img, open(it.geom_json_path, "rb") as f_json:
            files = {
                "file": (it.image_path.name, f_img, "application/octet-stream"),
                "geom_json": (it.geom_json_path.name, f_json, "application/json"),
            }
            resp = requests.post(index_endpoint, headers=headers, params=params, files=files, timeout=300)
        if resp.status_code // 100 != 2:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise SystemExit(f"index failed: {it.image_path} -> {resp.status_code}: {detail}")
        ok += 1

    if rebuild_index and ok > 0:
        resp = requests.post(rebuild_endpoint, headers=headers, timeout=120)
        if resp.status_code // 100 != 2:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise SystemExit(f"index rebuild failed: {resp.status_code}: {detail}")


def load_dataset(input_dir: Path, *, require_json: bool) -> List[DatasetItem]:
    items: List[DatasetItem] = []
    for img in _iter_images(input_dir):
        geom = _find_geom_json(img)
        if geom is None and require_json:
            raise SystemExit(f"Missing geom_json for {img}")
        items.append(DatasetItem(image_path=img, geom_json_path=geom, file_hash=_sha256_file(img)))
    if not items:
        raise SystemExit(f"No images found under: {input_dir}")
    return items


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_geom_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _local_l4_search_all_pairs(
    items: Sequence[DatasetItem],
    *,
    precision_profile: Optional[str],
    top_k: int,
    min_similarity: float,
    duplicate_threshold: float,
    similar_threshold: float,
    save_responses: bool,
    responses_jsonl_path: Path,
) -> Tuple[
    Dict[str, List[Tuple[str, float]]],
    Optional[List[Dict[str, Any]]],
]:
    """Return adjacency list: query_hash -> [(candidate_hash, score)].

    Uses PrecisionVerifier only (no vision recall). Complexity is O(N^2).
    """
    if any(it.geom_json_path is None for it in items):
        missing = [str(it.image_path) for it in items if it.geom_json_path is None]
        raise SystemExit(f"local_l4 requires geom_json for all items, missing={len(missing)}")

    verifier = PrecisionVerifier()
    geoms: List[Dict[str, Any]] = []
    for it in items:
        assert it.geom_json_path is not None
        geoms.append(_load_geom_json(it.geom_json_path))

    adjacency: Dict[str, List[Tuple[str, float]]] = {it.file_hash: [] for it in items}
    n = len(items)
    for i in range(n):
        for j in range(i + 1, n):
            res = verifier.score_pair(geoms[i], geoms[j], profile=precision_profile)
            s = float(res.score)
            if s < float(min_similarity):
                continue
            hi = items[i].file_hash
            hj = items[j].file_hash
            adjacency[hi].append((hj, s))
            adjacency[hj].append((hi, s))

    for h in adjacency:
        adjacency[h].sort(key=lambda t: t[1], reverse=True)
        adjacency[h] = adjacency[h][: max(1, int(top_k))]

    responses: Optional[List[Dict[str, Any]]] = None
    if save_responses:
        dup_th = float(duplicate_threshold)
        sim_th = float(similar_threshold)
        responses = []
        with open(responses_jsonl_path, "w", encoding="utf-8") as f:
            for it in items:
                ranked = [{"file_hash": h2, "similarity": s} for h2, s in adjacency[it.file_hash]]
                duplicates = []
                similar = []
                for m in ranked:
                    s = float(m["similarity"])
                    if s >= dup_th:
                        m["verdict"] = "duplicate"
                        duplicates.append(m)
                    elif s >= sim_th:
                        m["verdict"] = "similar"
                        similar.append(m)
                    else:
                        m["verdict"] = "different"
                resp = {
                    "success": True,
                    "engine": "local_l4",
                    "query_hash": it.file_hash,
                    "total_matches": len(duplicates) + len(similar),
                    "duplicates": duplicates,
                    "similar": similar,
                    "final_level": 4,
                    "warnings": ["local_l4_only"],
                }
                record = {"query_hash": it.file_hash, "query_path": str(it.image_path), "response": resp}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                responses.append(resp)
    return adjacency, responses


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch 2D dedup search and report generation.")
    parser.add_argument("input_dir", type=Path, help="Directory with PNG/JPG/PDF and optional *.v2.json")
    parser.add_argument(
        "--base-url",
        default=os.getenv("CAD_ML_PLATFORM_URL", "http://localhost:8000"),
        help="cad-ml-platform base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("CAD_ML_PLATFORM_API_KEY", "test"),
        help="X-API-Key header value (default: %(default)s)",
    )
    parser.add_argument(
        "--user-name",
        default=os.getenv("USER", "batch"),
        help="Indexing user_name query param when --index is enabled (default: %(default)s)",
    )
    parser.add_argument("--upload-to-s3", action="store_true", help="upload_to_s3=true when indexing")

    parser.add_argument(
        "--engine",
        choices=["api", "local_l4"],
        default="api",
        help="Search engine: api (cad-ml-platform) or local_l4 (pairwise PrecisionVerifier) (default: %(default)s)",
    )
    parser.add_argument(
        "--require-json",
        action="store_true",
        help="Fail if any image has no matching JSON file",
    )
    parser.add_argument(
        "--index",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Index input dataset before searching (default: %(default)s)",
    )
    parser.add_argument(
        "--rebuild-index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trigger vision L1/L2 index rebuild after indexing (default: %(default)s)",
    )

    parser.add_argument("--mode", default="balanced", help="Search mode (fast|balanced|precise)")
    parser.add_argument("--max-results", type=int, default=50, help="Search max_results (default: %(default)s)")
    parser.add_argument(
        "--compute-diff",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request diff output from vision side (default: %(default)s)",
    )
    parser.add_argument(
        "--enable-precision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable L4 precision when query has geom_json (default: %(default)s)",
    )
    parser.add_argument(
        "--preset",
        choices=["strict", "version", "loose"],
        default=None,
        help="Convenience preset that sets thresholds/top-n unless you override them (default: %(default)s)",
    )
    parser.add_argument(
        "--precision-profile",
        choices=["strict", "version"],
        default=None,
        help="L4 precision scoring profile (strict|version). If omitted, derives from --preset (default: %(default)s)",
    )
    parser.add_argument(
        "--version-gate",
        choices=["off", "auto", "file_name", "meta"],
        default=None,
        help="When using precision_profile=version, optionally gate candidates by meta/file name "
        "(off|auto|file_name|meta). If omitted, derives from --preset (default: %(default)s)",
    )
    parser.add_argument("--precision-top-n", type=int, default=20)
    parser.add_argument("--precision-visual-weight", type=float, default=0.3)
    parser.add_argument("--precision-geom-weight", type=float, default=0.7)
    parser.add_argument(
        "--precision-compute-diff",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request L4 JSON diff output from cad-ml-platform (default: %(default)s)",
    )
    parser.add_argument(
        "--precision-diff-top-n",
        type=int,
        default=5,
        help="How many candidates per query to request L4 diff for (default: %(default)s)",
    )
    parser.add_argument(
        "--precision-diff-max-paths",
        type=int,
        default=200,
        help="Max diff paths returned per candidate (default: %(default)s)",
    )
    parser.add_argument(
        "--duplicate-threshold",
        type=float,
        default=0.95,
        help="Threshold for marking duplicates (default: %(default)s)",
    )
    parser.add_argument(
        "--similar-threshold",
        type=float,
        default=0.80,
        help="Threshold for marking similar (default: %(default)s)",
    )

    parser.add_argument(
        "--within-input-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only keep matches whose file_hash is inside input_dir (default: %(default)s)",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Keep top-k matches per query (default: %(default)s)")
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.0,
        help="Only write matches >= this similarity to CSV (default: %(default)s)",
    )

    parser.add_argument(
        "--group-rule",
        choices=["verdict", "threshold"],
        default="verdict",
        help="How to build duplicate graph edges (default: %(default)s)",
    )
    parser.add_argument(
        "--group-threshold",
        type=float,
        default=0.95,
        help="Used when --group-rule=threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--include-singletons",
        action="store_true",
        help="Include singleton groups in groups outputs",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/dedup_report"),
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--save-responses",
        action="store_true",
        help="Save full JSON response per query (jsonl) under output dir",
    )
    parser.add_argument(
        "--save-diff-images",
        action="store_true",
        help="Decode diff_image_base64 and save PNG files (requires --compute-diff)",
    )
    parser.add_argument(
        "--save-precision-diffs",
        action="store_true",
        help="Save L4 JSON diffs per match under output_dir/precision_diffs (requires --precision-compute-diff)",
    )
    args = parser.parse_args()

    presets = {
        "strict": {
            "mode": "balanced",
            "precision_profile": "strict",
            "precision_top_n": 20,
            "precision_visual_weight": 0.3,
            "precision_geom_weight": 0.7,
            "duplicate_threshold": 0.95,
            "similar_threshold": 0.80,
            "group_rule": "verdict",
            "group_threshold": 0.95,
        },
        "version": {
            "mode": "balanced",
            "precision_profile": "version",
            "version_gate": "auto",
            "precision_top_n": 50,
            "precision_visual_weight": 0.5,
            "precision_geom_weight": 0.5,
            "duplicate_threshold": 0.95,
            "similar_threshold": 0.70,
            "group_rule": "threshold",
            "group_threshold": 0.70,
        },
        "loose": {
            "mode": "balanced",
            "precision_profile": "version",
            "version_gate": "off",
            "precision_top_n": 50,
            "precision_visual_weight": 0.6,
            "precision_geom_weight": 0.4,
            "duplicate_threshold": 0.90,
            "similar_threshold": 0.50,
            "group_rule": "threshold",
            "group_threshold": 0.50,
        },
    }

    # Apply preset defaults only when the caller did not change the corresponding arg from its default.
    if args.preset is not None:
        preset = presets[str(args.preset)]
        if args.mode == "balanced":
            args.mode = preset["mode"]
        if args.precision_profile is None:
            args.precision_profile = preset["precision_profile"]
        if args.version_gate is None:
            args.version_gate = preset.get("version_gate")
        if float(args.precision_visual_weight) == 0.3 and float(args.precision_geom_weight) == 0.7:
            args.precision_visual_weight = float(preset["precision_visual_weight"])
            args.precision_geom_weight = float(preset["precision_geom_weight"])
        if float(args.duplicate_threshold) == 0.95:
            args.duplicate_threshold = float(preset["duplicate_threshold"])
        if float(args.similar_threshold) == 0.80:
            args.similar_threshold = float(preset["similar_threshold"])
        if int(args.precision_top_n) == 20:
            args.precision_top_n = int(preset["precision_top_n"])
        if args.group_rule == "verdict" and float(args.group_threshold) == 0.95:
            args.group_rule = preset["group_rule"]
            args.group_threshold = float(preset["group_threshold"])

    if not (0.0 <= float(args.similar_threshold) <= float(args.duplicate_threshold) <= 1.0):
        raise SystemExit("Invalid thresholds: require 0 <= similar_threshold <= duplicate_threshold <= 1")

    input_dir: Path = args.input_dir
    if not input_dir.exists():
        raise SystemExit(f"input_dir not found: {input_dir}")

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    matches_csv = out_dir / "matches.csv"
    groups_json = out_dir / "groups.json"
    groups_csv = out_dir / "groups.csv"
    summary_json = out_dir / "summary.json"
    responses_jsonl = out_dir / "responses.jsonl"
    diffs_dir = out_dir / "diff_images"
    if args.save_diff_images:
        diffs_dir.mkdir(parents=True, exist_ok=True)
    precision_diffs_dir: Optional[Path] = None
    if args.save_precision_diffs:
        precision_diffs_dir = out_dir / "precision_diffs"
        precision_diffs_dir.mkdir(parents=True, exist_ok=True)

    items = load_dataset(input_dir, require_json=bool(args.require_json))
    hash_to_item: Dict[str, DatasetItem] = {it.file_hash: it for it in items}
    input_hashes: Set[str] = set(hash_to_item.keys())

    if args.engine == "local_l4" and not args.require_json:
        raise SystemExit("--engine local_l4 requires --require-json")

    if args.index:
        index_dataset(
            items,
            base_url=args.base_url,
            api_key=args.api_key,
            user_name=args.user_name,
            upload_to_s3=args.upload_to_s3,
            require_json=bool(args.require_json),
            rebuild_index=bool(args.rebuild_index),
        )

    search_endpoint = args.base_url.rstrip("/") + "/api/v1/dedup/2d/search"
    headers = {"X-API-Key": args.api_key}

    edges: Set[Tuple[str, str]] = set()
    queries_ok = 0
    queries_failed = 0
    rows_written = 0

    with open(matches_csv, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=[
                "query_hash",
                "query_path",
                "query_file_name",
                "candidate_hash",
                "candidate_path",
                "candidate_file_name",
                "candidate_drawing_id",
                "similarity",
                "visual_similarity",
                "precision_score",
                "verdict",
                "match_level",
            ],
        )
        writer.writeheader()

        resp_f = open(responses_jsonl, "w", encoding="utf-8") if args.save_responses else None
        try:
            if args.engine == "local_l4":
                if resp_f is not None:
                    resp_f.close()
                    resp_f = None  # local_l4 writes responses itself
                adjacency, _ = _local_l4_search_all_pairs(
                    items,
                    precision_profile=args.precision_profile,
                    top_k=int(args.top_k),
                    min_similarity=float(args.min_similarity),
                    duplicate_threshold=float(args.duplicate_threshold),
                    similar_threshold=float(args.similar_threshold),
                    save_responses=bool(args.save_responses),
                    responses_jsonl_path=responses_jsonl,
                )
                dup_th = float(args.duplicate_threshold)
                sim_th = float(args.similar_threshold)
                for it in items:
                    queries_ok += 1
                    ranked_local = adjacency.get(it.file_hash, [])
                    for cand_hash, sim in ranked_local:
                        cand_item = hash_to_item.get(cand_hash)
                        verdict = "different"
                        if sim >= dup_th:
                            verdict = "duplicate"
                        elif sim >= sim_th:
                            verdict = "similar"
                        writer.writerow(
                            {
                                "query_hash": it.file_hash,
                                "query_path": str(it.image_path),
                                "query_file_name": it.image_path.name,
                                "candidate_hash": cand_hash,
                                "candidate_path": str(cand_item.image_path) if cand_item else "",
                                "candidate_file_name": cand_item.image_path.name if cand_item else "",
                                "candidate_drawing_id": "",
                                "similarity": float(sim),
                                "visual_similarity": "",
                                "precision_score": float(sim),
                                "verdict": verdict,
                                "match_level": 4,
                            }
                        )
                        rows_written += 1

                        should_link = False
                        if args.group_rule == "verdict":
                            should_link = verdict == "duplicate"
                        else:
                            should_link = float(sim) >= float(args.group_threshold)
                        if should_link:
                            a, b = (it.file_hash, cand_hash)
                            if a != b:
                                if a > b:
                                    a, b = b, a
                                edges.add((a, b))
                queries_failed = 0
            else:
                for it in items:
                    files: Dict[str, Tuple[str, Any, str]] = {}
                    files["file"] = (
                        it.image_path.name,
                        open(it.image_path, "rb"),
                        "application/octet-stream",
                    )
                    if args.enable_precision and it.geom_json_path is not None:
                        files["geom_json"] = (
                            it.geom_json_path.name,
                            open(it.geom_json_path, "rb"),
                            "application/json",
                        )

                    params = {
                        "mode": args.mode,
                        "max_results": str(int(args.max_results)),
                        "compute_diff": "true" if args.compute_diff else "false",
                        "enable_precision": "true" if args.enable_precision else "false",
                        "precision_top_n": str(int(args.precision_top_n)),
                        "precision_visual_weight": str(float(args.precision_visual_weight)),
                        "precision_geom_weight": str(float(args.precision_geom_weight)),
                        "precision_compute_diff": "true" if args.precision_compute_diff else "false",
                        "precision_diff_top_n": str(int(args.precision_diff_top_n)),
                        "precision_diff_max_paths": str(int(args.precision_diff_max_paths)),
                        "duplicate_threshold": str(float(args.duplicate_threshold)),
                        "similar_threshold": str(float(args.similar_threshold)),
                    }
                    if args.precision_profile is not None:
                        params["precision_profile"] = str(args.precision_profile)
                    if args.version_gate is not None:
                        params["version_gate"] = str(args.version_gate)

                    try:
                        t0 = time.perf_counter()
                        resp = requests.post(
                            search_endpoint, headers=headers, params=params, files=files, timeout=300
                        )
                        _ = time.perf_counter() - t0
                    finally:
                        for _, fh, _ct in files.values():
                            try:
                                fh.close()  # type: ignore[attr-defined]
                            except Exception:
                                pass

                    if resp.status_code // 100 != 2:
                        queries_failed += 1
                        try:
                            detail = resp.json()
                        except Exception:
                            detail = resp.text
                        print(f"[fail] search {it.image_path} -> {resp.status_code}: {detail}")
                        continue

                    data = resp.json()
                    queries_ok += 1

                    if resp_f is not None:
                        record = {
                            "query_hash": it.file_hash,
                            "query_path": str(it.image_path),
                            "response": data,
                        }
                        resp_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    ranked = _ranked_matches(data)
                    # Remove self
                    ranked = [m for m in ranked if str(m.get("file_hash") or "") != it.file_hash]

                    if args.within_input_only:
                        ranked = [m for m in ranked if str(m.get("file_hash") or "") in input_hashes]

                    # Top-K
                    ranked = ranked[: max(1, int(args.top_k))]

                    for m in ranked:
                        cand_hash = str(m.get("file_hash") or "")
                        sim = _safe_float(m.get("similarity"))
                        if sim < float(args.min_similarity):
                            continue

                        cand_item = hash_to_item.get(cand_hash)
                        writer.writerow(
                            {
                                "query_hash": it.file_hash,
                                "query_path": str(it.image_path),
                                "query_file_name": it.image_path.name,
                                "candidate_hash": cand_hash,
                                "candidate_path": str(cand_item.image_path) if cand_item else "",
                                "candidate_file_name": str(m.get("file_name") or ""),
                                "candidate_drawing_id": str(m.get("drawing_id") or ""),
                                "similarity": sim,
                                "visual_similarity": _safe_float(m.get("visual_similarity")),
                                "precision_score": _safe_float(m.get("precision_score")),
                                "verdict": str(m.get("verdict") or ""),
                                "match_level": int(m.get("match_level") or 0),
                            }
                        )
                        rows_written += 1

                        # Build duplicate edges for grouping
                        should_link = False
                        if args.group_rule == "verdict":
                            should_link = str(m.get("verdict") or "") == "duplicate"
                        else:
                            should_link = sim >= float(args.group_threshold)
                        if should_link and cand_hash in input_hashes and it.file_hash in input_hashes:
                            a, b = (it.file_hash, cand_hash)
                            if a != b:
                                if a > b:
                                    a, b = b, a
                                edges.add((a, b))

                        # Optional: decode and save diff image if present
                        if args.save_diff_images and args.compute_diff:
                            b64 = m.get("diff_image_base64")
                            if isinstance(b64, str) and b64:
                                try:
                                    raw = base64.b64decode(b64)
                                    out_path = diffs_dir / f"{it.file_hash}__{cand_hash}.png"
                                    out_path.write_bytes(raw)
                                except Exception:
                                    pass

                        # Optional: save L4 JSON diffs if present
                        if precision_diffs_dir is not None and args.precision_compute_diff:
                            diff_obj = m.get("precision_diff")
                            if isinstance(diff_obj, dict) and diff_obj:
                                try:
                                    record = {
                                        "query_hash": it.file_hash,
                                        "query_file_name": it.image_path.name,
                                        "candidate_hash": cand_hash,
                                        "candidate_file_name": str(m.get("file_name") or ""),
                                        "similarity": sim,
                                        "precision_score": _safe_float(m.get("precision_score")),
                                        "precision_diff_similarity": _safe_float(
                                            m.get("precision_diff_similarity")
                                        ),
                                        "precision_diff": diff_obj,
                                    }
                                    out_path = precision_diffs_dir / f"{it.file_hash}__{cand_hash}.json"
                                    out_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
                                except Exception:
                                    pass
        finally:
            if resp_f is not None:
                resp_f.close()

    uf = _UnionFind(input_hashes)
    for a, b in edges:
        uf.union(a, b)

    root_to_members: Dict[str, List[str]] = {}
    for h in sorted(input_hashes):
        root = uf.find(h)
        root_to_members.setdefault(root, []).append(h)

    groups: List[Dict[str, Any]] = []
    for idx, (root, members) in enumerate(sorted(root_to_members.items()), start=1):
        if (not args.include_singletons) and len(members) <= 1:
            continue
        groups.append(
            {
                "group_id": idx,
                "root": root,
                "size": len(members),
                "members": [
                    {
                        "file_hash": h,
                        "path": str(hash_to_item[h].image_path),
                        "json": str(hash_to_item[h].geom_json_path) if hash_to_item[h].geom_json_path else None,
                    }
                    for h in members
                    if h in hash_to_item
                ],
            }
        )

    # groups.json / groups.csv / summary.json
    _write_json(groups_json, groups)
    with open(groups_csv, "w", newline="", encoding="utf-8") as f_gc:
        w = csv.DictWriter(f_gc, fieldnames=["group_id", "file_hash", "path"])
        w.writeheader()
        for g in groups:
            gid = int(g["group_id"])
            for m in g.get("members") or []:
                w.writerow({"group_id": gid, "file_hash": m.get("file_hash"), "path": m.get("path")})

    summary = {
        "input_dir": str(input_dir),
        "items_total": len(items),
        "queries_ok": queries_ok,
        "queries_failed": queries_failed,
        "within_input_only": bool(args.within_input_only),
        "group_rule": args.group_rule,
        "group_threshold": float(args.group_threshold),
        "edges": len(edges),
        "groups": len(groups),
        "matches_rows": rows_written,
        "outputs": {
            "matches_csv": str(matches_csv),
            "groups_json": str(groups_json),
            "groups_csv": str(groups_csv),
            "summary_json": str(summary_json),
            "responses_jsonl": str(responses_jsonl) if args.save_responses else None,
            "diff_images_dir": str(diffs_dir) if args.save_diff_images else None,
            "precision_diffs_dir": str(precision_diffs_dir) if args.save_precision_diffs else None,
        },
    }
    _write_json(summary_json, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if queries_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
