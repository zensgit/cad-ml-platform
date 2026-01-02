"""2D dedup pipeline (shared by API routes and async workers).

This module contains the orchestration logic for calling `dedupcad-vision`
for L1/L2 recall and (optionally) applying local L4 "precision" verification
using v2 geometry JSON.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import anyio

from src.core.dedupcad_precision import GeomJsonStoreProtocol, PrecisionVerifier
from src.core.dedupcad_precision.vendor.json_diff import compare_json
from src.core.dedupcad_vision import DedupCadVisionClient

_VERSION_GATE_MODES = {"off", "auto", "file_name", "meta"}
_VERSION_SUFFIX_RE = re.compile(r"(?:[_\-\s]?v\d+)$", re.IGNORECASE)


def _normalize_weights(visual_w: float, geom_w: float) -> tuple[float, float]:
    if visual_w < 0 or geom_w < 0:
        raise ValueError("weights must be >= 0")
    total = visual_w + geom_w
    if total <= 0:
        raise ValueError("weights sum must be > 0")
    return visual_w / total, geom_w / total


def _extract_meta_drawing_key(v2: Dict[str, Any]) -> Optional[str]:
    meta = v2.get("meta")
    if not isinstance(meta, dict):
        return None
    for key in (
        "drawing_number",
        "drawing_no",
        "drawingNo",
        "drawingNumber",
        "drawing_id",
        "drawingId",
        "number",
    ):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_file_stem_key(file_name: Optional[str]) -> Optional[str]:
    if not file_name:
        return None
    stem = Path(str(file_name)).stem.strip()
    if not stem:
        return None
    if _VERSION_SUFFIX_RE.search(stem) is None:
        return None
    base = _VERSION_SUFFIX_RE.sub("", stem).rstrip(" _-").strip()
    return base if base else None


def _apply_precision_l4(
    response: Dict[str, Any],
    query_geom: Dict[str, Any],
    query_file_name: Optional[str],
    geom_store: GeomJsonStoreProtocol,
    precision_verifier: PrecisionVerifier,
    max_results: int,
    precision_profile: Optional[str],
    version_gate: Optional[str],
    precision_top_n: int,
    precision_visual_weight: float,
    precision_geom_weight: float,
    duplicate_threshold: float,
    similar_threshold: float,
    precision_compute_diff: bool,
    precision_diff_top_n: int,
    precision_diff_max_paths: int,
) -> Dict[str, Any]:
    """Apply local v2 JSON precision scoring to a vision response (sync)."""
    precision_top_n = int(precision_top_n)
    if precision_top_n <= 0:
        precision_top_n = 1

    visual_w, geom_w = _normalize_weights(
        float(precision_visual_weight),
        float(precision_geom_weight),
    )

    matches: List[Dict[str, Any]] = list(response.get("duplicates") or []) + list(
        response.get("similar") or []
    )
    if not matches:
        return response

    raw_profile = str(precision_profile or "strict").strip().lower() or "strict"
    gate_mode = str(version_gate or "off").strip().lower() or "off"
    if gate_mode not in _VERSION_GATE_MODES:
        raise ValueError(f"Invalid version_gate: {version_gate}")
    query_key: Optional[str] = None
    gate_mode_effective = "off"
    if raw_profile == "version" and gate_mode != "off":
        query_key_meta = _extract_meta_drawing_key(query_geom)
        query_key_file = _extract_file_stem_key(query_file_name)
        if gate_mode == "meta":
            query_key = query_key_meta
            gate_mode_effective = "meta" if query_key else "off"
        elif gate_mode == "file_name":
            query_key = query_key_file
            gate_mode_effective = "file_name" if query_key else "off"
        else:  # auto
            if query_key_meta:
                query_key = query_key_meta
                gate_mode_effective = "meta"
            elif query_key_file:
                query_key = query_key_file
                gate_mode_effective = "file_name"

    missing_geom = 0
    verified = 0
    verified_hashes: set[str] = set()
    gated_out = 0

    duplicate_threshold = float(duplicate_threshold)
    similar_threshold = float(similar_threshold)
    if not (0.0 <= similar_threshold <= duplicate_threshold <= 1.0):
        raise ValueError(
            "invalid thresholds: require 0 <= similar_threshold <= duplicate_threshold <= 1"
        )

    matches_sorted = sorted(
        matches,
        key=lambda match: float(match.get("similarity") or 0.0),
        reverse=True,
    )
    diff_budget = int(precision_diff_top_n)
    if diff_budget < 0:
        diff_budget = 0
    diff_max_paths = int(precision_diff_max_paths)
    if diff_max_paths <= 0:
        diff_max_paths = 200

    def _diff_view(v2: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key in ("layers", "blocks", "dimensions", "hatches", "text_content"):
            if key in v2:
                out[key] = v2.get(key)
        return out

    for match in matches_sorted[: max(1, precision_top_n)]:
        file_hash = str(match.get("file_hash") or "")
        if not file_hash:
            continue

        if gate_mode_effective == "file_name" and query_key:
            cand_key = _extract_file_stem_key(str(match.get("file_name") or ""))
            if cand_key and cand_key != query_key:
                match["similarity"] = 0.0
                match["verdict"] = "different"
                levels = match.setdefault("levels", {})
                l4 = levels.setdefault("l4", {})
                l4["version_gate"] = {
                    "mode": gate_mode_effective,
                    "query_key": query_key,
                    "candidate_key": cand_key,
                    "match": False,
                }
                gated_out += 1
                continue

        candidate_geom = geom_store.load(file_hash)
        if candidate_geom is None:
            missing_geom += 1
            continue

        if gate_mode_effective == "meta" and query_key:
            cand_key = _extract_meta_drawing_key(candidate_geom)
            if cand_key and cand_key != query_key:
                match["similarity"] = 0.0
                match["verdict"] = "different"
                levels = match.setdefault("levels", {})
                l4 = levels.setdefault("l4", {})
                l4["version_gate"] = {
                    "mode": gate_mode_effective,
                    "query_key": query_key,
                    "candidate_key": cand_key,
                    "match": False,
                }
                gated_out += 1
                continue

        precision = precision_verifier.score_pair(
            query_geom,
            candidate_geom,
            profile=precision_profile,
        )
        visual_sim = float(match.get("similarity") or 0.0)
        final_sim = (visual_w * visual_sim) + (geom_w * precision.score)

        match["visual_similarity"] = visual_sim
        match["precision_score"] = precision.score
        match["precision_breakdown"] = precision.breakdown
        match["similarity"] = float(final_sim)

        levels = match.setdefault("levels", {})
        l4 = levels.setdefault("l4", {})
        l4["precision_score"] = precision.score
        l4["precision_breakdown"] = precision.breakdown
        l4["precision_profile"] = precision_profile or "strict"
        l4["geom_hash_left"] = precision.geom_hash_left
        l4["geom_hash_right"] = precision.geom_hash_right
        if precision_compute_diff and diff_budget > 0:
            try:
                diff, diff_sim = compare_json(
                    _diff_view(query_geom),
                    _diff_view(candidate_geom),
                    case_insensitive=True,
                    list_path_modes={
                        "dimensions": "unordered",
                        "hatches": "unordered",
                    },
                    max_diff_paths=diff_max_paths,
                )
                match["precision_diff"] = diff
                match["precision_diff_similarity"] = float(diff_sim)
                l4["precision_diff_similarity"] = float(diff_sim)
            except Exception as e:
                match["precision_diff"] = {"meta": {"error": str(e)}}
                match["precision_diff_similarity"] = None
                l4["precision_diff_similarity"] = None
            diff_budget -= 1

        try:
            match["match_level"] = max(int(match.get("match_level") or 0), 4)
        except Exception:
            match["match_level"] = 4

        verified += 1
        verified_hashes.add(file_hash)

    for match in matches:
        file_hash = str(match.get("file_hash") or "")
        if file_hash and file_hash in verified_hashes:
            continue
        if match.get("precision_score") is not None:
            continue
        visual_sim = float(match.get("similarity") or 0.0)
        match["visual_similarity"] = match.get("visual_similarity") or visual_sim
        match["similarity"] = float(visual_w * visual_sim)

    dup_th = duplicate_threshold
    sim_th = similar_threshold
    new_duplicates: List[Dict[str, Any]] = []
    new_similar: List[Dict[str, Any]] = []
    for match in matches:
        sim = float(match.get("similarity") or 0.0)
        if sim >= dup_th:
            match["verdict"] = "duplicate"
            new_duplicates.append(match)
        elif sim >= sim_th:
            match["verdict"] = "similar"
            new_similar.append(match)
        else:
            match["verdict"] = "different"

    new_duplicates.sort(key=lambda match: float(match.get("similarity") or 0.0), reverse=True)
    new_similar.sort(key=lambda match: float(match.get("similarity") or 0.0), reverse=True)

    response["duplicates"] = new_duplicates[:max_results]
    response["similar"] = new_similar[:max_results]
    response["total_matches"] = len(response["duplicates"]) + len(response["similar"])
    if verified:
        response["final_level"] = max(int(response.get("final_level") or 0), 4)
    else:
        response["final_level"] = response.get("final_level")

    warnings = response.setdefault("warnings", [])
    if missing_geom:
        warnings.append(f"precision_missing_geom_json:{missing_geom}")
    if gated_out:
        warnings.append(f"precision_version_gate_filtered:{gated_out}")
    if verified:
        warnings.append(f"precision_verified:{verified}")

    return response


async def run_dedup_2d_pipeline(
    *,
    client: DedupCadVisionClient,
    geom_store: GeomJsonStoreProtocol,
    precision_verifier: PrecisionVerifier,
    file_name: str,
    file_bytes: bytes,
    content_type: str,
    query_geom: Optional[Dict[str, Any]],
    mode: str,
    max_results: int,
    compute_diff: bool,
    enable_ml: bool,
    enable_geometric: bool,
    enable_precision: bool,
    precision_profile: Optional[str],
    version_gate: Optional[str],
    precision_top_n: int,
    precision_visual_weight: float,
    precision_geom_weight: float,
    precision_compute_diff: bool,
    precision_diff_top_n: int,
    precision_diff_max_paths: int,
    duplicate_threshold: float,
    similar_threshold: float,
) -> Dict[str, Any]:
    response = await client.search_2d(
        file_name=file_name or "unknown",
        file_bytes=file_bytes,
        content_type=content_type or "application/octet-stream",
        mode=mode,
        max_results=max_results,
        compute_diff=compute_diff,
        enable_ml=enable_ml,
        enable_geometric=enable_geometric,
    )

    if query_geom is None:
        return response
    if not (enable_precision or enable_geometric or mode == "precise"):
        return response

    precision_start = time.perf_counter()
    response = await anyio.to_thread.run_sync(
        _apply_precision_l4,
        response,
        query_geom,
        file_name,
        geom_store,
        precision_verifier,
        max_results,
        precision_profile,
        version_gate,
        precision_top_n,
        precision_visual_weight,
        precision_geom_weight,
        duplicate_threshold,
        similar_threshold,
        precision_compute_diff,
        precision_diff_top_n,
        precision_diff_max_paths,
    )
    precision_ms = (time.perf_counter() - precision_start) * 1000
    timing = response.setdefault("timing", {})
    timing["precision_ms"] = precision_ms
    try:
        timing["l4_ms"] = float(timing.get("l4_ms") or 0.0) + precision_ms
    except Exception:
        timing["l4_ms"] = precision_ms
    try:
        timing["total_ms"] = float(timing.get("total_ms") or 0.0) + precision_ms
    except Exception:
        timing["total_ms"] = precision_ms
    return response


__all__ = ["run_dedup_2d_pipeline"]
