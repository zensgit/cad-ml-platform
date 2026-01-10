#!/usr/bin/env python3
"""Audit vector store persistence and layout metadata.

Outputs a Markdown report summarizing FAISS/Redis state and migration guidance.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from typing import Any, Dict, List, Optional


def _utc_now() -> str:
    return _dt.datetime.utcnow().isoformat() + "Z"


def _format_bytes(num: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024:
            return f"{num:.2f} {unit}"
        num /= 1024
    return f"{num:.2f} PB"


def _check_faiss(path: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": path,
        "exists": False,
        "size_bytes": 0,
        "modified_at": None,
        "dimension": None,
        "count": None,
        "error": None,
    }
    if not os.path.exists(path):
        return info
    info["exists"] = True
    info["size_bytes"] = os.path.getsize(path)
    info["modified_at"] = _dt.datetime.utcfromtimestamp(os.path.getmtime(path)).isoformat() + "Z"
    try:
        import faiss  # type: ignore
        idx = faiss.read_index(path)
        info["dimension"] = int(getattr(idx, "d", 0))
        info["count"] = int(getattr(idx, "ntotal", 0))
    except Exception as exc:  # pragma: no cover - optional dependency
        info["error"] = str(exc)
    return info


def _check_redis(url: Optional[str], sample_limit: int) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "url": url,
        "reachable": False,
        "vector_key_count": 0,
        "samples": [],
        "error": None,
    }
    if not url:
        info["error"] = "REDIS_URL not provided"
        return info
    try:
        import redis  # type: ignore
        client = redis.Redis.from_url(url, decode_responses=True)
        client.ping()
    except Exception as exc:
        info["error"] = str(exc)
        return info
    info["reachable"] = True
    cursor = 0
    keys: List[str] = []
    count = 0
    try:
        while True:
            cursor, batch = client.scan(cursor=cursor, match="vector:*", count=500)
            count += len(batch)
            if len(keys) < sample_limit:
                keys.extend(batch[: max(sample_limit - len(keys), 0)])
            if cursor == 0:
                break
    except Exception as exc:
        info["error"] = str(exc)
        return info
    info["vector_key_count"] = count
    samples: List[Dict[str, Any]] = []
    for key in keys:
        try:
            data = client.hgetall(key)
            raw_vec = data.get("v", "")
            vec_len = len([x for x in raw_vec.split(",") if x]) if raw_vec else 0
            raw_meta = data.get("m", "{}")
            meta = json.loads(raw_meta) if raw_meta else {}
            samples.append(
                {
                    "key": key,
                    "dimension": vec_len,
                    "feature_version": meta.get("feature_version"),
                    "vector_layout": meta.get("vector_layout"),
                    "ts": data.get("ts"),
                }
            )
        except Exception:
            continue
    info["samples"] = samples
    return info


def _build_report(
    backend: str,
    feature_version: str,
    faiss_info: Dict[str, Any],
    redis_info: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# Vector Store Layout Audit")
    lines.append("")
    lines.append(f"- Timestamp: {_utc_now()}")
    lines.append(f"- VECTOR_STORE_BACKEND: `{backend}`")
    lines.append(f"- FEATURE_VERSION: `{feature_version}`")
    lines.append("")
    lines.append("## FAISS Index")
    lines.append(f"- Path: `{faiss_info['path']}`")
    lines.append(f"- Exists: `{faiss_info['exists']}`")
    if faiss_info["exists"]:
        lines.append(f"- Size: `{_format_bytes(int(faiss_info['size_bytes']))}`")
        lines.append(f"- Modified: `{faiss_info['modified_at']}`")
        if faiss_info.get("dimension") is not None:
            lines.append(f"- Dimension: `{faiss_info['dimension']}`")
        if faiss_info.get("count") is not None:
            lines.append(f"- Vector count: `{faiss_info['count']}`")
    if faiss_info.get("error"):
        lines.append(f"- Error: `{faiss_info['error']}`")
    lines.append("")
    lines.append("## Redis Vectors")
    lines.append(f"- URL: `{redis_info.get('url')}`")
    lines.append(f"- Reachable: `{redis_info.get('reachable')}`")
    if redis_info.get("error"):
        lines.append(f"- Error: `{redis_info.get('error')}`")
    lines.append(f"- Vector keys: `{redis_info.get('vector_key_count')}`")
    if redis_info.get("samples"):
        lines.append("")
        lines.append("### Sample Vectors")
        for sample in redis_info["samples"]:
            lines.append(
                f"- `{sample['key']}` dim={sample['dimension']} "
                f"version={sample.get('feature_version')} layout={sample.get('vector_layout')} ts={sample.get('ts')}"
            )
    lines.append("")
    lines.append("## Migration Guidance")
    if redis_info.get("reachable") and redis_info.get("vector_key_count", 0) > 0:
        lines.append("- Redis contains persisted vectors; plan a full migration to the new layout.")
    else:
        lines.append("- No persisted Redis vectors detected (or Redis unreachable in this environment).")
    if faiss_info.get("exists"):
        lines.append("- FAISS index exists; must be rebuilt or re-exported after vector migration.")
    else:
        lines.append("- No FAISS index file detected at the default path.")
    lines.append("- If classification models consume vectors, retrain or regenerate features after layout change.")
    lines.append("")
    lines.append("## Suggested Steps (Prod)")
    lines.append("1. Freeze writes or route new writes to a fresh index.")
    lines.append("2. Export IDs from Redis `vector:*` keys or analysis cache index.")
    lines.append("3. Recompute vectors with `FeatureExtractor.flatten()` order and update meta `vector_layout`.")
    lines.append("4. Rebuild FAISS index from migrated vectors and verify dimension consistency.")
    lines.append("5. Run similarity regression tests and compare score distributions.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit vector store layout state")
    parser.add_argument("--output", default="reports/VECTOR_STORE_LAYOUT_AUDIT.md")
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL"))
    parser.add_argument("--faiss-path", default=os.getenv("FAISS_INDEX_PATH", "data/faiss_index.bin"))
    parser.add_argument("--sample-limit", type=int, default=5)
    args = parser.parse_args()

    backend = os.getenv("VECTOR_STORE_BACKEND", "memory")
    feature_version = os.getenv("FEATURE_VERSION", "v1")

    faiss_info = _check_faiss(args.faiss_path)
    redis_info = _check_redis(args.redis_url, args.sample_limit)

    report = _build_report(backend, feature_version, faiss_info, redis_info)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print(args.output)


if __name__ == "__main__":
    main()
