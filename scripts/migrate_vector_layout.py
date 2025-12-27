#!/usr/bin/env python3
"""Migrate legacy vector layout stored in Redis.

Legacy layout: geometric_all + semantic
Canonical layout: base_geometric + semantic + geometric_extensions
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core.feature_extractor import FeatureExtractor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate vector layout in Redis")
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL"))
    parser.add_argument("--dry-run", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=5)
    parser.add_argument("--assume-version", default=None)
    parser.add_argument("--output", default="reports/VECTOR_LAYOUT_MIGRATION.md")
    return parser.parse_args()


def _load_meta(raw_meta: str) -> Dict[str, Any]:
    if not raw_meta:
        return {}
    try:
        return json.loads(raw_meta)
    except Exception:
        return {}


def _join_vector(vec: List[float]) -> str:
    return ",".join(str(float(x)) for x in vec)


def main() -> None:
    args = _parse_args()
    extractor = FeatureExtractor()
    backend_version = os.getenv("FEATURE_VERSION", "v1")
    url = args.redis_url

    report_lines: List[str] = []
    report_lines.append("# Vector Layout Migration Report")
    report_lines.append("")
    report_lines.append(f"- redis_url: `{url}`")
    report_lines.append(f"- dry_run: `{bool(args.dry_run)}`")
    report_lines.append(f"- assume_version: `{args.assume_version}`")
    report_lines.append(f"- env_feature_version: `{backend_version}`")
    report_lines.append("")

    if not url:
        report_lines.append("- Error: REDIS_URL not provided")
        _write_report(args.output, report_lines)
        return

    try:
        import redis  # type: ignore
    except Exception as exc:
        report_lines.append(f"- Error: redis client unavailable ({exc})")
        _write_report(args.output, report_lines)
        return

    client = redis.Redis.from_url(url, decode_responses=True)
    try:
        client.ping()
    except Exception as exc:
        report_lines.append(f"- Error: Redis ping failed ({exc})")
        _write_report(args.output, report_lines)
        return

    counts = {
        "total": 0,
        "migrated": 0,
        "skipped": 0,
        "errors": 0,
        "already_canonical": 0,
        "missing_version": 0,
    }
    by_version: Dict[str, int] = {}
    samples: List[str] = []
    cursor = 0
    processed = 0
    while True:
        cursor, batch = client.scan(cursor=cursor, match="vector:*", count=500)
        for key in batch:
            counts["total"] += 1
            if args.limit and processed >= args.limit:
                break
            processed += 1
            data = client.hgetall(key)
            raw_vec = data.get("v", "")
            if not raw_vec:
                counts["skipped"] += 1
                continue
            vec = [float(x) for x in raw_vec.split(",") if x]
            meta = _load_meta(data.get("m", ""))
            version = meta.get("feature_version") or args.assume_version or backend_version
            layout = meta.get("vector_layout")
            if layout == "base_sem_ext_v1":
                counts["already_canonical"] += 1
                counts["skipped"] += 1
                continue
            if not meta.get("feature_version") and args.assume_version is None:
                counts["missing_version"] += 1
                counts["skipped"] += 1
                continue
            try:
                new_vec = extractor.reorder_legacy_vector(vec, version)
            except Exception:
                counts["errors"] += 1
                continue

            if args.dry_run:
                counts["migrated"] += 1
            else:
                meta["feature_version"] = version
                meta["vector_layout"] = "base_sem_ext_v1"
                client.hset(
                    key,
                    mapping={
                        "v": _join_vector(new_vec),
                        "m": json.dumps(meta),
                        "ts": data.get("ts", ""),
                    },
                )
                counts["migrated"] += 1
                if len(samples) < args.sample_limit:
                    samples.append(f"{key} dim={len(new_vec)} version={version}")

            by_version[version] = by_version.get(version, 0) + 1

        if args.limit and processed >= args.limit:
            break
        if cursor == 0:
            break

    report_lines.append("## Summary")
    for k, v in counts.items():
        report_lines.append(f"- {k}: `{v}`")
    report_lines.append("")
    if by_version:
        report_lines.append("## Versions")
        for ver, cnt in sorted(by_version.items()):
            report_lines.append(f"- {ver}: `{cnt}`")
        report_lines.append("")
    if samples:
        report_lines.append("## Samples")
        for s in samples:
            report_lines.append(f"- {s}")
        report_lines.append("")
    report_lines.append("## Notes")
    report_lines.append("- Rebuild FAISS index after applying layout migration.")
    report_lines.append("- Run similarity regression checks after migration.")
    report_lines.append("")

    _write_report(args.output, report_lines)


def _write_report(path: str, lines: List[str]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(path)


if __name__ == "__main__":
    main()
