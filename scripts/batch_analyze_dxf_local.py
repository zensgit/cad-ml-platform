#!/usr/bin/env python3
"""Run local /api/v1/analyze on a batch of DXF files via TestClient."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root on sys.path for TestClient usage
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from fastapi.testclient import TestClient
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"FastAPI TestClient import failed: {exc}")

try:
    from src.main import app
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Failed to import app: {exc}")


def _collect_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.dxf") if p.is_file()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch analyze DXF via local TestClient")
    parser.add_argument("--dxf-dir", required=True, help="DXF directory")
    parser.add_argument("--output-dir", default="reports/experiments/20260122/batch_analysis")
    parser.add_argument("--max-files", type=int, default=200)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--min-confidence", type=float, default=0.6)
    args = parser.parse_args()

    dxf_dir = Path(args.dxf_dir)
    if not dxf_dir.exists():
        raise SystemExit(f"DXF dir not found: {dxf_dir}")

    files = _collect_files(dxf_dir)
    if not files:
        raise SystemExit("No DXF files found")

    random.seed(args.seed)
    random.shuffle(files)
    files = files[: args.max_files]

    os.environ.setdefault("GRAPH2D_ENABLED", "true")
    os.environ.setdefault("GRAPH2D_FUSION_ENABLED", "true")
    os.environ.setdefault("FUSION_ANALYZER_ENABLED", "true")

    client = TestClient(app)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "batch_results.csv"
    low_conf_path = out_dir / "batch_low_confidence.csv"
    summary_path = out_dir / "summary.json"
    label_dist_path = out_dir / "label_distribution.csv"

    options = {"extract_features": True, "classify_parts": True}

    rows: List[Dict[str, Any]] = []
    stats = Counter()
    label_counts = Counter()
    label_conf = defaultdict(list)
    conf_buckets = Counter()

    for item in files:
        stats["total"] += 1
        try:
            payload = item.read_bytes()
        except Exception as exc:
            rows.append({"file": str(item), "status": "read_error", "error": str(exc)})
            stats["error"] += 1
            continue

        resp = client.post(
            "/api/v1/analyze/",
            files={"file": (item.name, payload, "application/dxf")},
            data={"options": json.dumps(options)},
            headers={"x-api-key": os.getenv("API_KEY", "test")},
        )

        if resp.status_code != 200:
            rows.append({
                "file": str(item),
                "status": "error",
                "error": resp.text,
            })
            stats["error"] += 1
            continue

        data = resp.json()
        classification = data.get("results", {}).get("classification", {})
        graph2d = classification.get("graph2d_prediction", {}) or {}
        fusion = classification.get("fusion_decision", {}) or {}
        part_type = classification.get("part_type")
        confidence = float(classification.get("confidence") or 0.0)

        label_counts[part_type] += 1
        label_conf[part_type].append(confidence)
        if confidence < 0.4:
            conf_buckets["lt_0_4"] += 1
        elif confidence < 0.6:
            conf_buckets["0_4_0_6"] += 1
        elif confidence < 0.8:
            conf_buckets["0_6_0_8"] += 1
        else:
            conf_buckets["gte_0_8"] += 1

        rows.append({
            "file": str(item),
            "status": "ok",
            "part_type": part_type,
            "confidence": f"{confidence:.3f}",
            "confidence_source": classification.get("confidence_source"),
            "rule_version": classification.get("rule_version"),
            "graph2d_label": graph2d.get("label"),
            "graph2d_confidence": graph2d.get("confidence"),
            "fusion_label": fusion.get("primary_label"),
            "fusion_confidence": fusion.get("confidence"),
        })
        stats["success"] += 1

    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    low_conf = [r for r in rows if r.get("status") == "ok" and float(r.get("confidence") or 0) <= args.min_confidence]
    if low_conf:
        with low_conf_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=low_conf[0].keys())
            writer.writeheader()
            writer.writerows(low_conf)

    with label_dist_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "count", "share", "avg_confidence"])
        writer.writeheader()
        for label, count in label_counts.most_common():
            avg_conf = sum(label_conf[label]) / max(1, len(label_conf[label]))
            writer.writerow({
                "label": label,
                "count": count,
                "share": f"{count / max(1, stats['success']):.3f}",
                "avg_confidence": f"{avg_conf:.3f}",
            })

    summary = {
        "total": stats["total"],
        "success": stats["success"],
        "error": stats["error"],
        "confidence_buckets": dict(conf_buckets),
        "label_counts": dict(label_counts),
        "low_confidence_count": len(low_conf),
        "sample_size": len(files),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"results: {results_path}")
    print(f"low_conf: {low_conf_path if low_conf else 'none'}")
    print(f"summary: {summary_path}")
    print(f"label_dist: {label_dist_path}")


if __name__ == "__main__":
    main()
