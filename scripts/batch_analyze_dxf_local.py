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


def _collect_from_manifest(manifest: Path, dxf_dir: Path) -> List[Path]:
    files: List[Path] = []
    seen: set[Path] = set()
    with manifest.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            relative_path = (row.get("relative_path") or "").strip()
            file_name = (row.get("file_name") or row.get("file") or "").strip()
            source_dir = (row.get("source_dir") or "").strip()
            candidates: List[Path] = []
            if relative_path:
                candidates.append(dxf_dir / relative_path)
            if file_name:
                candidates.append(dxf_dir / file_name)
            if source_dir and file_name:
                candidates.append(dxf_dir / source_dir / file_name)
            for candidate in candidates:
                if candidate in seen:
                    break
                if candidate.exists():
                    files.append(candidate)
                    seen.add(candidate)
                    break
    return files


def _sanitize_file_column(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a copy of rows with `file` column reduced to basename.

    Batch results often contain absolute local paths; we keep a sanitized copy
    suitable for committing into git.
    """

    sanitized: List[Dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        raw = copied.get("file")
        if isinstance(raw, str) and raw:
            copied["file"] = Path(raw).name
        sanitized.append(copied)
    return sanitized


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch analyze DXF via local TestClient")
    parser.add_argument("--dxf-dir", required=True, help="DXF directory")
    parser.add_argument("--manifest", help="Optional manifest CSV to select files")
    parser.add_argument("--output-dir", default="reports/experiments/20260122/batch_analysis")
    parser.add_argument("--max-files", type=int, default=200)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--min-confidence", type=float, default=0.6)
    args = parser.parse_args()

    dxf_dir = Path(args.dxf_dir)
    if not dxf_dir.exists():
        raise SystemExit(f"DXF dir not found: {dxf_dir}")

    manifest_path = Path(args.manifest) if args.manifest else None
    if manifest_path:
        if not manifest_path.exists():
            raise SystemExit(f"Manifest not found: {manifest_path}")
        files = _collect_from_manifest(manifest_path, dxf_dir)
    else:
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
    (out_dir / ".gitignore").write_text(
        "\n".join(
            [
                # These include absolute local paths; keep untracked.
                "batch_results.csv",
                "batch_low_confidence.csv",
                "",
            ]
        ),
        encoding="utf-8",
    )
    results_path = out_dir / "batch_results.csv"
    results_sanitized_path = out_dir / "batch_results_sanitized.csv"
    low_conf_path = out_dir / "batch_low_confidence.csv"
    low_conf_sanitized_path = out_dir / "batch_low_confidence_sanitized.csv"
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
        filename_pred = classification.get("filename_prediction", {}) or {}
        hybrid_decision = classification.get("hybrid_decision", {}) or {}
        titleblock_pred = classification.get("titleblock_prediction", {}) or {}
        fusion = classification.get("fusion_decision", {}) or {}
        soft_override = classification.get("soft_override_suggestion", {}) or {}
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
            "graph2d_temperature": graph2d.get("temperature"),
            "graph2d_temperature_source": graph2d.get("temperature_source"),
            "graph2d_is_drawing_type": graph2d.get("is_drawing_type"),
            "graph2d_ensemble_enabled": graph2d.get("ensemble_enabled"),
            "graph2d_ensemble_size": graph2d.get("ensemble_size"),
            "graph2d_voting": graph2d.get("voting"),
            "filename_label": filename_pred.get("label"),
            "filename_confidence": filename_pred.get("confidence"),
            "filename_match_type": filename_pred.get("match_type"),
            "filename_extracted_name": filename_pred.get("extracted_name"),
            "hybrid_label": hybrid_decision.get("label"),
            "hybrid_confidence": hybrid_decision.get("confidence"),
            "hybrid_source": hybrid_decision.get("source"),
            "hybrid_path": ";".join(hybrid_decision.get("decision_path", []) or []),
            "titleblock_label": titleblock_pred.get("label"),
            "titleblock_confidence": titleblock_pred.get("confidence"),
            "titleblock_part_name": (
                titleblock_pred.get("title_block_info", {}) or {}
            ).get("part_name"),
            "titleblock_drawing_number": (
                titleblock_pred.get("title_block_info", {}) or {}
            ).get("drawing_number"),
            "titleblock_material": (
                titleblock_pred.get("title_block_info", {}) or {}
            ).get("material"),
            "titleblock_raw_texts_count": (
                titleblock_pred.get("title_block_info", {}) or {}
            ).get("raw_texts_count"),
            "titleblock_region_entities_count": (
                titleblock_pred.get("title_block_info", {}) or {}
            ).get("region_entities_count"),
            "fusion_label": fusion.get("primary_label"),
            "fusion_confidence": fusion.get("confidence"),
            "soft_override_eligible": soft_override.get("eligible"),
            "soft_override_label": soft_override.get("label"),
            "soft_override_confidence": soft_override.get("confidence"),
            "soft_override_threshold": soft_override.get("threshold"),
            "soft_override_reason": soft_override.get("reason"),
        })
        stats["success"] += 1
        if soft_override.get("eligible"):
            stats["soft_override_candidates"] += 1

    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    sanitized_rows = _sanitize_file_column(rows)
    with results_sanitized_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sanitized_rows[0].keys())
        writer.writeheader()
        writer.writerows(sanitized_rows)

    low_conf = [
        r
        for r in rows
        if r.get("status") == "ok"
        and float(r.get("confidence") or 0) <= args.min_confidence
    ]
    if low_conf:
        with low_conf_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=low_conf[0].keys())
            writer.writeheader()
            writer.writerows(low_conf)
        sanitized_low_conf = _sanitize_file_column(low_conf)
        with low_conf_sanitized_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sanitized_low_conf[0].keys())
            writer.writeheader()
            writer.writerows(sanitized_low_conf)

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
        "soft_override_candidates": stats.get("soft_override_candidates", 0),
        "sample_size": len(files),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"results: {results_path}")
    print(f"results_sanitized: {results_sanitized_path}")
    print(f"low_conf: {low_conf_path if low_conf else 'none'}")
    print(f"low_conf_sanitized: {low_conf_sanitized_path if low_conf else 'none'}")
    print(f"summary: {summary_path}")
    print(f"label_dist: {label_dist_path}")


if __name__ == "__main__":
    main()
