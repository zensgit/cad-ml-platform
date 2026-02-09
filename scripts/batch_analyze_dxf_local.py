#!/usr/bin/env python3
"""Run local /api/v1/analyze on a batch of DXF files via TestClient."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root on sys.path for TestClient usage
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep ezdxf cache out of $HOME by default (helpful for sandboxed environments).
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
# Avoid slow/fragile network checks for remote model hosters during local evaluation.
# Some upstream libraries treat only "1" as truthy for this flag.
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "1")

# Reduce noisy multipart parser warnings when TestClient uploads files.
logging.getLogger("python_multipart.multipart").setLevel(logging.ERROR)
logging.getLogger("python_multipart").setLevel(logging.ERROR)

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
    parser.add_argument(
        "--mask-filename",
        action="store_true",
        help=(
            "Upload DXFs with a masked/anonymous file name (e.g. file_0001.dxf) "
            "to evaluate behavior when filenames do not carry semantic labels."
        ),
    )
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
                # Per-file outputs may contain absolute local paths and/or file names.
                # Keep them untracked by default; commit only aggregated summaries.
                "batch_results.csv",
                "batch_results_sanitized.csv",
                "batch_low_confidence.csv",
                "batch_low_confidence_sanitized.csv",
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

    for idx, item in enumerate(files):
        stats["total"] += 1
        try:
            payload = item.read_bytes()
        except Exception as exc:
            rows.append({"file": str(item), "status": "read_error", "error": str(exc)})
            stats["error"] += 1
            continue

        upload_name = item.name
        if bool(args.mask_filename):
            upload_name = f"file_{idx+1:04d}{item.suffix.lower() or '.dxf'}"

        resp = client.post(
            "/api/v1/analyze/",
            files={"file": (upload_name, payload, "application/dxf")},
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
            "graph2d_status": graph2d.get("status"),
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
            "titleblock_status": titleblock_pred.get("status"),
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

    ok_rows = [r for r in rows if r.get("status") == "ok"]

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
    else:
        # Remove stale low-confidence outputs from previous runs.
        for path in (low_conf_path, low_conf_sanitized_path):
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass

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

    titleblock_status_counts: Counter[str] = Counter()
    titleblock_conf_all: List[float] = []
    titleblock_conf_nonzero: List[float] = []
    titleblock_texts_present = 0
    titleblock_part_name_present = 0
    titleblock_label_present = 0
    titleblock_any_signal_present = 0

    filename_match_type_counts: Counter[str] = Counter()
    filename_label_present = 0
    filename_conf_all: List[float] = []
    filename_conf_nonzero: List[float] = []
    filename_extracted_name_present = 0

    hybrid_source_counts: Counter[str] = Counter()
    hybrid_label_present = 0
    hybrid_conf_all: List[float] = []
    hybrid_conf_nonzero: List[float] = []

    graph2d_status_counts: Counter[str] = Counter()
    graph2d_label_present = 0
    graph2d_conf_all: List[float] = []
    graph2d_conf_nonzero: List[float] = []

    for row in ok_rows:
        status = (row.get("titleblock_status") or "").strip()
        titleblock_status_counts[status or "unknown"] += 1

        try:
            conf = float(row.get("titleblock_confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        titleblock_conf_all.append(conf)
        if conf > 0:
            titleblock_conf_nonzero.append(conf)

        part_name = (row.get("titleblock_part_name") or "").strip()
        drawing_number = (row.get("titleblock_drawing_number") or "").strip()
        material = (row.get("titleblock_material") or "").strip()
        label = (row.get("titleblock_label") or "").strip()

        try:
            raw_texts = int(row.get("titleblock_raw_texts_count") or 0)
        except (TypeError, ValueError):
            raw_texts = 0
        try:
            region_entities = int(row.get("titleblock_region_entities_count") or 0)
        except (TypeError, ValueError):
            region_entities = 0

        if raw_texts > 0 or region_entities > 0:
            titleblock_texts_present += 1
        if part_name:
            titleblock_part_name_present += 1
        if label:
            titleblock_label_present += 1
        if part_name or drawing_number or material:
            titleblock_any_signal_present += 1

        # --- Filename classifier signals (via HybridClassifier) ---
        filename_label = (row.get("filename_label") or "").strip()
        if filename_label:
            filename_label_present += 1

        filename_extracted = (row.get("filename_extracted_name") or "").strip()
        if filename_extracted:
            filename_extracted_name_present += 1

        match_type = (row.get("filename_match_type") or "").strip() or "unknown"
        filename_match_type_counts[match_type] += 1

        try:
            fconf = float(row.get("filename_confidence") or 0.0)
        except (TypeError, ValueError):
            fconf = 0.0
        filename_conf_all.append(fconf)
        if fconf > 0:
            filename_conf_nonzero.append(fconf)

        # --- Hybrid decision ---
        h_source = (row.get("hybrid_source") or "").strip() or "unknown"
        hybrid_source_counts[h_source] += 1
        h_label = (row.get("hybrid_label") or "").strip()
        if h_label:
            hybrid_label_present += 1
        try:
            hconf = float(row.get("hybrid_confidence") or 0.0)
        except (TypeError, ValueError):
            hconf = 0.0
        hybrid_conf_all.append(hconf)
        if hconf > 0:
            hybrid_conf_nonzero.append(hconf)

        # --- Graph2D prediction ---
        g_status = (row.get("graph2d_status") or "").strip() or "unknown"
        graph2d_status_counts[g_status] += 1
        g_label = (row.get("graph2d_label") or "").strip()
        if g_label:
            graph2d_label_present += 1
        try:
            gconf = float(row.get("graph2d_confidence") or 0.0)
        except (TypeError, ValueError):
            gconf = 0.0
        graph2d_conf_all.append(gconf)
        if gconf > 0:
            graph2d_conf_nonzero.append(gconf)

    def _mean(values: List[float]) -> float:
        return (sum(values) / len(values)) if values else 0.0

    def _median(values: List[float]) -> float:
        return float(statistics.median(values)) if values else 0.0

    summary = {
        "total": stats["total"],
        "success": stats["success"],
        "error": stats["error"],
        "confidence_buckets": dict(conf_buckets),
        "label_counts": dict(label_counts),
        "low_confidence_count": len(low_conf),
        "soft_override_candidates": stats.get("soft_override_candidates", 0),
        "sample_size": len(files),
        "filename": {
            "label_present_count": filename_label_present,
            "label_present_rate": (filename_label_present / max(1, len(ok_rows))),
            "extracted_name_present_count": filename_extracted_name_present,
            "extracted_name_present_rate": (
                filename_extracted_name_present / max(1, len(ok_rows))
            ),
            "match_type_counts": dict(filename_match_type_counts),
            "confidence": {
                "mean_all": round(_mean(filename_conf_all), 6),
                "median_all": round(_median(filename_conf_all), 6),
                "mean_nonzero": round(_mean(filename_conf_nonzero), 6),
                "median_nonzero": round(_median(filename_conf_nonzero), 6),
            },
        },
        "hybrid": {
            "label_present_count": hybrid_label_present,
            "label_present_rate": (hybrid_label_present / max(1, len(ok_rows))),
            "source_counts": dict(hybrid_source_counts),
            "confidence": {
                "mean_all": round(_mean(hybrid_conf_all), 6),
                "median_all": round(_median(hybrid_conf_all), 6),
                "mean_nonzero": round(_mean(hybrid_conf_nonzero), 6),
                "median_nonzero": round(_median(hybrid_conf_nonzero), 6),
            },
        },
        "graph2d": {
            "label_present_count": graph2d_label_present,
            "label_present_rate": (graph2d_label_present / max(1, len(ok_rows))),
            "status_counts": dict(graph2d_status_counts),
            "confidence": {
                "mean_all": round(_mean(graph2d_conf_all), 6),
                "median_all": round(_median(graph2d_conf_all), 6),
                "mean_nonzero": round(_mean(graph2d_conf_nonzero), 6),
                "median_nonzero": round(_median(graph2d_conf_nonzero), 6),
            },
        },
        "titleblock": {
            "enabled": os.getenv("TITLEBLOCK_ENABLED", "false").lower() == "true",
            "total_ok": len(ok_rows),
            "texts_present_count": titleblock_texts_present,
            "texts_present_rate": (
                titleblock_texts_present / max(1, len(ok_rows))
            ),
            "any_signal_count": titleblock_any_signal_present,
            "any_signal_rate": (
                titleblock_any_signal_present / max(1, len(ok_rows))
            ),
            "part_name_present_count": titleblock_part_name_present,
            "part_name_present_rate": (
                titleblock_part_name_present / max(1, len(ok_rows))
            ),
            "label_present_count": titleblock_label_present,
            "label_present_rate": (
                titleblock_label_present / max(1, len(ok_rows))
            ),
            "status_counts": dict(titleblock_status_counts),
            "confidence": {
                "mean_all": round(_mean(titleblock_conf_all), 6),
                "median_all": round(_median(titleblock_conf_all), 6),
                "mean_nonzero": round(_mean(titleblock_conf_nonzero), 6),
                "median_nonzero": round(_median(titleblock_conf_nonzero), 6),
            },
        },
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
