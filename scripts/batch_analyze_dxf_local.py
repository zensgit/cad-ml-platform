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
from typing import Any, Dict, List, Optional, Tuple

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


def _canonicalize_label(label: Optional[str], matcher: Dict[str, str]) -> str:
    """Canonicalize a label using the filename-synonyms matcher (best-effort)."""
    if not label:
        return ""
    cleaned = str(label).strip()
    if not cleaned:
        return ""
    return matcher.get(cleaned.lower(), cleaned)


def _build_fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
    """Build stable CSV fieldnames covering all rows (avoids first-row schema issues)."""
    keys: set[str] = set()
    for row in rows:
        keys.update(str(k) for k in row.keys())
    return sorted(keys)


def _score_against_true(
    pred_label: Optional[str],
    true_label: Optional[str],
    matcher: Dict[str, str],
) -> Tuple[bool, str, str]:
    """Return (correct, pred_canon, true_canon) for label comparisons."""
    pred_canon = _canonicalize_label(pred_label, matcher)
    true_canon = _canonicalize_label(true_label, matcher)
    return bool(pred_canon and true_canon and pred_canon == true_canon), pred_canon, true_canon


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch analyze DXF via local TestClient")
    parser.add_argument("--dxf-dir", required=True, help="DXF directory")
    parser.add_argument("--manifest", help="Optional manifest CSV to select files")
    parser.add_argument("--output-dir", default="reports/experiments/20260122/batch_analysis")
    parser.add_argument("--max-files", type=int, default=200)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument(
        "--weak-label-min-confidence",
        type=float,
        default=0.8,
        help=(
            "Minimum confidence for accepting a weak 'true label' derived from the original "
            "filename via FilenameClassifier (default: 0.8)."
        ),
    )
    parser.add_argument(
        "--mask-filename",
        action="store_true",
        help=(
            "Upload DXFs with a masked/anonymous file name (e.g. file_0001.dxf) "
            "to evaluate behavior when filenames do not carry semantic labels."
        ),
    )
    parser.add_argument(
        "--strip-text",
        action="store_true",
        help=(
            "Strip TEXT/MTEXT/DIMENSION/ATTRIB entities from the uploaded DXF bytes "
            "to simulate geometry-only inference (also strips block definitions to "
            "avoid leaking titleblock text via INSERT virtual entities)."
        ),
    )
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help=(
            "Convenience mode: implies --mask-filename and --strip-text, and disables "
            "Hybrid text branches (TITLEBLOCK/PROCESS/FILENAME) via env vars so the "
            "run focuses on Graph2D behavior."
        ),
    )
    args = parser.parse_args()

    dxf_dir = Path(args.dxf_dir)
    if not dxf_dir.exists():
        raise SystemExit(f"DXF dir not found: {dxf_dir}")

    if bool(args.geometry_only):
        args.mask_filename = True
        args.strip_text = True
        # Force-disable HybridClassifier text branches for this evaluation mode.
        os.environ["TITLEBLOCK_ENABLED"] = "false"
        os.environ["PROCESS_FEATURES_ENABLED"] = "false"
        os.environ["FILENAME_CLASSIFIER_ENABLED"] = "false"

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

    # Weak-supervised "true labels" from the original file name (not the upload name).
    weak_label_classifier = None
    weak_label_matcher: Dict[str, str] = {}
    weak_label_min_conf = float(args.weak_label_min_confidence)
    try:
        from src.ml.filename_classifier import FilenameClassifier

        weak_label_classifier = FilenameClassifier()
        weak_label_matcher = dict(getattr(weak_label_classifier, "matcher", {}) or {})
    except Exception:
        weak_label_classifier = None
        weak_label_matcher = {}

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
                "fine_label_distribution.csv",
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
    fine_label_counts = Counter()
    fine_label_conf = defaultdict(list)

    weak_true_status_counts: Counter[str] = Counter()
    weak_true_match_type_counts: Counter[str] = Counter()
    weak_true_label_counts: Counter[str] = Counter()
    weak_true_covered = 0

    accuracy_counters: Dict[str, Counter[str]] = {
        "final_part_type": Counter(),
        "fine_part_type": Counter(),
        "hybrid_label": Counter(),
        "graph2d_label": Counter(),
        "titleblock_label": Counter(),
        "hybrid_filename_label": Counter(),
    }
    confusion_final: Counter[Tuple[str, str]] = Counter()

    for idx, item in enumerate(files):
        stats["total"] += 1
        try:
            payload = item.read_bytes()
        except Exception as exc:
            rows.append({"file": str(item), "status": "read_error", "error": str(exc)})
            stats["error"] += 1
            continue

        if bool(args.strip_text):
            try:
                from src.utils.dxf_io import strip_dxf_text_entities_from_bytes

                payload = strip_dxf_text_entities_from_bytes(payload, strip_blocks=True)
            except Exception as exc:
                rows.append(
                    {
                        "file": str(item),
                        "status": "strip_text_error",
                        "error": str(exc),
                    }
                )
                stats["error"] += 1
                continue

        weak_true_payload: Dict[str, Any] = {}
        weak_true_label: Optional[str] = None
        weak_true_conf = 0.0
        weak_true_status = "disabled"
        weak_true_match_type = "none"
        weak_true_extracted_name: Optional[str] = None
        weak_true_accepted = False
        if weak_label_classifier is not None:
            try:
                weak_true_payload = weak_label_classifier.predict(item.name)
            except Exception:
                weak_true_payload = {}

            weak_true_label = weak_true_payload.get("label")
            weak_true_extracted_name = weak_true_payload.get("extracted_name")
            weak_true_status = str(weak_true_payload.get("status") or "unknown")
            weak_true_match_type = str(weak_true_payload.get("match_type") or "none")
            try:
                weak_true_conf = float(weak_true_payload.get("confidence") or 0.0)
            except (TypeError, ValueError):
                weak_true_conf = 0.0
            if weak_true_status == "matched" and weak_true_label and weak_true_conf >= weak_label_min_conf:
                weak_true_accepted = True

            weak_true_status_counts[weak_true_status] += 1
            weak_true_match_type_counts[weak_true_match_type] += 1
            if weak_true_accepted:
                weak_true_covered += 1
                weak_true_label_counts[_canonicalize_label(weak_true_label, weak_label_matcher)] += 1

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
        fine_part_type = classification.get("fine_part_type")
        fine_confidence = float(classification.get("fine_confidence") or 0.0)
        fine_source = classification.get("fine_source")
        fine_rule_version = classification.get("fine_rule_version")

        # Weak-label scoring (best-effort).
        if weak_true_accepted and weak_label_matcher:
            true_label = _canonicalize_label(weak_true_label, weak_label_matcher)
            pred_final = _canonicalize_label(part_type, weak_label_matcher)
            if pred_final:
                accuracy_counters["final_part_type"]["evaluated"] += 1
                if pred_final == true_label:
                    accuracy_counters["final_part_type"]["correct"] += 1
                else:
                    confusion_final[(true_label, pred_final)] += 1
            else:
                accuracy_counters["final_part_type"]["missing_pred"] += 1

            pred_fine = _canonicalize_label(fine_part_type, weak_label_matcher)
            if pred_fine:
                accuracy_counters["fine_part_type"]["evaluated"] += 1
                if pred_fine == true_label:
                    accuracy_counters["fine_part_type"]["correct"] += 1
            else:
                accuracy_counters["fine_part_type"]["missing_pred"] += 1

            hybrid_label = (hybrid_decision.get("label") or "").strip()
            pred_hybrid = _canonicalize_label(hybrid_label, weak_label_matcher)
            if pred_hybrid:
                accuracy_counters["hybrid_label"]["evaluated"] += 1
                if pred_hybrid == true_label:
                    accuracy_counters["hybrid_label"]["correct"] += 1
            else:
                accuracy_counters["hybrid_label"]["missing_pred"] += 1

            g_label = (graph2d.get("label") or "").strip()
            pred_graph2d = _canonicalize_label(g_label, weak_label_matcher)
            if pred_graph2d:
                accuracy_counters["graph2d_label"]["evaluated"] += 1
                if pred_graph2d == true_label:
                    accuracy_counters["graph2d_label"]["correct"] += 1
            else:
                accuracy_counters["graph2d_label"]["missing_pred"] += 1

            t_label = (titleblock_pred.get("label") or "").strip()
            pred_title = _canonicalize_label(t_label, weak_label_matcher)
            if pred_title:
                accuracy_counters["titleblock_label"]["evaluated"] += 1
                if pred_title == true_label:
                    accuracy_counters["titleblock_label"]["correct"] += 1
            else:
                accuracy_counters["titleblock_label"]["missing_pred"] += 1

            h_fname_label = (filename_pred.get("label") or "").strip()
            pred_fname = _canonicalize_label(h_fname_label, weak_label_matcher)
            if pred_fname:
                accuracy_counters["hybrid_filename_label"]["evaluated"] += 1
                if pred_fname == true_label:
                    accuracy_counters["hybrid_filename_label"]["correct"] += 1
            else:
                accuracy_counters["hybrid_filename_label"]["missing_pred"] += 1

        label_counts[part_type] += 1
        label_conf[part_type].append(confidence)
        fine_label_counts[fine_part_type] += 1
        fine_label_conf[fine_part_type].append(fine_confidence)
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
            "upload_name": upload_name,
            "status": "ok",
            "part_type": part_type,
            "confidence": f"{confidence:.3f}",
            "confidence_source": classification.get("confidence_source"),
            "rule_version": classification.get("rule_version"),
            "fine_part_type": fine_part_type,
            "fine_confidence": f"{fine_confidence:.3f}",
            "fine_source": fine_source,
            "fine_rule_version": fine_rule_version,
            "weak_true_label": _canonicalize_label(weak_true_label, weak_label_matcher) if weak_true_accepted else "",
            "weak_true_confidence": f"{weak_true_conf:.3f}",
            "weak_true_status": weak_true_status,
            "weak_true_match_type": weak_true_match_type,
            "weak_true_extracted_name": weak_true_extracted_name or "",
            "weak_true_accepted": bool(weak_true_accepted),
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

    fieldnames = _build_fieldnames(rows)
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    sanitized_rows = _sanitize_file_column(rows)
    with results_sanitized_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(low_conf)
        sanitized_low_conf = _sanitize_file_column(low_conf)
        with low_conf_sanitized_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
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

    fine_label_dist_path = out_dir / "fine_label_distribution.csv"
    with fine_label_dist_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "count", "share", "avg_confidence"],
        )
        writer.writeheader()
        for label, count in fine_label_counts.most_common():
            avg_conf = sum(fine_label_conf[label]) / max(1, len(fine_label_conf[label]))
            writer.writerow(
                {
                    "label": label,
                    "count": count,
                    "share": f"{count / max(1, stats['success']):.3f}",
                    "avg_confidence": f"{avg_conf:.3f}",
                }
            )

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

    fine_source_counts: Counter[str] = Counter()
    fine_label_present = 0
    fine_conf_all: List[float] = []
    fine_conf_nonzero: List[float] = []

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

        # --- Fine label (API additive field; typically HybridClassifier output) ---
        fine_source = (row.get("fine_source") or "").strip() or "unknown"
        fine_source_counts[fine_source] += 1
        fine_label = (row.get("fine_part_type") or "").strip()
        if fine_label:
            fine_label_present += 1
        try:
            fine_conf = float(row.get("fine_confidence") or 0.0)
        except (TypeError, ValueError):
            fine_conf = 0.0
        fine_conf_all.append(fine_conf)
        if fine_conf > 0:
            fine_conf_nonzero.append(fine_conf)

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

    def _env_bool(key: str, default: bool) -> bool:
        raw = os.getenv(key)
        if raw is None:
            return bool(default)
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    titleblock_enabled_effective = (
        os.getenv("TITLEBLOCK_ENABLED", "false").lower() == "true"
    )
    titleblock_override_effective = (
        os.getenv("TITLEBLOCK_OVERRIDE_ENABLED", "false").lower() == "true"
    )
    titleblock_min_conf_effective: float | None = None
    try:
        titleblock_min_conf_effective = (
            float(os.getenv("TITLEBLOCK_MIN_CONF", "").strip())
            if os.getenv("TITLEBLOCK_MIN_CONF") is not None
            else None
        )
    except Exception:
        titleblock_min_conf_effective = None

    # Prefer effective HybridClassifier config when available; env always wins.
    try:
        from src.ml.hybrid_config import get_config

        cfg = get_config()
        titleblock_enabled_effective = _env_bool(
            "TITLEBLOCK_ENABLED", bool(cfg.titleblock.enabled)
        )
        titleblock_override_effective = _env_bool(
            "TITLEBLOCK_OVERRIDE_ENABLED", bool(cfg.titleblock.override_enabled)
        )
        if titleblock_min_conf_effective is None:
            titleblock_min_conf_effective = float(cfg.titleblock.min_confidence)
    except Exception:
        # Best-effort only; keep env-derived values.
        if titleblock_min_conf_effective is None:
            titleblock_min_conf_effective = 0.0

    summary = {
        "mask_filename": bool(args.mask_filename),
        "strip_text": bool(args.strip_text),
        "geometry_only": bool(args.geometry_only),
        "total": stats["total"],
        "success": stats["success"],
        "error": stats["error"],
        "confidence_buckets": dict(conf_buckets),
        "label_counts": dict(label_counts),
        "fine_label_counts": dict(fine_label_counts),
        "low_confidence_count": len(low_conf),
        "soft_override_candidates": stats.get("soft_override_candidates", 0),
        "sample_size": len(files),
        "weak_labels": {
            "enabled": weak_label_classifier is not None,
            "source": "filename",
            "min_confidence": weak_label_min_conf,
            "covered_count": weak_true_covered,
            "covered_rate": (weak_true_covered / max(1, len(ok_rows))),
            "status_counts": dict(weak_true_status_counts),
            "match_type_counts": dict(weak_true_match_type_counts),
            "label_counts": dict(weak_true_label_counts),
            "accuracy": {
                key: {
                    "evaluated": int(counter.get("evaluated", 0)),
                    "correct": int(counter.get("correct", 0)),
                    "missing_pred": int(counter.get("missing_pred", 0)),
                    "accuracy": (
                        float(counter.get("correct", 0))
                        / float(max(1, counter.get("evaluated", 0)))
                    ),
                }
                for key, counter in accuracy_counters.items()
            },
            "top_confusions_final": [
                {"true": true, "pred": pred, "count": int(count)}
                for (true, pred), count in confusion_final.most_common(20)
            ],
        },
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
        "fine": {
            "label_present_count": fine_label_present,
            "label_present_rate": (fine_label_present / max(1, len(ok_rows))),
            "source_counts": dict(fine_source_counts),
            "confidence": {
                "mean_all": round(_mean(fine_conf_all), 6),
                "median_all": round(_median(fine_conf_all), 6),
                "mean_nonzero": round(_mean(fine_conf_nonzero), 6),
                "median_nonzero": round(_median(fine_conf_nonzero), 6),
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
            "enabled": titleblock_enabled_effective,
            "override_enabled": titleblock_override_effective,
            "min_confidence_effective": (
                round(float(titleblock_min_conf_effective or 0.0), 6)
                if titleblock_min_conf_effective is not None
                else None
            ),
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
    print(f"fine_label_dist: {fine_label_dist_path}")


if __name__ == "__main__":
    main()
