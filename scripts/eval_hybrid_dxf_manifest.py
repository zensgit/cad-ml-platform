#!/usr/bin/env python3
"""Evaluate DXF API / hybrid classification against a labeled manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ensure_local_cache() -> None:
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
    os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "1")
    os.environ.setdefault("GRAPH2D_ENABLED", "true")
    os.environ.setdefault("GRAPH2D_FUSION_ENABLED", "true")
    os.environ.setdefault("FUSION_ANALYZER_ENABLED", "true")
    os.environ.setdefault("HYBRID_CLASSIFIER_ENABLED", "true")


def _load_alias_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    alias_map: Dict[str, str] = {}
    for label, values in payload.items():
        canonical = str(label or "").strip()
        if not canonical:
            continue
        alias_map[canonical.lower()] = canonical
        if isinstance(values, list):
            for value in values:
                cleaned = str(value or "").strip()
                if cleaned:
                    alias_map[cleaned.lower()] = canonical
    return alias_map


def _canonicalize_label(label: Optional[str], alias_map: Dict[str, str]) -> str:
    cleaned = str(label or "").strip()
    if not cleaned:
        return ""
    return alias_map.get(cleaned.lower(), cleaned)


def _exact_eval_label(label: Optional[str], alias_map: Dict[str, str]) -> str:
    return _canonicalize_label(label, alias_map)


def _coarse_eval_label(label: Optional[str], alias_map: Dict[str, str]) -> str:
    from src.core.classification import normalize_coarse_label

    canonical = _canonicalize_label(label, alias_map)
    if not canonical:
        return ""
    return str(normalize_coarse_label(canonical) or "")


@dataclass
class EvalCase:
    file_path: Path
    file_name: str
    true_label: str
    source_dir: str = ""
    relative_path: str = ""


def _load_manifest_cases(manifest_path: Path, dxf_dir: Path) -> List[EvalCase]:
    cases: List[EvalCase] = []
    seen: set[Path] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            file_name = str(row.get("file_name") or row.get("file") or "").strip()
            true_label = str(row.get("label_cn") or "").strip()
            if not file_name or not true_label:
                continue
            relative_path = str(row.get("relative_path") or "").strip()
            source_dir = str(row.get("source_dir") or "").strip()

            candidates: List[Path] = []
            if relative_path:
                candidates.append(dxf_dir / relative_path)
            candidates.append(dxf_dir / file_name)
            if source_dir:
                candidates.append(dxf_dir / source_dir / file_name)

            resolved = None
            for candidate in candidates:
                if candidate.exists():
                    resolved = candidate
                    break
            if resolved is None or resolved in seen:
                continue
            seen.add(resolved)
            cases.append(
                EvalCase(
                    file_path=resolved,
                    file_name=file_name,
                    true_label=true_label,
                    source_dir=source_dir,
                    relative_path=relative_path,
                )
            )
    return cases


def _score_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    branch_to_column: Dict[str, str],
    alias_map: Dict[str, str],
    normalizer: Callable[[Optional[str], Dict[str, str]], str],
) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for branch, column in branch_to_column.items():
        evaluated = 0
        correct = 0
        missing = 0
        confusion: Counter[Tuple[str, str]] = Counter()
        for row in rows:
            true_label = normalizer(row.get("true_label"), alias_map)
            pred_label = normalizer(row.get(column), alias_map)
            if not true_label:
                continue
            if not pred_label:
                missing += 1
                continue
            evaluated += 1
            if pred_label == true_label:
                correct += 1
            else:
                confusion[(true_label, pred_label)] += 1
        summary[branch] = {
            "evaluated": evaluated,
            "correct": correct,
            "missing_pred": missing,
            "accuracy": (correct / evaluated) if evaluated else 0.0,
            "top_confusions": [
                {"true": true, "pred": pred, "count": int(count)}
                for (true, pred), count in confusion.most_common(10)
            ],
        }
    return summary


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_cell(value: Any) -> str:
    if not value:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _collect_prep_fields(results_payload: Dict[str, Any]) -> Dict[str, Any]:
    from src.ml.vision_3d import prepare_brep_features_for_report

    classification = results_payload.get("classification", {}) or {}
    history_pred = classification.get("history_prediction", {}) or {}
    history_input = classification.get("history_sequence_input", {}) or {}
    raw_brep_hints = (
        classification.get("brep_feature_hints")
        or results_payload.get("brep_feature_hints")
        or {}
    )
    brep_summary = prepare_brep_features_for_report(
        results_payload.get("features_3d", {}) or {},
        brep_feature_hints=raw_brep_hints if isinstance(raw_brep_hints, dict) else None,
    )

    return {
        "history_label": history_pred.get("label"),
        "history_confidence": history_pred.get("confidence"),
        "history_status": history_pred.get("status"),
        "history_source": history_pred.get("source"),
        "history_shadow_only": history_pred.get("shadow_only"),
        "history_used_for_fusion": history_pred.get("used_for_fusion"),
        "history_input_resolved": history_input.get("resolved"),
        "history_input_source": history_input.get("source"),
        "brep_valid_3d": brep_summary.get("valid_3d"),
        "brep_faces": brep_summary.get("faces"),
        "brep_primary_surface_type": brep_summary.get("primary_surface_type"),
        "brep_primary_surface_ratio": brep_summary.get("primary_surface_ratio"),
        "brep_surface_types": _json_cell(brep_summary.get("surface_types") or {}),
        "brep_feature_hints": _json_cell(brep_summary.get("feature_hints") or {}),
        "brep_feature_hint_top_label": brep_summary.get("top_hint_label"),
        "brep_feature_hint_top_score": brep_summary.get("top_hint_score"),
        "brep_embedding_dim": brep_summary.get("embedding_dim"),
    }


def _build_ok_row(case: EvalCase, results_payload: Dict[str, Any]) -> Dict[str, Any]:
    classification = results_payload.get("classification", {}) or {}
    graph2d = classification.get("graph2d_prediction", {}) or {}
    filename_pred = classification.get("filename_prediction", {}) or {}
    titleblock_pred = classification.get("titleblock_prediction", {}) or {}
    hybrid_decision = classification.get("hybrid_decision", {}) or {}

    row = {
        "file_name": case.file_name,
        "relative_path": case.relative_path,
        "source_dir": case.source_dir,
        "true_label": case.true_label,
        "true_label_exact": case.true_label,
        "true_label_coarse": _coarse_eval_label(case.true_label, {}),
        "status": "ok",
        "part_type": classification.get("part_type"),
        "confidence": classification.get("confidence"),
        "coarse_part_type": classification.get("coarse_part_type"),
        "fine_part_type": classification.get("fine_part_type"),
        "fine_confidence": classification.get("fine_confidence"),
        "coarse_fine_part_type": classification.get("coarse_fine_part_type"),
        "graph2d_label": graph2d.get("label"),
        "graph2d_confidence": graph2d.get("confidence"),
        "coarse_graph2d_label": classification.get("coarse_graph2d_label"),
        "filename_label": filename_pred.get("label"),
        "filename_confidence": filename_pred.get("confidence"),
        "coarse_filename_label": classification.get("coarse_filename_label"),
        "titleblock_label": titleblock_pred.get("label"),
        "titleblock_confidence": titleblock_pred.get("confidence"),
        "coarse_titleblock_label": classification.get("coarse_titleblock_label"),
        "hybrid_label": hybrid_decision.get("label"),
        "hybrid_confidence": hybrid_decision.get("confidence"),
        "hybrid_source": hybrid_decision.get("source"),
        "coarse_hybrid_label": classification.get("coarse_hybrid_label"),
        "decision_path": json.dumps(
            hybrid_decision.get("decision_path") or [],
            ensure_ascii=False,
        ),
        "source_contributions": json.dumps(
            classification.get("source_contributions") or {}, ensure_ascii=False
        ),
        "hybrid_explanation_summary": (
            (classification.get("hybrid_explanation") or {}).get("summary")
        ),
    }
    row.update(_collect_prep_fields(results_payload))
    return row


def _summarize_prep_signals(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    history_status_counts: Counter[str] = Counter()
    brep_top_hint_counts: Counter[str] = Counter()
    history_prediction_count = 0
    history_input_resolved_count = 0
    history_used_for_fusion_true = 0
    history_used_for_fusion_false = 0
    history_shadow_only_true = 0
    brep_valid_3d_count = 0
    brep_feature_hints_count = 0

    for row in rows:
        if row.get("history_label") or row.get("history_status"):
            history_prediction_count += 1
        if row.get("history_input_resolved") is True:
            history_input_resolved_count += 1
        if row.get("history_used_for_fusion") is True:
            history_used_for_fusion_true += 1
        if row.get("history_used_for_fusion") is False:
            history_used_for_fusion_false += 1
        if row.get("history_shadow_only") is True:
            history_shadow_only_true += 1
        history_status = str(row.get("history_status") or "").strip()
        if history_status:
            history_status_counts[history_status] += 1

        if row.get("brep_valid_3d") is True:
            brep_valid_3d_count += 1
        top_hint_label = str(row.get("brep_feature_hint_top_label") or "").strip()
        if top_hint_label:
            brep_feature_hints_count += 1
            brep_top_hint_counts[top_hint_label] += 1

    return {
        "history_prediction_count": history_prediction_count,
        "history_input_resolved_count": history_input_resolved_count,
        "history_used_for_fusion_true": history_used_for_fusion_true,
        "history_used_for_fusion_false": history_used_for_fusion_false,
        "history_shadow_only_true": history_shadow_only_true,
        "history_status_counts": dict(history_status_counts),
        "brep_valid_3d_count": brep_valid_3d_count,
        "brep_feature_hints_count": brep_feature_hints_count,
        "brep_top_hint_counts": dict(brep_top_hint_counts.most_common(10)),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate hybrid DXF classification against a labeled manifest."
    )
    parser.add_argument("--dxf-dir", required=True, help="DXF directory root.")
    parser.add_argument("--manifest", required=True, help="Manifest CSV with label_cn.")
    parser.add_argument(
        "--output-dir",
        default=f"reports/experiments/{time.strftime('%Y%m%d')}/hybrid_dxf_manifest_eval",
        help="Directory for CSV/JSON outputs.",
    )
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--graph2d-model-path",
        default="",
        help="Optional Graph2D checkpoint path. Exported to GRAPH2D_MODEL_PATH.",
    )
    parser.add_argument("--mask-filename", action="store_true")
    parser.add_argument("--strip-text", action="store_true")
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="Mask filename, strip text, and disable text-heavy hybrid branches.",
    )
    parser.add_argument(
        "--synonyms-json",
        default="data/knowledge/label_synonyms_template.json",
        help="Synonyms file used for canonical/normalized scoring.",
    )
    args = parser.parse_args(argv)

    _ensure_local_cache()

    if args.geometry_only:
        args.mask_filename = True
        args.strip_text = True
        os.environ["TITLEBLOCK_ENABLED"] = "false"
        os.environ["PROCESS_FEATURES_ENABLED"] = "false"
        os.environ["FILENAME_CLASSIFIER_ENABLED"] = "false"

    if str(args.graph2d_model_path or "").strip():
        os.environ["GRAPH2D_MODEL_PATH"] = str(args.graph2d_model_path).strip()

    dxf_dir = Path(args.dxf_dir)
    manifest_path = Path(args.manifest)
    if not dxf_dir.exists():
        raise SystemExit(f"DXF dir not found: {dxf_dir}")
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    cases = _load_manifest_cases(manifest_path, dxf_dir)
    if not cases:
        raise SystemExit("No manifest cases resolved to existing DXF files")

    random.seed(int(args.seed))
    random.shuffle(cases)
    if int(args.max_files) > 0:
        cases = cases[: int(args.max_files)]

    from fastapi.testclient import TestClient

    from src.main import app
    from src.utils.dxf_io import strip_dxf_text_entities_from_bytes

    client = TestClient(app)
    options = {"extract_features": True, "classify_parts": True}
    alias_map = _load_alias_map(REPO_ROOT / str(args.synonyms_json))

    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    status_counts: Counter[str] = Counter()

    for idx, case in enumerate(cases):
        payload = case.file_path.read_bytes()
        if args.strip_text:
            payload = strip_dxf_text_entities_from_bytes(payload, strip_blocks=True)
        upload_name = (
            f"file_{idx+1:04d}{case.file_path.suffix.lower() or '.dxf'}"
            if args.mask_filename
            else case.file_name
        )
        response = client.post(
            "/api/v1/analyze/",
            files={"file": (upload_name, payload, "application/dxf")},
            data={"options": json.dumps(options)},
            headers={"x-api-key": os.getenv("API_KEY", "test")},
        )
        if response.status_code != 200:
            status_counts["error"] += 1
            rows.append(
                {
                    "file_name": case.file_name,
                    "true_label": case.true_label,
                    "status": "error",
                    "http_status": response.status_code,
                    "error": response.text,
                }
            )
            continue

        results_payload = response.json().get("results", {}) or {}
        status_counts["ok"] += 1
        rows.append(_build_ok_row(case, results_payload))

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    accuracy = _score_rows(
        ok_rows,
        branch_to_column={
            "final_part_type": "part_type",
            "graph2d_label": "graph2d_label",
            "filename_label": "filename_label",
            "titleblock_label": "titleblock_label",
            "hybrid_label": "hybrid_label",
            "fine_part_type": "fine_part_type",
        },
        alias_map=alias_map,
        normalizer=_coarse_eval_label,
    )
    exact_accuracy = _score_rows(
        ok_rows,
        branch_to_column={
            "final_part_type": "part_type",
            "graph2d_label": "graph2d_label",
            "filename_label": "filename_label",
            "titleblock_label": "titleblock_label",
            "hybrid_label": "hybrid_label",
            "fine_part_type": "fine_part_type",
        },
        alias_map=alias_map,
        normalizer=_exact_eval_label,
    )

    def _confidence_stats(rows_in: List[Dict[str, Any]], column: str) -> Dict[str, Any]:
        vals: List[float] = []
        for row in rows_in:
            try:
                vals.append(float(row.get(column) or 0.0))
            except (TypeError, ValueError):
                continue
        vals.sort()
        if not vals:
            return {"count": 0, "p50": 0.0, "p90": 0.0, "low_conf_rate": 0.0}
        p50 = vals[len(vals) // 2]
        p90 = vals[min(len(vals) - 1, int(len(vals) * 0.9))]
        low_conf = sum(1 for value in vals if value < 0.2)
        return {
            "count": len(vals),
            "p50": round(p50, 6),
            "p90": round(p90, 6),
            "low_conf_rate": round(low_conf / len(vals), 6),
        }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
    _write_csv(out_dir / "results.csv", rows)

    summary = {
        "manifest": str(manifest_path),
        "sample_size": len(cases),
        "status_counts": dict(status_counts),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "mode": {
            "mask_filename": bool(args.mask_filename),
            "strip_text": bool(args.strip_text),
            "geometry_only": bool(args.geometry_only),
            "graph2d_model_path": bool(str(args.graph2d_model_path or "").strip()),
        },
        "accuracy": accuracy,
        "exact_accuracy": exact_accuracy,
        "coarse_accuracy": accuracy,
        "confidence": {
            "final_part_type": _confidence_stats(ok_rows, "confidence"),
            "graph2d_label": _confidence_stats(ok_rows, "graph2d_confidence"),
            "filename_label": _confidence_stats(ok_rows, "filename_confidence"),
            "titleblock_label": _confidence_stats(ok_rows, "titleblock_confidence"),
            "hybrid_label": _confidence_stats(ok_rows, "hybrid_confidence"),
            "fine_part_type": _confidence_stats(ok_rows, "fine_confidence"),
        },
        "prep_signals": _summarize_prep_signals(ok_rows),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
