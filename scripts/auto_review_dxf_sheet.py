#!/usr/bin/env python3
"""Auto-review DXF sheet using Graph2D predictions and synonym rules."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.ml.vision_2d import Graph2DClassifier


def _load_synonyms(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return {str(k): [str(v) for v in vals] for k, vals in data.items() if isinstance(vals, list)}


def _build_alias_map(synonyms: Dict[str, List[str]]) -> Dict[str, str]:
    alias: Dict[str, str] = {}
    for label, values in synonyms.items():
        alias[label] = label
        for value in values:
            alias[value.strip().lower()] = label
    return alias


def _canonical(label: str, alias_map: Dict[str, str]) -> str:
    cleaned = str(label or "").strip()
    if not cleaned:
        return ""
    return alias_map.get(cleaned.lower(), cleaned)


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-review DXF sheet.")
    parser.add_argument("--input", required=True, help="Review sheet CSV")
    parser.add_argument("--output", required=True, help="Output reviewed CSV")
    parser.add_argument("--conflicts", required=True, help="Output conflicts CSV")
    parser.add_argument(
        "--synonyms-json",
        default="data/knowledge/label_synonyms_template.json",
        help="Synonyms JSON path",
    )
    parser.add_argument(
        "--graph2d-model",
        default="models/graph2d_merged_latest.pth",
        help="Graph2D checkpoint path",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.05,
        help="Minimum Graph2D confidence to auto-approve",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    conflicts_path = Path(args.conflicts)

    synonyms = _load_synonyms(Path(args.synonyms_json))
    alias_map = _build_alias_map(synonyms)

    classifier = Graph2DClassifier(model_path=args.graph2d_model)

    with input_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    extra_fields = [
        "graph2d_label",
        "graph2d_confidence",
        "auto_review_verdict",
        "auto_review_reason",
    ]
    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    conflicts_path.parent.mkdir(parents=True, exist_ok=True)

    conflicts: List[Dict[str, str]] = []

    for row in rows:
        source_path = (row.get("source_path") or "").strip()
        suggested_label = (row.get("suggested_label_cn") or "").strip()
        review_notes = (row.get("review_notes") or "").strip()
        review_status = (row.get("review_status") or "").strip().lower()
        reviewer_label = (row.get("reviewer_label_cn") or "").strip()
        manual_locked = (
            "manual_priority_decision" in review_notes
            or (review_status == "confirmed" and bool(reviewer_label))
        )
        suggested_canon = _canonical(suggested_label, alias_map)

        graph2d_label = ""
        graph2d_confidence = 0.0
        if source_path:
            try:
                with open(source_path, "rb") as handle:
                    data = handle.read()
                result = classifier.predict_from_bytes(data, Path(source_path).name)
                graph2d_label = str(result.get("label") or "")
                graph2d_confidence = float(result.get("confidence") or 0.0)
            except Exception:
                graph2d_label = ""
                graph2d_confidence = 0.0

        graph2d_canon = _canonical(graph2d_label, alias_map)

        verdict = "needs_followup"
        reason = ""

        if manual_locked:
            verdict = "manual_confirmed"
            reason = "manual_override"
            row["graph2d_label"] = graph2d_label
            row["graph2d_confidence"] = (
                f"{graph2d_confidence:.4f}" if graph2d_label else ""
            )
            row["auto_review_verdict"] = verdict
            row["auto_review_reason"] = reason
            if review_status != "confirmed":
                row["review_status"] = "confirmed"
            continue

        if suggested_canon and graph2d_canon:
            if suggested_canon == graph2d_canon and graph2d_confidence >= args.confidence_threshold:
                verdict = "confirmed"
                reason = "suggested_matches_graph2d"
            else:
                reason = "label_conflict"
        elif suggested_canon and not graph2d_canon:
            verdict = "needs_followup"
            reason = "graph2d_missing"
        elif graph2d_canon and graph2d_confidence >= args.confidence_threshold:
            verdict = "suggested_missing_graph2d_present"
            reason = "suggested_missing"
        else:
            reason = "insufficient_signal"

        row["graph2d_label"] = graph2d_label
        row["graph2d_confidence"] = f"{graph2d_confidence:.4f}" if graph2d_label else ""
        row["auto_review_verdict"] = verdict
        row["auto_review_reason"] = reason

        if verdict == "confirmed":
            row["review_status"] = "confirmed"
            existing_notes = (row.get("review_notes") or "").strip()
            marker = "auto_review:graph2d_match"
            if marker not in existing_notes:
                row["review_notes"] = f"{existing_notes}; {marker}" if existing_notes else marker

        if verdict != "confirmed":
            conflicts.append(row)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with conflicts_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(conflicts)

    print(f"Wrote {output_path} ({len(rows)} rows)")
    print(f"Wrote {conflicts_path} ({len(conflicts)} conflicts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
