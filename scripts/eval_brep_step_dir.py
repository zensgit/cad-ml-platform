#!/usr/bin/env python3
"""Evaluate STEP/B-Rep files in a directory and emit CSV/JSON reports."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class StepCase:
    file_path: Path
    relative_path: str


def _json_cell(value: Any) -> str:
    if not value:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _load_step_cases(step_dir: Path, patterns: Iterable[str]) -> List[StepCase]:
    seen: set[Path] = set()
    cases: List[StepCase] = []
    for pattern in patterns:
        for candidate in sorted(step_dir.rglob(pattern)):
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            cases.append(
                StepCase(
                    file_path=candidate,
                    relative_path=str(candidate.relative_to(step_dir)),
                )
            )
    return sorted(cases, key=lambda item: item.relative_path.lower())


def _build_ok_row(
    case: StepCase,
    features: Dict[str, Any],
    graph: Dict[str, Any],
) -> Dict[str, Any]:
    from src.ml.vision_3d import prepare_brep_features_for_report

    brep_summary = prepare_brep_features_for_report(features)
    bbox = features.get("bbox") or {}

    return {
        "file_name": case.file_path.name,
        "relative_path": case.relative_path,
        "status": "ok",
        "shape_loaded": True,
        "brep_valid_3d": bool(features.get("valid_3d")),
        "faces": int(features.get("faces") or 0),
        "edges": int(features.get("edges") or 0),
        "vertices": int(features.get("vertices") or 0),
        "solids": int(features.get("solids") or 0),
        "shells": int(features.get("shells") or 0),
        "volume": float(features.get("volume") or 0.0),
        "surface_area": float(features.get("surface_area") or 0.0),
        "bbox_diag": float(bbox.get("diag") or 0.0),
        "is_assembly": bool(features.get("is_assembly")),
        "surface_types": _json_cell(features.get("surface_types") or {}),
        "primary_surface_type": brep_summary.get("primary_surface_type"),
        "primary_surface_ratio": brep_summary.get("primary_surface_ratio"),
        "graph_schema_version": graph.get("graph_schema_version"),
        "node_count": int(graph.get("node_count") or 0),
        "edge_count": int(graph.get("edge_count") or 0),
        "feature_hints": _json_cell(brep_summary.get("feature_hints") or {}),
        "top_hint_label": brep_summary.get("top_hint_label"),
        "top_hint_score": float(brep_summary.get("top_hint_score") or 0.0),
        "graph_metadata": _json_cell(graph.get("graph_metadata") or {}),
        "error": "",
    }


def _build_error_row(case: StepCase, status: str, error: str = "") -> Dict[str, Any]:
    return {
        "file_name": case.file_path.name,
        "relative_path": case.relative_path,
        "status": status,
        "shape_loaded": False,
        "brep_valid_3d": False,
        "faces": 0,
        "edges": 0,
        "vertices": 0,
        "solids": 0,
        "shells": 0,
        "volume": 0.0,
        "surface_area": 0.0,
        "bbox_diag": 0.0,
        "is_assembly": False,
        "surface_types": "",
        "primary_surface_type": "",
        "primary_surface_ratio": 0.0,
        "graph_schema_version": "",
        "node_count": 0,
        "edge_count": 0,
        "feature_hints": "",
        "top_hint_label": "",
        "top_hint_score": 0.0,
        "graph_metadata": "",
        "error": error,
    }


def _evaluate_cases(
    cases: List[StepCase],
    *,
    has_occ: Optional[bool] = None,
    engine: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    if has_occ is None or engine is None:
        from src.core.geometry.engine import HAS_OCC, get_geometry_engine

        if has_occ is None:
            has_occ = HAS_OCC
        if engine is None:
            engine = get_geometry_engine()

    if not has_occ:
        raise RuntimeError("pythonocc-core is not available in the current environment")
    rows: List[Dict[str, Any]] = []
    for case in cases:
        try:
            shape = engine.load_step(case.file_path.read_bytes(), case.file_path.name)
            if shape is None:
                rows.append(_build_error_row(case, "load_failed"))
                continue
            features = engine.extract_brep_features(shape)
            graph = engine.extract_brep_graph(shape)
            rows.append(_build_ok_row(case, features, graph))
        except Exception as exc:  # pragma: no cover - defensive
            rows.append(_build_error_row(case, "error", str(exc)))
    return rows


def _summarize_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    row_list = list(rows)
    status_counts: Counter[str] = Counter()
    primary_surface_counts: Counter[str] = Counter()
    top_hint_counts: Counter[str] = Counter()
    schema_counts: Counter[str] = Counter()
    valid_count = 0
    for row in row_list:
        status_counts[str(row.get("status") or "")] += 1
        if bool(row.get("brep_valid_3d")):
            valid_count += 1
        primary_surface = str(row.get("primary_surface_type") or "").strip()
        if primary_surface:
            primary_surface_counts[primary_surface] += 1
        top_hint = str(row.get("top_hint_label") or "").strip()
        if top_hint:
            top_hint_counts[top_hint] += 1
        schema = str(row.get("graph_schema_version") or "").strip()
        if schema:
            schema_counts[schema] += 1
    return {
        "sample_size": len(row_list),
        "status_counts": dict(status_counts),
        "valid_3d_count": valid_count,
        "primary_surface_type_counts": dict(primary_surface_counts),
        "top_hint_label_counts": dict(top_hint_counts),
        "graph_schema_version_counts": dict(schema_counts),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate STEP/B-Rep files in a directory and emit CSV/JSON reports."
    )
    parser.add_argument("--step-dir", required=True, help="Directory containing STEP files.")
    parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        default=None,
        help="Glob pattern relative to step-dir. Can be provided multiple times.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max file count.")
    parser.add_argument(
        "--output-dir",
        default=f"reports/experiments/{time.strftime('%Y%m%d')}/brep_step_dir_eval",
        help="Directory for generated CSV/JSON reports.",
    )
    args = parser.parse_args(argv)

    step_dir = Path(args.step_dir).expanduser().resolve()
    if not step_dir.exists():
        raise SystemExit(f"STEP directory not found: {step_dir}")

    patterns = args.patterns or ["*.step", "*.stp", "*.STEP", "*.STP"]
    cases = _load_step_cases(step_dir, patterns)
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _evaluate_cases(cases)
    summary = _summarize_rows(rows)
    summary.update(
        {
            "step_dir": str(step_dir),
            "patterns": patterns,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )

    _write_csv(output_dir / "results.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
