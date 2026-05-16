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


STEP_FORMATS = {".step", ".stp"}
IGES_FORMATS = {".iges", ".igs"}
DEFAULT_PATTERNS = [
    "*.step",
    "*.stp",
    "*.STEP",
    "*.STP",
    "*.iges",
    "*.igs",
    "*.IGES",
    "*.IGS",
]


def _json_cell(value: Any) -> str:
    if not value:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _case_format(case: StepCase) -> str:
    suffix = case.file_path.suffix.lower()
    if suffix in STEP_FORMATS:
        return "step"
    if suffix in IGES_FORMATS:
        return "iges"
    return suffix.lstrip(".") or "unknown"


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _is_synthetic_payload(payload: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    for key in ("synthetic_geometry", "demo_geometry", "is_synthetic", "mock_geometry"):
        if _to_bool(payload.get(key)):
            return True
    source = str(
        payload.get("source")
        or payload.get("geometry_source")
        or payload.get("data_source")
        or ""
    ).lower()
    if any(token in source for token in ("synthetic", "demo", "mock")):
        return True
    metadata = payload.get("graph_metadata") or payload.get("metadata")
    if isinstance(metadata, dict) and metadata:
        return _is_synthetic_payload(metadata)
    return False


def _parse_json_cell(value: Any) -> Dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _load_manifest_cases(manifest_path: Path) -> List[StepCase]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("B-Rep manifest root must be a JSON object")
    root = Path(str(manifest.get("root") or ".")).expanduser()
    if not root.is_absolute():
        root = (manifest_path.parent / root).resolve()
    cases = manifest.get("cases") or []
    if not isinstance(cases, list):
        raise ValueError("B-Rep manifest `cases` must be a list")

    loaded: List[StepCase] = []
    for index, item in enumerate(cases):
        if not isinstance(item, dict):
            raise ValueError(f"B-Rep manifest case[{index}] must be an object")
        raw_path = str(item.get("path") or "").strip()
        if not raw_path:
            raise ValueError(f"B-Rep manifest case[{index}] is missing `path`")
        file_path = Path(raw_path).expanduser()
        if not file_path.is_absolute():
            file_path = root / file_path
        loaded.append(
            StepCase(
                file_path=file_path,
                relative_path=raw_path,
            )
        )
    return loaded


def _build_ok_row(
    case: StepCase,
    features: Dict[str, Any],
    graph: Dict[str, Any],
    *,
    evaluation_mode: str = "standard",
    demo_geometry_allowed: bool = False,
    extraction_latency_ms: float = 0.0,
) -> Dict[str, Any]:
    from src.ml.vision_3d import prepare_brep_features_for_report

    brep_summary = prepare_brep_features_for_report(features)
    bbox = features.get("bbox") or {}
    graph_valid_flag = graph.get("valid_3d")
    graph_valid = graph_valid_flag is not False and int(graph.get("node_count") or 0) > 0
    synthetic_geometry = _is_synthetic_payload(features) or _is_synthetic_payload(graph)

    return {
        "file_name": case.file_path.name,
        "relative_path": case.relative_path,
        "file_format": _case_format(case),
        "evaluation_mode": evaluation_mode,
        "status": "ok",
        "failure_reason": "",
        "parse_success": True,
        "shape_loaded": True,
        "brep_valid_3d": bool(features.get("valid_3d")),
        "graph_valid": graph_valid,
        "synthetic_geometry": synthetic_geometry,
        "demo_geometry_allowed": demo_geometry_allowed,
        "extraction_latency_ms": round(float(extraction_latency_ms), 4),
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


def _build_error_row(
    case: StepCase,
    status: str,
    error: str = "",
    *,
    failure_reason: str = "",
    parse_success: bool = False,
    evaluation_mode: str = "standard",
    demo_geometry_allowed: bool = False,
    extraction_latency_ms: float = 0.0,
) -> Dict[str, Any]:
    return {
        "file_name": case.file_path.name,
        "relative_path": case.relative_path,
        "file_format": _case_format(case),
        "evaluation_mode": evaluation_mode,
        "status": status,
        "failure_reason": failure_reason or status,
        "parse_success": parse_success,
        "shape_loaded": False,
        "brep_valid_3d": False,
        "graph_valid": False,
        "synthetic_geometry": False,
        "demo_geometry_allowed": demo_geometry_allowed,
        "extraction_latency_ms": round(float(extraction_latency_ms), 4),
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


def _load_shape(case: StepCase, engine: Any) -> tuple[Any, str]:
    content = case.file_path.read_bytes()
    file_format = _case_format(case)
    if file_format == "step":
        shape = engine.load_step(content, case.file_path.name)
        return shape, "" if shape is not None else "step_parse_failed"
    if file_format == "iges":
        load_iges = getattr(engine, "load_iges", None)
        if not callable(load_iges):
            return None, "iges_loader_missing"
        shape = load_iges(content, case.file_path.name)
        return shape, "" if shape is not None else "iges_parse_failed"
    return None, "unsupported_file_format"


def _strict_failure_reason(row: Dict[str, Any]) -> str:
    if bool(row.get("synthetic_geometry")) and not bool(row.get("demo_geometry_allowed")):
        return "synthetic_geometry_not_allowed"
    if not bool(row.get("brep_valid_3d")):
        return "brep_features_invalid"
    if int(row.get("faces") or 0) <= 0:
        return "brep_faces_missing"
    if not bool(row.get("graph_valid")):
        return "brep_graph_invalid"
    return ""


def _strict_status_from_reason(reason: str) -> str:
    if reason == "synthetic_geometry_not_allowed":
        return "demo_geometry_rejected"
    if reason == "brep_graph_invalid":
        return "graph_invalid"
    return "invalid_brep"


def _evaluate_cases(
    cases: List[StepCase],
    *,
    has_occ: Optional[bool] = None,
    engine: Optional[Any] = None,
    strict: bool = False,
    demo_geometry_allowed: bool = False,
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
        started = time.perf_counter()
        try:
            shape, load_failure_reason = _load_shape(case, engine)
            elapsed_ms = (time.perf_counter() - started) * 1000
            if shape is None:
                status = (
                    "unsupported_format"
                    if load_failure_reason.endswith("_missing")
                    else "load_failed"
                )
                if load_failure_reason == "unsupported_file_format":
                    status = "unsupported_format"
                rows.append(
                    _build_error_row(
                        case,
                        status,
                        failure_reason=load_failure_reason,
                        evaluation_mode="strict" if strict else "standard",
                        demo_geometry_allowed=demo_geometry_allowed,
                        extraction_latency_ms=elapsed_ms,
                    )
                )
                continue
            features = engine.extract_brep_features(shape) or {}
            graph = engine.extract_brep_graph(shape) or {}
            elapsed_ms = (time.perf_counter() - started) * 1000
            row = _build_ok_row(
                case,
                features,
                graph,
                evaluation_mode="strict" if strict else "standard",
                demo_geometry_allowed=demo_geometry_allowed,
                extraction_latency_ms=elapsed_ms,
            )
            if strict:
                failure_reason = _strict_failure_reason(row)
                if failure_reason:
                    row["status"] = _strict_status_from_reason(failure_reason)
                    row["failure_reason"] = failure_reason
            rows.append(row)
        except Exception as exc:  # pragma: no cover - defensive
            elapsed_ms = (time.perf_counter() - started) * 1000
            rows.append(
                _build_error_row(
                    case,
                    "error",
                    str(exc),
                    failure_reason="exception",
                    evaluation_mode="strict" if strict else "standard",
                    demo_geometry_allowed=demo_geometry_allowed,
                    extraction_latency_ms=elapsed_ms,
                )
            )
    return rows


def _summarize_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    row_list = list(rows)
    status_counts: Counter[str] = Counter()
    primary_surface_counts: Counter[str] = Counter()
    top_hint_counts: Counter[str] = Counter()
    schema_counts: Counter[str] = Counter()
    failure_reason_counts: Counter[str] = Counter()
    surface_type_histogram: Counter[str] = Counter()
    shape_loaded_count = 0
    parse_success_count = 0
    valid_count = 0
    graph_valid_count = 0
    hint_coverage_count = 0
    assembly_count = 0
    face_count_total = 0
    edge_count_total = 0
    solid_count_total = 0
    extraction_latencies: List[float] = []
    ok_rows: List[Dict[str, Any]] = []
    for row in row_list:
        status_counts[str(row.get("status") or "")] += 1
        failure_reason = str(row.get("failure_reason") or "").strip()
        if failure_reason:
            failure_reason_counts[failure_reason] += 1
        if bool(row.get("shape_loaded")):
            shape_loaded_count += 1
        if bool(row.get("parse_success")):
            parse_success_count += 1
        if bool(row.get("brep_valid_3d")):
            valid_count += 1
        if bool(row.get("graph_valid")):
            graph_valid_count += 1
        if bool(row.get("is_assembly")):
            assembly_count += 1
        face_count_total += int(row.get("faces") or 0)
        edge_count_total += int(row.get("edges") or 0)
        solid_count_total += int(row.get("solids") or 0)
        extraction_latencies.append(float(row.get("extraction_latency_ms") or 0.0))
        for surface_type, count in _parse_json_cell(row.get("surface_types")).items():
            surface_type_histogram[str(surface_type)] += int(count or 0)
        primary_surface = str(row.get("primary_surface_type") or "").strip()
        if primary_surface:
            primary_surface_counts[primary_surface] += 1
        top_hint = str(row.get("top_hint_label") or "").strip()
        if top_hint:
            top_hint_counts[top_hint] += 1
            hint_coverage_count += 1
        schema = str(row.get("graph_schema_version") or "").strip()
        if schema:
            schema_counts[schema] += 1
        if str(row.get("status") or "") == "ok":
            ok_rows.append(row)

    def _avg_int(key: str) -> float:
        if not ok_rows:
            return 0.0
        total = sum(int(item.get(key) or 0) for item in ok_rows)
        return round(total / len(ok_rows), 4)

    def _avg_latency() -> float:
        if not extraction_latencies:
            return 0.0
        return round(sum(extraction_latencies) / len(extraction_latencies), 4)

    return {
        "sample_size": len(row_list),
        "status_counts": dict(status_counts),
        "failure_reason_counts": dict(failure_reason_counts),
        "parse_success_count": parse_success_count,
        "shape_loaded_count": shape_loaded_count,
        "valid_3d_count": valid_count,
        "graph_valid_count": graph_valid_count,
        "hint_coverage_count": hint_coverage_count,
        "assembly_count": assembly_count,
        "face_count_total": face_count_total,
        "edge_count_total": edge_count_total,
        "solid_count_total": solid_count_total,
        "avg_faces_ok": _avg_int("faces"),
        "avg_nodes_ok": _avg_int("node_count"),
        "avg_edges_ok": _avg_int("edge_count"),
        "avg_extraction_latency_ms": _avg_latency(),
        "max_extraction_latency_ms": round(max(extraction_latencies), 4)
        if extraction_latencies
        else 0.0,
        "surface_type_histogram": dict(surface_type_histogram),
        "primary_surface_type_counts": dict(primary_surface_counts),
        "top_hint_label_counts": dict(top_hint_counts),
        "graph_schema_version_counts": dict(schema_counts),
    }


def _build_graph_qa_report(rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> Dict[str, Any]:
    sample_size = int(summary.get("sample_size") or 0)
    graph_valid_count = int(summary.get("graph_valid_count") or 0)
    parse_success_count = int(summary.get("parse_success_count") or 0)
    invalid_rows = [
        {
            "relative_path": row.get("relative_path"),
            "status": row.get("status"),
            "failure_reason": row.get("failure_reason"),
            "parse_success": row.get("parse_success"),
            "brep_valid_3d": row.get("brep_valid_3d"),
            "graph_valid": row.get("graph_valid"),
            "faces": row.get("faces"),
            "node_count": row.get("node_count"),
            "edge_count": row.get("edge_count"),
        }
        for row in rows
        if row.get("status") != "ok" or not row.get("graph_valid")
    ]
    if sample_size <= 0:
        status = "empty"
    elif invalid_rows:
        status = "failed"
    else:
        status = "passed"
    return {
        "status": status,
        "sample_size": sample_size,
        "parse_success_count": parse_success_count,
        "graph_valid_count": graph_valid_count,
        "graph_valid_ratio": round(graph_valid_count / sample_size, 6)
        if sample_size
        else 0.0,
        "invalid_graph_rows": invalid_rows,
        "failure_reason_counts": dict(summary.get("failure_reason_counts") or {}),
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
    parser.add_argument("--step-dir", default="", help="Directory containing STEP/IGES files.")
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional B-Rep golden manifest JSON. Overrides directory discovery.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        default=None,
        help="Glob pattern relative to step-dir. Can be provided multiple times.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max file count.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail rows with invalid B-Rep, invalid graph, or demo geometry.",
    )
    parser.add_argument(
        "--allow-demo-geometry",
        action="store_true",
        help="Allow rows explicitly marked as demo/synthetic geometry in strict mode.",
    )
    parser.add_argument(
        "--output-dir",
        default=f"reports/experiments/{time.strftime('%Y%m%d')}/brep_step_dir_eval",
        help="Directory for generated CSV/JSON reports.",
    )
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else None
    patterns = args.patterns or DEFAULT_PATTERNS
    step_dir: Optional[Path] = None
    if manifest_path is not None:
        if not manifest_path.exists():
            raise SystemExit(f"B-Rep golden manifest not found: {manifest_path}")
        cases = _load_manifest_cases(manifest_path)
    else:
        if not args.step_dir:
            raise SystemExit("Either --step-dir or --manifest is required")
        step_dir = Path(args.step_dir).expanduser().resolve()
        if not step_dir.exists():
            raise SystemExit(f"STEP directory not found: {step_dir}")
        cases = _load_step_cases(step_dir, patterns)
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _evaluate_cases(
        cases,
        strict=args.strict,
        demo_geometry_allowed=args.allow_demo_geometry,
    )
    summary = _summarize_rows(rows)
    summary.update(
        {
            "step_dir": str(step_dir) if step_dir else "",
            "manifest": str(manifest_path) if manifest_path else "",
            "patterns": patterns,
            "strict_mode": bool(args.strict),
            "demo_geometry_allowed": bool(args.allow_demo_geometry),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    graph_qa = _build_graph_qa_report(rows, summary)

    _write_csv(output_dir / "results.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "graph_qa.json").write_text(
        json.dumps(graph_qa, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
