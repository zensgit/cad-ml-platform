#!/usr/bin/env python3
"""Export a batch OCR/drawing review pack as CSV and JSON."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

LIST_FIELDS = (
    "recognized_keys",
    "missing_keys",
    "critical_fields",
    "present_critical_fields",
    "missing_critical_fields",
    "review_reasons",
    "recommended_actions",
    "symbol_types",
    "gdt_symbol_types",
    "materials_detected",
    "standards_candidates",
)
TITLE_BLOCK_FIELDS = (
    "drawing_number",
    "part_name",
    "revision",
    "material",
    "scale",
    "sheet",
    "date",
    "weight",
    "company",
    "projection",
)
CSV_FIELDNAMES = [
    "record_id",
    "source_path",
    "source_name",
    "surface",
    "document_ref",
    "provider",
    "success",
    "confidence",
    "drawing_number",
    "part_name",
    "revision",
    "material",
    "scale",
    "sheet",
    "date",
    "weight",
    "company",
    "projection",
    "review_candidate",
    "review_priority",
    "review_priority_score",
    "review_recommended",
    "primary_gap",
    "review_reasons",
    "recommended_actions",
    "automation_ready",
    "readiness_score",
    "readiness_band",
    "recognized_count",
    "total_fields",
    "coverage_ratio",
    "recognized_keys",
    "missing_keys",
    "critical_fields",
    "present_critical_fields",
    "missing_critical_fields",
    "identifiers_count",
    "has_identifiers",
    "has_dimensions",
    "has_symbols",
    "has_process_requirements",
    "has_standards_candidates",
    "dimension_count",
    "symbol_count",
    "symbol_types",
    "gdt_symbol_types",
    "has_surface_finish",
    "has_gdt",
    "process_heat_treatments",
    "process_surface_treatments",
    "process_welding",
    "process_general_notes",
    "materials_detected",
    "standards_candidates",
]
RECORD_LIST_FIELDS = {
    "review_reasons",
    "recommended_actions",
    "recognized_keys",
    "missing_keys",
    "critical_fields",
    "present_critical_fields",
    "missing_critical_fields",
    "symbol_types",
    "gdt_symbol_types",
    "materials_detected",
    "standards_candidates",
}
RECORD_FLOAT_FIELDS = {
    "confidence",
    "review_priority_score",
    "readiness_score",
    "coverage_ratio",
}
RECORD_INT_FIELDS = {
    "recognized_count",
    "total_fields",
    "identifiers_count",
    "dimension_count",
    "symbol_count",
    "process_heat_treatments",
    "process_surface_treatments",
    "process_welding",
    "process_general_notes",
}
RECORD_BOOL_FIELDS = {
    "success",
    "review_candidate",
    "review_recommended",
    "automation_ready",
    "has_identifiers",
    "has_dimensions",
    "has_symbols",
    "has_process_requirements",
    "has_standards_candidates",
    "has_surface_finish",
    "has_gdt",
}
PRIORITY_RANK = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return _clean_text(value).lower() in {"1", "true", "yes", "y", "on"}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = _clean_text(value)
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [token.strip() for token in text.split(";") if token.strip()]
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [token.strip() for token in text.split(";") if token.strip()]


def _dict_value(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    text = _clean_text(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _ranked_counts(counter: Counter[str]) -> List[Dict[str, Any]]:
    return [{"name": name, "count": int(count)} for name, count in counter.most_common()]


def _load_json_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                if isinstance(payload, dict):
                    records.append(payload)
        return records

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("records", "items", "responses", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [payload]
    return []


def _discover_input_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if not input_path.is_dir():
        raise FileNotFoundError(f"Unsupported input path: {input_path}")

    files = sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
    )
    if not files:
        raise FileNotFoundError(f"No .json or .jsonl files found under: {input_path}")
    return files


def _detect_surface(record: Dict[str, Any]) -> str:
    explicit = _clean_text(record.get("surface") or record.get("endpoint"))
    if explicit:
        return explicit
    if "fields" in record or "field_confidence" in record:
        return "drawing"
    return "ocr"


def _document_ref(record: Dict[str, Any], title_block: Dict[str, Any], index: int) -> str:
    for key in ("document_ref", "document_id", "file_name", "filename", "file", "source_path"):
        value = _clean_text(record.get(key))
        if value:
            return value
    for key in ("drawing_number", "part_name"):
        value = _clean_text(title_block.get(key))
        if value:
            return value
    return f"record-{index:04d}"


def _review_priority_score(
    *,
    review_priority: str,
    readiness_score: float,
    missing_critical_fields: Sequence[str],
    review_reasons: Sequence[str],
    coverage_ratio: float,
) -> float:
    score = float(PRIORITY_RANK.get(review_priority, 0) * 100)
    score += max(0.0, 1.0 - readiness_score) * 10.0
    score += len(missing_critical_fields) * 2.0
    score += len(review_reasons) * 0.5
    score += max(0.0, 1.0 - coverage_ratio) * 2.0
    return round(score, 4)


def _flatten_record_for_csv(record: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for key in CSV_FIELDNAMES:
        value = record.get(key)
        if key in RECORD_LIST_FIELDS:
            row[key] = ";".join(_string_list(value))
        else:
            row[key] = value
    return row


def _normalize_record(
    record: Dict[str, Any],
    *,
    source_path: Path,
    input_index: int,
) -> Optional[Dict[str, Any]]:
    review_hints = _dict_value(record.get("review_hints"))
    field_coverage = _dict_value(record.get("field_coverage"))
    engineering_signals = _dict_value(record.get("engineering_signals"))
    if not review_hints or not field_coverage or not engineering_signals:
        return None

    title_block = _dict_value(record.get("title_block"))
    missing_critical_fields = _string_list(review_hints.get("missing_critical_fields"))
    review_reasons = _string_list(review_hints.get("review_reasons"))
    recommended_actions = _string_list(review_hints.get("recommended_actions"))
    priority = _clean_text(review_hints.get("review_priority")).lower() or "low"
    coverage_ratio = _to_float(field_coverage.get("coverage_ratio"))
    readiness_score = _to_float(review_hints.get("readiness_score"))
    identifiers = record.get("identifiers")
    identifiers_count = (
        len(identifiers)
        if isinstance(identifiers, list)
        else _to_int(record.get("identifiers_count"))
    )
    process_counts = _dict_value(engineering_signals.get("process_requirement_counts"))

    normalized: Dict[str, Any] = {
        "record_id": _clean_text(record.get("record_id"))
        or f"{source_path.name}:{input_index}",
        "source_path": str(source_path),
        "source_name": source_path.name,
        "surface": _detect_surface(record),
        "document_ref": _document_ref(record, title_block, input_index),
        "provider": _clean_text(record.get("provider")),
        "success": _to_bool(record.get("success", True)),
        "confidence": _to_float(record.get("confidence"), default=0.0),
        "review_candidate": _to_bool(review_hints.get("review_recommended")),
        "review_priority": priority,
        "review_recommended": _to_bool(review_hints.get("review_recommended")),
        "primary_gap": _clean_text(review_hints.get("primary_gap")) or "ready",
        "review_reasons": review_reasons,
        "recommended_actions": recommended_actions,
        "automation_ready": _to_bool(review_hints.get("automation_ready")),
        "readiness_score": readiness_score,
        "readiness_band": _clean_text(review_hints.get("readiness_band")).lower() or "unknown",
        "recognized_count": _to_int(field_coverage.get("recognized_count")),
        "total_fields": _to_int(field_coverage.get("total_fields")),
        "coverage_ratio": coverage_ratio,
        "recognized_keys": _string_list(field_coverage.get("recognized_keys")),
        "missing_keys": _string_list(field_coverage.get("missing_keys")),
        "critical_fields": _string_list(review_hints.get("critical_fields")),
        "present_critical_fields": _string_list(review_hints.get("present_critical_fields")),
        "missing_critical_fields": missing_critical_fields,
        "identifiers_count": identifiers_count,
        "has_identifiers": _to_bool(review_hints.get("has_identifiers")),
        "has_dimensions": _to_bool(review_hints.get("has_dimensions")),
        "has_symbols": _to_bool(review_hints.get("has_symbols")),
        "has_process_requirements": _to_bool(review_hints.get("has_process_requirements")),
        "has_standards_candidates": _to_bool(review_hints.get("has_standards_candidates")),
        "dimension_count": _to_int(engineering_signals.get("dimension_count")),
        "symbol_count": _to_int(engineering_signals.get("symbol_count")),
        "symbol_types": _string_list(engineering_signals.get("symbol_types")),
        "gdt_symbol_types": _string_list(engineering_signals.get("gdt_symbol_types")),
        "has_surface_finish": _to_bool(engineering_signals.get("has_surface_finish")),
        "has_gdt": _to_bool(engineering_signals.get("has_gdt")),
        "process_heat_treatments": _to_int(process_counts.get("heat_treatments")),
        "process_surface_treatments": _to_int(process_counts.get("surface_treatments")),
        "process_welding": _to_int(process_counts.get("welding")),
        "process_general_notes": _to_int(process_counts.get("general_notes")),
        "materials_detected": _string_list(engineering_signals.get("materials_detected")),
        "standards_candidates": _string_list(engineering_signals.get("standards_candidates")),
        "title_block": title_block,
        "field_coverage": field_coverage,
        "engineering_signals": engineering_signals,
        "review_hints": review_hints,
    }
    for key in TITLE_BLOCK_FIELDS:
        normalized[key] = _clean_text(title_block.get(key))

    normalized["review_priority_score"] = _review_priority_score(
        review_priority=priority,
        readiness_score=readiness_score,
        missing_critical_fields=missing_critical_fields,
        review_reasons=review_reasons,
        coverage_ratio=coverage_ratio,
    )
    return normalized


def _load_normalized_records(input_path: Path) -> Tuple[List[Dict[str, Any]], List[Path], int]:
    files = _discover_input_files(input_path)
    normalized: List[Dict[str, Any]] = []
    total_input_records = 0
    record_index = 0

    for path in files:
        for record in _load_json_records(path):
            total_input_records += 1
            item = _normalize_record(record, source_path=path, input_index=record_index)
            record_index += 1
            if item is not None:
                normalized.append(item)
    return normalized, files, total_input_records


def _sort_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: (
            -_to_float(record.get("review_priority_score")),
            _to_float(record.get("readiness_score"), default=1.0),
            _clean_text(record.get("document_ref")),
            _clean_text(record.get("record_id")),
        ),
    )


def _select_records(
    records: List[Dict[str, Any]],
    *,
    include_ready: bool,
    top_k: int,
) -> List[Dict[str, Any]]:
    candidates = [
        record
        for record in records
        if include_ready or _to_bool(record.get("review_candidate"))
    ]
    candidates = _sort_records(candidates)
    if top_k > 0:
        candidates = candidates[:top_k]
    return candidates


def _average(records: List[Dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return round(
        sum(_to_float(record.get(key)) for record in records) / len(records),
        4,
    )


def _top_from_list(records: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(_string_list(record.get(key)))
    return _ranked_counts(counter)


def _top_from_value(records: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    counter: Counter[str] = Counter()
    for record in records:
        value = _clean_text(record.get(key))
        if value:
            counter[value] += 1
    return _ranked_counts(counter)


def _build_summary(
    *,
    input_path: Path,
    input_files: List[Path],
    total_input_records: int,
    normalized_records: List[Dict[str, Any]],
    exported_records: List[Dict[str, Any]],
    include_ready: bool,
    top_k: int,
) -> Dict[str, Any]:
    summary = {
        "input_path": str(input_path),
        "input_files": [str(path) for path in input_files],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_input_records": int(total_input_records),
        "normalized_records": int(len(normalized_records)),
        "skipped_records": int(total_input_records - len(normalized_records)),
        "exported_records": int(len(exported_records)),
        "include_ready": bool(include_ready),
        "top_k": int(top_k),
        "review_candidate_count": sum(
            1 for record in normalized_records if _to_bool(record.get("review_candidate"))
        ),
        "automation_ready_count": sum(
            1 for record in exported_records if _to_bool(record.get("automation_ready"))
        ),
        "average_readiness_score": _average(exported_records, "readiness_score"),
        "average_coverage_ratio": _average(exported_records, "coverage_ratio"),
        "surface_counts": _top_from_value(exported_records, "surface"),
        "review_priority_counts": _top_from_value(exported_records, "review_priority"),
        "readiness_band_counts": _top_from_value(exported_records, "readiness_band"),
        "primary_gap_counts": _top_from_value(exported_records, "primary_gap"),
        "top_review_reasons": _top_from_list(exported_records, "review_reasons"),
        "top_recommended_actions": _top_from_list(exported_records, "recommended_actions"),
        "top_missing_critical_fields": _top_from_list(
            exported_records, "missing_critical_fields"
        ),
        "top_materials_detected": _top_from_list(exported_records, "materials_detected"),
        "top_standards_candidates": _top_from_list(
            exported_records, "standards_candidates"
        ),
        "sample_records": [
            {
                "record_id": _clean_text(record.get("record_id")),
                "document_ref": _clean_text(record.get("document_ref")),
                "surface": _clean_text(record.get("surface")),
                "review_priority": _clean_text(record.get("review_priority")),
                "primary_gap": _clean_text(record.get("primary_gap")),
                "readiness_band": _clean_text(record.get("readiness_band")),
                "readiness_score": _to_float(record.get("readiness_score")),
                "coverage_ratio": _to_float(record.get("coverage_ratio")),
                "missing_critical_fields": _string_list(
                    record.get("missing_critical_fields")
                ),
                "recommended_actions": _string_list(record.get("recommended_actions")),
            }
            for record in exported_records[:3]
        ],
    }
    return summary


def _write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(_flatten_record_for_csv(record))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a batch OCR/drawing review pack from JSON/JSONL responses using "
            "review_hints, field_coverage, and engineering_signals."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON/JSONL file or directory containing OCR/drawing responses.",
    )
    parser.add_argument("--output-csv", required=True, help="Flattened CSV review pack path.")
    parser.add_argument(
        "--output-json",
        default="",
        help="Structured JSON review pack path (default: <output-csv>.json).",
    )
    parser.add_argument(
        "--include-ready",
        action="store_true",
        help="Include automation-ready rows instead of exporting only review candidates.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Keep only the top K records after ranking (0 means no limit).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    input_path = Path(args.input)
    output_csv = Path(args.output_csv)
    output_json = (
        Path(args.output_json)
        if _clean_text(args.output_json)
        else output_csv.with_suffix(".json")
    )

    normalized_records, input_files, total_input_records = _load_normalized_records(input_path)
    exported_records = _select_records(
        normalized_records,
        include_ready=bool(args.include_ready),
        top_k=int(args.top_k),
    )
    summary = _build_summary(
        input_path=input_path,
        input_files=input_files,
        total_input_records=total_input_records,
        normalized_records=normalized_records,
        exported_records=exported_records,
        include_ready=bool(args.include_ready),
        top_k=int(args.top_k),
    )
    payload = {
        "summary": summary,
        "records": exported_records,
    }

    _write_csv(output_csv, exported_records)
    _write_json(output_json, payload)
    print(
        f"Exported {len(exported_records)} OCR review records to {output_csv} "
        f"and {output_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
