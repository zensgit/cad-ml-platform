#!/usr/bin/env python3
"""Export assistant/analyze explainability records into CSV and JSON summaries."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple


JSON_EXTENSIONS = {".json", ".jsonl"}
SUMMARY_FIELDS = (
    "evidence",
    "sources",
    "decision_path",
    "source_contributions",
    "explanation_summary",
)
EVIDENCE_FIELD_NAMES = (
    "evidence[].source",
    "evidence[].summary",
    "evidence[].match_type",
    "evidence[].reference_id",
)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _truncate(value: str, limit: int = 160) -> str:
    text = _clean_text(value)
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


def _safe_json_loads(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        return None


def _get_nested(payload: Any, path: Sequence[str]) -> Any:
    current = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _coerce_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    text = _clean_text(value)
    if not text:
        return {}
    parsed = _safe_json_loads(text)
    if isinstance(parsed, dict):
        return dict(parsed)
    return {}


def _coerce_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    text = _clean_text(value)
    if not text:
        return []
    parsed = _safe_json_loads(text)
    if isinstance(parsed, list):
        return list(parsed)
    if ";" in text:
        return [part.strip() for part in text.split(";") if part.strip()]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def _coerce_string_list(value: Any) -> List[str]:
    items = _coerce_list(value)
    results: List[str] = []
    for item in items:
        if isinstance(item, dict):
            token = _first_text(
                item,
                ("name",),
                ("source",),
                ("title",),
                ("summary",),
                ("label",),
            )
            if token:
                results.append(token)
            continue
        token = _clean_text(item)
        if token:
            results.append(token)
    return results


def _first_text(payload: Dict[str, Any], *paths: Sequence[str]) -> str:
    for path in paths:
        value = _get_nested(payload, path)
        text = _clean_text(value)
        if text:
            return text
    return ""


def _first_list(
    payload: Dict[str, Any],
    *paths: Sequence[str],
    string_values: bool = False,
) -> List[Any]:
    for path in paths:
        value = _get_nested(payload, path)
        items = _coerce_string_list(value) if string_values else _coerce_list(value)
        if items:
            return items
    return []


def _first_dict(payload: Dict[str, Any], *paths: Sequence[str]) -> Dict[str, Any]:
    for path in paths:
        value = _get_nested(payload, path)
        mapping = _coerce_dict(value)
        if mapping:
            return mapping
    return {}


def _detect_record_kind(
    payload: Dict[str, Any],
    *,
    evidence: List[Dict[str, Any]],
    answer: str,
    decision_path: List[str],
) -> str:
    if answer or evidence or _get_nested(payload, ("assistant",)) or _get_nested(
        payload, ("results", "assistant")
    ):
        return "assistant"
    if decision_path or _get_nested(payload, ("results", "classification")) or _get_nested(
        payload, ("score_breakdown",)
    ):
        return "analyze"
    return "generic"


def _evidence_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    values = _first_list(
        payload,
        ("evidence",),
        ("assistant", "evidence"),
        ("assistant_response", "evidence"),
        ("response", "evidence"),
        ("results", "assistant", "evidence"),
        ("score_breakdown", "evidence"),
    )
    return [dict(item) for item in values if isinstance(item, dict)]


def _sources(payload: Dict[str, Any]) -> List[str]:
    return _first_list(
        payload,
        ("sources",),
        ("assistant", "sources"),
        ("assistant_response", "sources"),
        ("response", "sources"),
        ("results", "assistant", "sources"),
        string_values=True,
    )


def _decision_path(payload: Dict[str, Any]) -> List[str]:
    return [
        token
        for token in _first_list(
            payload,
            ("decision_path",),
            ("classification", "decision_path"),
            ("results", "classification", "decision_path"),
            ("score_breakdown", "decision_path"),
            ("hybrid_path",),
            string_values=True,
        )
        if token
    ]


def _source_contributions(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _first_dict(
        payload,
        ("source_contributions",),
        ("classification", "source_contributions"),
        ("results", "classification", "source_contributions"),
        ("score_breakdown", "source_contributions"),
        ("hybrid_source_contributions",),
    )


def _explanation_summary(payload: Dict[str, Any]) -> str:
    return _first_text(
        payload,
        ("explanation", "summary"),
        ("assistant", "explanation", "summary"),
        ("assistant_response", "explanation", "summary"),
        ("hybrid_explanation", "summary"),
        ("classification", "hybrid_explanation", "summary"),
        ("results", "classification", "hybrid_explanation", "summary"),
        ("score_breakdown", "hybrid_explanation", "summary"),
        ("hybrid_explanation_summary",),
        ("summary",),
    )


def _query(payload: Dict[str, Any]) -> str:
    return _first_text(
        payload,
        ("query",),
        ("question",),
        ("prompt",),
        ("assistant", "query"),
        ("assistant_response", "query"),
        ("request", "query"),
        ("metadata", "query"),
    )


def _answer(payload: Dict[str, Any]) -> str:
    return _first_text(
        payload,
        ("answer",),
        ("assistant", "answer"),
        ("assistant_response", "answer"),
        ("response", "answer"),
        ("results", "assistant", "answer"),
        ("message",),
        ("output_text",),
    )


def _record_id(payload: Dict[str, Any], fallback: str) -> str:
    return _first_text(
        payload,
        ("analysis_id",),
        ("doc_id",),
        ("id",),
        ("request_id",),
        ("trace_id",),
        ("file",),
        ("filename",),
        ("metadata", "request_id"),
        ("metadata", "analysis_id"),
    ) or fallback


def _evidence_source(item: Dict[str, Any]) -> str:
    return _first_text(
        item,
        ("source",),
        ("module",),
        ("provider",),
        ("name",),
    )


def _evidence_summary(item: Dict[str, Any]) -> str:
    return _first_text(
        item,
        ("summary",),
        ("snippet",),
        ("text",),
        ("content",),
    )


def _evidence_type(item: Dict[str, Any]) -> str:
    return (
        _first_text(item, ("type",), ("evidence_type",), ("match_type",), ("kind",))
        or "unknown"
    )


def _evidence_reference(item: Dict[str, Any]) -> str:
    return _first_text(item, ("reference_id",), ("id",), ("ref",))


def _top_named_counts(counter: Counter[str], limit: int) -> List[Dict[str, Any]]:
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return [
        {"name": name, "count": int(count)}
        for name, count in ranked[: max(0, int(limit))]
    ]


def _coverage_summary(count: int, total: int) -> Dict[str, Any]:
    pct = 0.0
    if total > 0:
        pct = round(float(count) / float(total), 4)
    return {
        "count": int(count),
        "total": int(total),
        "missing_count": int(max(0, total - count)),
        "coverage_pct": pct,
    }


def _looks_like_record(payload: Dict[str, Any]) -> bool:
    if not payload:
        return False
    for key in (
        "answer",
        "evidence",
        "sources",
        "results",
        "classification",
        "score_breakdown",
        "analysis_id",
        "doc_id",
    ):
        if key in payload:
            return True
    return False


def _records_from_json_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("records", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
                return [dict(item) for item in value]
        return [payload] if _looks_like_record(payload) or payload else []
    return []


def _discover_input_files(paths: Sequence[str]) -> List[Path]:
    discovered: List[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        input_path = Path(raw_path).expanduser()
        if not input_path.exists():
            raise SystemExit(f"Input path not found: {input_path}")
        if input_path.is_dir():
            for candidate in sorted(path for path in input_path.rglob("*") if path.is_file()):
                if candidate.suffix.lower() not in JSON_EXTENSIONS:
                    continue
                resolved = candidate.resolve()
                if resolved not in seen:
                    discovered.append(candidate)
                    seen.add(resolved)
            continue
        if input_path.suffix.lower() not in JSON_EXTENSIONS:
            raise SystemExit(f"Unsupported input type: {input_path}")
        resolved = input_path.resolve()
        if resolved not in seen:
            discovered.append(input_path)
            seen.add(resolved)
    if not discovered:
        raise SystemExit("No JSON/JSONL inputs found")
    return discovered


def _iter_records(files: Sequence[Path]) -> Iterator[Tuple[Path, str, Dict[str, Any]]]:
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            for index, record in enumerate(_records_from_json_payload(payload), start=1):
                yield path, f"{path.name}#{index}", record
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                if not isinstance(payload, dict):
                    raise ValueError(f"JSONL record must be an object: {path} line {line_number}")
                yield path, f"{path.name}#L{line_number}", payload


def _normalize_record(path: Path, locator: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    evidence = _evidence_items(payload)
    sources = _sources(payload)
    decision_path = _decision_path(payload)
    source_contributions = _source_contributions(payload)
    explanation_summary = _explanation_summary(payload)
    query = _query(payload)
    answer = _answer(payload)

    evidence_type_counter: Counter[str] = Counter()
    evidence_source_tokens: List[str] = []
    evidence_missing_fields: Counter[str] = Counter()
    for item in evidence:
        evidence_type_counter[_evidence_type(item)] += 1
        source = _evidence_source(item)
        summary = _evidence_summary(item)
        match_type = _evidence_type(item)
        reference_id = _evidence_reference(item)
        if source:
            evidence_source_tokens.append(source)
        else:
            evidence_missing_fields["evidence[].source"] += 1
        if not summary:
            evidence_missing_fields["evidence[].summary"] += 1
        if match_type == "unknown":
            evidence_missing_fields["evidence[].match_type"] += 1
        if not reference_id:
            evidence_missing_fields["evidence[].reference_id"] += 1

    structured_sources = sorted(
        {
            token
            for token in evidence_source_tokens + [str(key).strip() for key in source_contributions.keys()]
            if token
        }
    )
    record_missing_fields: List[str] = []
    if not evidence:
        record_missing_fields.append("evidence")
    if not sources:
        record_missing_fields.append("sources")
    if not decision_path:
        record_missing_fields.append("decision_path")
    if not source_contributions:
        record_missing_fields.append("source_contributions")

    evidence_types = sorted(evidence_type_counter.keys())
    sources_preview = ";".join(_truncate(item, limit=60) for item in sources[:3])

    fallback_id = locator.replace("#", "-")
    record_kind = _detect_record_kind(
        payload,
        evidence=evidence,
        answer=answer,
        decision_path=decision_path,
    )
    if not explanation_summary and record_kind != "assistant":
        record_missing_fields.append("explanation_summary")

    return {
        "input_path": str(path),
        "record_locator": locator,
        "record_id": _record_id(payload, fallback=fallback_id),
        "record_kind": record_kind,
        "query": _truncate(query, limit=120),
        "answer_preview": _truncate(answer, limit=160),
        "evidence_count": int(len(evidence)),
        "evidence_types": ";".join(evidence_types),
        "evidence_sources": ";".join(sorted(set(evidence_source_tokens))),
        "evidence_missing_fields": ";".join(sorted(evidence_missing_fields.keys())),
        "sources_count": int(len(sources)),
        "sources_preview": sources_preview,
        "source_contributions_count": int(len(source_contributions)),
        "source_contribution_sources": ";".join(sorted(source_contributions.keys())),
        "decision_path_count": int(len(decision_path)),
        "decision_path": ";".join(decision_path),
        "explanation_summary": _truncate(explanation_summary, limit=160),
        "missing_fields": ";".join(sorted(record_missing_fields)),
        "_structured_sources": structured_sources,
        "_evidence_field_presence": {
            "evidence[].source": sum(1 for item in evidence if _evidence_source(item)),
            "evidence[].summary": sum(1 for item in evidence if _evidence_summary(item)),
            "evidence[].match_type": sum(
                1 for item in evidence if _evidence_type(item) != "unknown"
            ),
            "evidence[].reference_id": sum(1 for item in evidence if _evidence_reference(item)),
        },
        "_evidence_missing_counts": dict(evidence_missing_fields),
    }


def _build_summary(
    normalized_rows: Sequence[Dict[str, Any]],
    *,
    input_paths: Sequence[str],
    scanned_files: Sequence[Path],
    top_k: int,
) -> Dict[str, Any]:
    total_records = len(normalized_rows)
    total_evidence_items = sum(int(row["evidence_count"]) for row in normalized_rows)

    kind_counter: Counter[str] = Counter()
    evidence_type_counter: Counter[str] = Counter()
    structured_source_counter: Counter[str] = Counter()
    decision_step_counter: Counter[str] = Counter()
    missing_field_counter: Counter[str] = Counter()

    evidence_present = 0
    sources_present = 0
    source_contributions_present = 0
    any_source_signal_present = 0
    decision_path_present = 0
    explanation_present = 0

    evidence_field_presence: Counter[str] = Counter()

    for row in normalized_rows:
        kind_counter[str(row["record_kind"])] += 1

        evidence_count = int(row["evidence_count"])
        sources_count = int(row["sources_count"])
        source_contributions_count = int(row["source_contributions_count"])
        decision_path_count = int(row["decision_path_count"])

        if evidence_count > 0:
            evidence_present += 1
        if sources_count > 0:
            sources_present += 1
        if source_contributions_count > 0:
            source_contributions_present += 1
        if row["_structured_sources"] or sources_count > 0:
            any_source_signal_present += 1
        if decision_path_count > 0:
            decision_path_present += 1
        if _clean_text(row["explanation_summary"]):
            explanation_present += 1

        for token in _coerce_string_list(row["evidence_types"]):
            evidence_type_counter[token] += 1
        for token in row["_structured_sources"]:
            structured_source_counter[token] += 1
        for token in _coerce_string_list(row["decision_path"]):
            decision_step_counter[token] += 1
        for token in _coerce_string_list(row["missing_fields"]):
            missing_field_counter[token] += 1
        for field_name, count in row["_evidence_missing_counts"].items():
            missing_field_counter[str(field_name)] += int(count)
        for field_name in EVIDENCE_FIELD_NAMES:
            evidence_field_presence[field_name] += int(
                row["_evidence_field_presence"].get(field_name, 0)
            )

    average_evidence_count = 0.0
    average_sources_count = 0.0
    average_decision_path_count = 0.0
    if total_records > 0:
        average_evidence_count = round(
            sum(int(row["evidence_count"]) for row in normalized_rows) / total_records,
            3,
        )
        average_sources_count = round(
            sum(int(row["sources_count"]) for row in normalized_rows) / total_records,
            3,
        )
        average_decision_path_count = round(
            sum(int(row["decision_path_count"]) for row in normalized_rows) / total_records,
            3,
        )

    return {
        "input_paths": [str(Path(path).expanduser()) for path in input_paths],
        "scanned_files": [str(path) for path in scanned_files],
        "scanned_file_count": int(len(scanned_files)),
        "total_records": int(total_records),
        "total_evidence_items": int(total_evidence_items),
        "average_evidence_count": average_evidence_count,
        "average_sources_count": average_sources_count,
        "average_decision_path_count": average_decision_path_count,
        "coverage": {
            "records_with_evidence": _coverage_summary(evidence_present, total_records),
            "records_with_sources": _coverage_summary(sources_present, total_records),
            "records_with_source_contributions": _coverage_summary(
                source_contributions_present, total_records
            ),
            "records_with_any_source_signal": _coverage_summary(
                any_source_signal_present, total_records
            ),
            "records_with_decision_path": _coverage_summary(
                decision_path_present, total_records
            ),
            "records_with_explanation_summary": _coverage_summary(
                explanation_present, total_records
            ),
        },
        "evidence_item_coverage": {
            field_name: _coverage_summary(
                int(evidence_field_presence[field_name]),
                total_evidence_items,
            )
            for field_name in EVIDENCE_FIELD_NAMES
        },
        "top_record_kinds": _top_named_counts(kind_counter, limit=top_k),
        "top_evidence_types": _top_named_counts(evidence_type_counter, limit=top_k),
        "top_structured_sources": _top_named_counts(
            structured_source_counter, limit=top_k
        ),
        "top_decision_steps": _top_named_counts(decision_step_counter, limit=top_k),
        "top_missing_fields": _top_named_counts(missing_field_counter, limit=top_k),
        "expected_record_fields": list(SUMMARY_FIELDS),
    }


def _write_csv(output_path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "input_path",
        "record_locator",
        "record_id",
        "record_kind",
        "query",
        "answer_preview",
        "evidence_count",
        "evidence_types",
        "evidence_sources",
        "evidence_missing_fields",
        "sources_count",
        "sources_preview",
        "source_contributions_count",
        "source_contribution_sources",
        "decision_path_count",
        "decision_path",
        "explanation_summary",
        "missing_fields",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export assistant/analyze explainability JSON or JSONL into a flat CSV "
            "and a summary JSON."
        )
    )
    parser.add_argument(
        "--input-path",
        action="append",
        required=True,
        help="JSON, JSONL, or directory containing them. Repeatable.",
    )
    parser.add_argument("--output-csv", required=True, help="Flat CSV output path.")
    parser.add_argument(
        "--summary-json",
        default="",
        help="Optional summary JSON path. Defaults to <output-csv>.summary.json.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of entries to keep in top-* summary lists.",
    )
    args = parser.parse_args(argv)

    if int(args.top_k) < 1:
        raise SystemExit("--top-k must be >= 1")

    files = _discover_input_files(args.input_path)
    normalized_rows = [
        _normalize_record(path, locator, payload)
        for path, locator, payload in _iter_records(files)
    ]
    if not normalized_rows:
        raise SystemExit("No records found in inputs")

    output_csv = Path(args.output_csv)
    summary_path = (
        Path(args.summary_json)
        if _clean_text(args.summary_json)
        else output_csv.with_suffix(".summary.json")
    )
    _write_csv(output_csv, normalized_rows)

    summary = _build_summary(
        normalized_rows,
        input_paths=args.input_path,
        scanned_files=files,
        top_k=int(args.top_k),
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"Exported {len(normalized_rows)} records to {output_csv} "
        f"(summary: {summary_path})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
