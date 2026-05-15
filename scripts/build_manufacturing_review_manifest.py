#!/usr/bin/env python3
"""Build and validate manufacturing evidence review manifests."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_hybrid_dxf_manifest import (  # noqa: E402
    MANUFACTURING_PAYLOAD_DETAIL_PREFIX,
    MANUFACTURING_PAYLOAD_FIELDS,
    _clean_expected_payload_value,
    _extract_expected_manufacturing_payloads,
    _extract_expected_manufacturing_sources,
    _flatten_expected_detail_fields,
    _normalize_manufacturing_source_token,
    _parse_manufacturing_source_tokens,
)

REVIEW_MANIFEST_COLUMNS = [
    "file_name",
    "label_cn",
    "relative_path",
    "source_dir",
    "reviewed_manufacturing_evidence_sources",
    "reviewed_manufacturing_evidence_payload_json",
    "review_status",
    "reviewer",
    "reviewed_at",
    "review_notes",
    "suggested_manufacturing_evidence_sources",
    "suggested_manufacturing_evidence_payload_json",
    "actual_manufacturing_evidence",
]

REVIEW_GAP_COLUMNS = [
    "row_id",
    "file_name",
    "label_cn",
    "relative_path",
    "source_dir",
    "review_status",
    "reviewer",
    "reviewed_at",
    "gap_reasons",
    "gap_reason_count",
    "source_ready",
    "payload_ready",
    "detail_ready",
    "suggested_manufacturing_evidence_sources",
    "suggested_manufacturing_evidence_payload_json",
    "reviewed_manufacturing_evidence_sources",
    "reviewed_manufacturing_evidence_payload_json",
    "review_notes",
]

REVIEW_CONTEXT_COLUMNS = [
    "row_id",
    "file_name",
    "label_cn",
    "relative_path",
    "source_dir",
    "review_status",
    "reviewer",
    "reviewed_at",
    "gap_reasons",
    "gap_reason_count",
    "source_ready",
    "payload_ready",
    "detail_ready",
    "suggested_manufacturing_evidence_sources",
    "suggested_source_count",
    "suggested_payload_field_count",
    "suggested_detail_field_count",
    "suggested_payload_fields",
    "suggested_manufacturing_evidence_payload_json",
    "actual_evidence_item_count",
    "actual_evidence_sources",
    "actual_evidence_summary",
    "actual_evidence_detail_keys",
    "actual_manufacturing_evidence",
    "reviewed_manufacturing_evidence_sources",
    "reviewed_manufacturing_evidence_payload_json",
    "review_notes",
]

REVIEW_BATCH_COLUMNS = [
    "review_batch",
    "batch_rank",
    "label_gap_row_count",
    "row_id",
    "file_name",
    "label_cn",
    "relative_path",
    "source_dir",
    "review_status",
    "gap_reasons",
    "gap_reason_count",
    "source_gap",
    "payload_gap",
    "detail_gap",
    "approval_gap",
    "metadata_gap",
    "suggested_manufacturing_evidence_sources",
    "actual_evidence_sources",
    "reviewer",
    "reviewed_at",
    "review_notes",
]

REVIEWER_TEMPLATE_COLUMNS = [
    "row_id",
    "file_name",
    "label_cn",
    "relative_path",
    "source_dir",
    "review_status",
    "reviewer",
    "reviewed_at",
    "reviewed_manufacturing_evidence_sources",
    "reviewed_manufacturing_evidence_payload_json",
    "review_notes",
    "suggested_manufacturing_evidence_sources",
    "suggested_manufacturing_evidence_payload_json",
    "gap_reasons",
]

REVIEW_BATCH_TEMPLATE_COLUMNS = [
    "review_batch",
    "batch_rank",
    "label_gap_row_count",
    *REVIEWER_TEMPLATE_COLUMNS,
]

REVIEWER_TEMPLATE_PREFLIGHT_GAP_COLUMNS = [
    "row_id",
    "file_name",
    "label_cn",
    "relative_path",
    "source_dir",
    "review_status",
    "reviewer",
    "reviewed_at",
    "preflight_reasons",
    "preflight_reason_count",
    "duplicate_row",
    "matched_manifest_row",
    "ambiguous_file_name_match",
    "source_ready",
    "payload_ready",
    "detail_ready",
    "reviewed_manufacturing_evidence_sources",
    "reviewed_manufacturing_evidence_payload_json",
    "review_notes",
    "suggested_manufacturing_evidence_sources",
    "suggested_manufacturing_evidence_payload_json",
]

REVIEWER_TEMPLATE_APPLY_AUDIT_COLUMNS = [
    "row_id",
    "file_name",
    "label_cn",
    "relative_path",
    "source_dir",
    "apply_status",
    "apply_reasons",
    "matched_manifest_row",
    "review_status",
    "reviewer",
    "reviewed_at",
    "source_ready",
    "payload_ready",
    "detail_ready",
    "reviewed_manufacturing_evidence_sources",
    "reviewed_manufacturing_evidence_payload_json",
    "review_notes",
]

REVIEW_MANIFEST_MERGE_AUDIT_COLUMNS = [
    "row_id",
    "file_name",
    "label_cn",
    "relative_path",
    "source_dir",
    "merge_status",
    "merge_reasons",
    "matched_base_row",
    "review_status",
    "reviewer",
    "reviewed_at",
    "source_ready",
    "payload_ready",
    "detail_ready",
    "reviewed_manufacturing_evidence_sources",
    "reviewed_manufacturing_evidence_payload_json",
    "review_notes",
]

MERGED_REVIEW_COLUMNS = [
    "reviewed_manufacturing_evidence_sources",
    "reviewed_manufacturing_evidence_payload_json",
    "review_status",
    "reviewer",
    "reviewed_at",
    "review_notes",
]

APPROVED_REVIEW_STATUSES = {
    "accepted",
    "approved",
    "confirmed",
    "release_approved",
    "reviewed",
}
REVIEWER_COLUMNS = ("reviewer", "reviewed_by", "reviewer_id")
REVIEWED_AT_COLUMNS = ("reviewed_at", "review_date", "reviewed_date")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    rows, _columns = _read_csv_with_columns(path)
    return rows


def _read_csv_with_columns(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _json_cell(value: Any) -> str:
    if not value:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _decode_json_cell(value: Any) -> Any:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _row_file_name(row: Dict[str, Any]) -> str:
    return str(
        row.get("file_name")
        or row.get("filename")
        or row.get("file")
        or row.get("path")
        or ""
    ).strip()


def _row_label(row: Dict[str, Any]) -> str:
    return str(
        row.get("label_cn")
        or row.get("true_label")
        or row.get("expected_label")
        or row.get("part_type")
        or ""
    ).strip()


def _evidence_items_from_row(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    decoded = _decode_json_cell(row.get("manufacturing_evidence"))
    if isinstance(decoded, list):
        return [dict(item) for item in decoded if isinstance(item, dict)]

    sources, _reviewed = _parse_manufacturing_source_tokens(
        row.get("manufacturing_evidence_sources")
    )
    return [{"source": source} for source in sources]


def _suggest_payloads_from_evidence(
    evidence: Iterable[Dict[str, Any]],
) -> Dict[str, Dict[str, str]]:
    payloads: Dict[str, Dict[str, str]] = {}
    for item in evidence:
        source = _normalize_manufacturing_source_token(item.get("source"))
        if not source:
            continue
        source_payload = payloads.setdefault(source, {})
        for field_name in MANUFACTURING_PAYLOAD_FIELDS:
            cleaned = _clean_expected_payload_value(item.get(field_name))
            if cleaned:
                source_payload[field_name] = cleaned
        source_payload.update(_flatten_expected_detail_fields(item.get("details")))
    return {source: payload for source, payload in payloads.items() if payload}


def _source_tokens_from_evidence(evidence: Iterable[Dict[str, Any]]) -> Tuple[str, ...]:
    sources: List[str] = []
    for item in evidence:
        source = _normalize_manufacturing_source_token(item.get("source"))
        if source:
            sources.append(source)
    return tuple(dict.fromkeys(sources))


def build_review_rows(
    rows: Iterable[Dict[str, str]],
    *,
    prefill_reviewed_from_suggestions: bool = False,
) -> List[Dict[str, str]]:
    review_rows: List[Dict[str, str]] = []
    for row in rows:
        evidence = _evidence_items_from_row(row)
        suggested_sources = _source_tokens_from_evidence(evidence)
        suggested_payloads = _suggest_payloads_from_evidence(evidence)
        suggested_sources_text = ";".join(suggested_sources)
        suggested_payload_json = _json_cell(suggested_payloads)
        review_rows.append(
            {
                "file_name": _row_file_name(row),
                "label_cn": _row_label(row),
                "relative_path": str(row.get("relative_path") or "").strip(),
                "source_dir": str(row.get("source_dir") or "").strip(),
                "reviewed_manufacturing_evidence_sources": (
                    suggested_sources_text if prefill_reviewed_from_suggestions else ""
                ),
                "reviewed_manufacturing_evidence_payload_json": (
                    suggested_payload_json if prefill_reviewed_from_suggestions else ""
                ),
                "review_status": "needs_human_review",
                "reviewer": "",
                "reviewed_at": "",
                "review_notes": "",
                "suggested_manufacturing_evidence_sources": suggested_sources_text,
                "suggested_manufacturing_evidence_payload_json": suggested_payload_json,
                "actual_manufacturing_evidence": _json_cell(evidence),
            }
        )
    return review_rows


def _has_detail_payload(payloads: Dict[str, Dict[str, str]]) -> bool:
    for source_payload in payloads.values():
        for field_name in source_payload:
            if str(field_name).startswith(MANUFACTURING_PAYLOAD_DETAIL_PREFIX):
                return True
    return False


def _payload_field_count(payloads: Dict[str, Dict[str, str]]) -> int:
    return sum(len(source_payload) for source_payload in payloads.values())


def _detail_field_count(payloads: Dict[str, Dict[str, str]]) -> int:
    return sum(
        1
        for source_payload in payloads.values()
        for field_name in source_payload
        if str(field_name).startswith(MANUFACTURING_PAYLOAD_DETAIL_PREFIX)
    )


def _payload_field_names(payloads: Dict[str, Dict[str, str]]) -> Tuple[str, ...]:
    names: List[str] = []
    for source, source_payload in sorted(payloads.items()):
        for field_name in sorted(source_payload):
            names.append(f"{source}.{field_name}")
    return tuple(names)


def _normalize_payload_fields(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    normalized: Dict[str, str] = {}
    for field_name in MANUFACTURING_PAYLOAD_FIELDS:
        cleaned = _clean_expected_payload_value(value.get(field_name))
        if cleaned:
            normalized[field_name] = cleaned
    normalized.update(_flatten_expected_detail_fields(value.get("details")))
    for field_name, raw_value in value.items():
        cleaned_field = str(field_name or "").strip()
        if not cleaned_field.startswith(MANUFACTURING_PAYLOAD_DETAIL_PREFIX):
            continue
        cleaned_value = _clean_expected_payload_value(raw_value)
        if cleaned_value:
            normalized[cleaned_field] = cleaned_value
    return normalized


def _payloads_from_json_cell(value: Any) -> Dict[str, Dict[str, str]]:
    decoded = _decode_json_cell(value)
    if not isinstance(decoded, dict):
        return {}

    payloads: Dict[str, Dict[str, str]] = {}
    for source, source_value in decoded.items():
        normalized_source = _normalize_manufacturing_source_token(source)
        if not normalized_source:
            continue
        normalized_fields = _normalize_payload_fields(source_value)
        if normalized_fields:
            payloads.setdefault(normalized_source, {}).update(normalized_fields)
    return payloads


def _evidence_items_from_json_cell(value: Any) -> List[Dict[str, Any]]:
    decoded = _decode_json_cell(value)
    if isinstance(decoded, list):
        return [dict(item) for item in decoded if isinstance(item, dict)]
    if isinstance(decoded, dict):
        items: List[Dict[str, Any]] = []
        for source, source_value in decoded.items():
            item = {"source": source}
            if isinstance(source_value, dict):
                item.update(source_value)
            elif source_value:
                item["value"] = source_value
            items.append(item)
        return items
    return []


def _actual_evidence_items_from_row(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    for column in ("actual_manufacturing_evidence", "manufacturing_evidence"):
        evidence = _evidence_items_from_json_cell(row.get(column))
        if evidence:
            return evidence
    return _evidence_items_from_row(row)


def _detail_keys_from_evidence(
    evidence: Iterable[Dict[str, Any]],
) -> Tuple[str, ...]:
    keys: List[str] = []
    for item in evidence:
        keys.extend(_flatten_expected_detail_fields(item.get("details")).keys())
    return tuple(dict.fromkeys(keys))


def _actual_evidence_summary(evidence: Iterable[Dict[str, Any]]) -> str:
    summaries: List[str] = []
    for item in evidence:
        parts: List[str] = []
        source = _normalize_manufacturing_source_token(item.get("source"))
        if source:
            parts.append(f"source={source}")
        for field_name in MANUFACTURING_PAYLOAD_FIELDS:
            cleaned = _clean_expected_payload_value(item.get(field_name))
            if cleaned:
                parts.append(f"{field_name}={cleaned}")
        detail_keys = _detail_keys_from_evidence([item])
        if detail_keys:
            parts.append(f"details={','.join(detail_keys)}")
        if parts:
            summaries.append(",".join(parts))
    return " | ".join(summaries)


def _review_status_approved(row: Dict[str, str]) -> bool:
    if "review_status" not in row:
        return True
    status = str(row.get("review_status") or "").strip().lower()
    return status in APPROVED_REVIEW_STATUSES


def _reviewer_metadata_present(row: Dict[str, str]) -> bool:
    reviewer = any(str(row.get(column) or "").strip() for column in REVIEWER_COLUMNS)
    reviewed_at = any(str(row.get(column) or "").strip() for column in REVIEWED_AT_COLUMNS)
    return reviewer and reviewed_at


def _first_non_empty(row: Dict[str, str], columns: Iterable[str]) -> str:
    for column in columns:
        value = str(row.get(column) or "").strip()
        if value:
            return value
    return ""


def _row_has_review_content(row: Dict[str, str]) -> bool:
    _sources, source_reviewed = _extract_expected_manufacturing_sources(row)
    _payloads, payload_reviewed = _extract_expected_manufacturing_payloads(row)
    return bool(source_reviewed or payload_reviewed)


def _manifest_indices(
    rows: Iterable[Dict[str, str]],
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    by_relative_path: Dict[str, Dict[str, str]] = {}
    by_file_name: Dict[str, Dict[str, str]] = {}
    for row in rows:
        relative_path = str(row.get("relative_path") or "").strip()
        if relative_path and relative_path not in by_relative_path:
            by_relative_path[relative_path] = row
        file_name = _row_file_name(row)
        if file_name and file_name not in by_file_name:
            by_file_name[file_name] = row
    return by_relative_path, by_file_name


def _duplicate_row_identifiers(rows: Iterable[Dict[str, str]]) -> List[str]:
    counts: Dict[str, int] = {}
    duplicates: List[str] = []
    for row in rows:
        identifier = _row_identifier(row)
        counts[identifier] = counts.get(identifier, 0) + 1
        if counts[identifier] == 2:
            duplicates.append(identifier)
    return duplicates


def _duplicate_file_names(rows: Iterable[Dict[str, str]]) -> List[str]:
    counts: Dict[str, int] = {}
    duplicates: List[str] = []
    for row in rows:
        file_name = _row_file_name(row)
        if not file_name:
            continue
        counts[file_name] = counts.get(file_name, 0) + 1
        if counts[file_name] == 2:
            duplicates.append(file_name)
    return duplicates


def _ambiguous_file_name_match(
    review_row: Dict[str, str],
    *,
    duplicate_file_names: set[str],
    by_relative_path: Dict[str, Dict[str, str]],
) -> bool:
    relative_path = str(review_row.get("relative_path") or "").strip()
    if relative_path and relative_path in by_relative_path:
        return False
    file_name = _row_file_name(review_row)
    return bool(file_name and file_name in duplicate_file_names)


def _matching_base_row(
    review_row: Dict[str, str],
    by_relative_path: Dict[str, Dict[str, str]],
    by_file_name: Dict[str, Dict[str, str]],
) -> Optional[Dict[str, str]]:
    relative_path = str(review_row.get("relative_path") or "").strip()
    if relative_path and relative_path in by_relative_path:
        return by_relative_path[relative_path]
    file_name = _row_file_name(review_row)
    if file_name and file_name in by_file_name:
        return by_file_name[file_name]
    return None


def _review_updates(review_row: Dict[str, str]) -> Dict[str, str]:
    updates = {
        "reviewed_manufacturing_evidence_sources": _first_non_empty(
            review_row,
            (
                "reviewed_manufacturing_evidence_sources",
                "reviewed_manufacturing_sources",
                "expected_manufacturing_evidence_sources",
                "expected_manufacturing_sources",
            ),
        ),
        "reviewed_manufacturing_evidence_payload_json": _first_non_empty(
            review_row,
            (
                "reviewed_manufacturing_evidence_payload_json",
                "reviewed_manufacturing_payload_json",
                "expected_manufacturing_evidence_payload_json",
                "expected_manufacturing_payload_json",
            ),
        ),
        "review_status": _first_non_empty(review_row, ("review_status",)),
        "reviewer": _first_non_empty(review_row, REVIEWER_COLUMNS),
        "reviewed_at": _first_non_empty(review_row, REVIEWED_AT_COLUMNS),
        "review_notes": _first_non_empty(review_row, ("review_notes", "notes")),
    }
    non_empty_updates = {
        column: value for column, value in updates.items() if str(value or "").strip()
    }
    return non_empty_updates


def _apply_review_values(base_row: Dict[str, str], review_row: Dict[str, str]) -> bool:
    non_empty_updates = _review_updates(review_row)
    if not non_empty_updates:
        return False
    base_row.update(non_empty_updates)
    return True


def merge_approved_review_rows(
    base_rows: Iterable[Dict[str, str]],
    review_rows: Iterable[Dict[str, str]],
    *,
    require_reviewer_metadata: bool = False,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    merged_rows = [dict(row) for row in base_rows]
    review_materialized_rows = [dict(row) for row in review_rows]
    base_manifest_duplicate_identifiers = _duplicate_row_identifiers(merged_rows)
    base_manifest_duplicate_file_names = set(_duplicate_file_names(merged_rows))
    by_relative_path, by_file_name = _manifest_indices(merged_rows)

    review_row_count = 0
    approved_review_row_count = 0
    merged_row_count = 0
    skipped_no_review_content_row_count = 0
    skipped_unapproved_review_row_count = 0
    skipped_missing_metadata_row_count = 0
    unmatched_review_row_count = 0
    ambiguous_file_name_match_row_count = 0

    for review_row in review_materialized_rows:
        review_row_count += 1
        if not _row_has_review_content(review_row):
            skipped_no_review_content_row_count += 1
            continue
        if not _review_status_approved(review_row):
            skipped_unapproved_review_row_count += 1
            continue
        approved_review_row_count += 1
        if require_reviewer_metadata and not _reviewer_metadata_present(review_row):
            skipped_missing_metadata_row_count += 1
            continue
        if base_manifest_duplicate_identifiers:
            continue
        if _ambiguous_file_name_match(
            review_row,
            duplicate_file_names=base_manifest_duplicate_file_names,
            by_relative_path=by_relative_path,
        ):
            ambiguous_file_name_match_row_count += 1
            continue
        base_row = _matching_base_row(review_row, by_relative_path, by_file_name)
        if base_row is None:
            unmatched_review_row_count += 1
            continue
        if _apply_review_values(base_row, review_row):
            merged_row_count += 1

    blocking_reasons: List[str] = []
    if merged_row_count == 0:
        blocking_reasons.append("no_approved_review_rows_merged")
    if base_manifest_duplicate_identifiers:
        blocking_reasons.append("base_manifest_duplicate_rows")
    if ambiguous_file_name_match_row_count > 0:
        blocking_reasons.append("ambiguous_file_name_match_rows")

    return merged_rows, {
        "base_row_count": len(merged_rows),
        "base_manifest_duplicate_identity_count": len(
            base_manifest_duplicate_identifiers
        ),
        "base_manifest_duplicate_identifiers": (
            base_manifest_duplicate_identifiers[:20]
        ),
        "review_row_count": review_row_count,
        "approved_review_row_count": approved_review_row_count,
        "merged_row_count": merged_row_count,
        "skipped_no_review_content_row_count": skipped_no_review_content_row_count,
        "skipped_unapproved_review_row_count": skipped_unapproved_review_row_count,
        "skipped_missing_metadata_row_count": skipped_missing_metadata_row_count,
        "unmatched_review_row_count": unmatched_review_row_count,
        "ambiguous_file_name_match_row_count": (
            ambiguous_file_name_match_row_count
        ),
        "require_reviewer_metadata": bool(require_reviewer_metadata),
        "status": "merged" if not blocking_reasons else "blocked",
        "blocking_reasons": blocking_reasons,
    }


def build_review_manifest_merge_audit_rows(
    base_rows: Iterable[Dict[str, str]],
    review_rows: Iterable[Dict[str, str]],
    *,
    require_reviewer_metadata: bool = False,
) -> List[Dict[str, str]]:
    materialized_base_rows = [dict(row) for row in base_rows]
    base_manifest_duplicate_identifiers = _duplicate_row_identifiers(
        materialized_base_rows
    )
    base_manifest_duplicate_file_names = set(
        _duplicate_file_names(materialized_base_rows)
    )
    by_relative_path, by_file_name = _manifest_indices(materialized_base_rows)
    audit_rows: List[Dict[str, str]] = []

    for review_row in review_rows:
        identifier = _row_identifier(review_row)
        reasons: List[str] = []
        matched_base_row = False
        merge_status = "merged"

        if not _row_has_review_content(review_row):
            merge_status = "skipped_no_review_content"
            reasons.append("fill reviewed source or payload fields")
        elif not _review_status_approved(review_row):
            merge_status = "skipped_unapproved_review"
            reasons.append("set approved review_status")
        elif (
            require_reviewer_metadata
            and not _reviewer_metadata_present(review_row)
        ):
            merge_status = "skipped_missing_metadata"
            reasons.append("fill reviewer and reviewed_at")
        elif base_manifest_duplicate_identifiers:
            merge_status = "blocked_duplicate_base_manifest"
            reasons.append("deduplicate base benchmark manifest")
        elif _ambiguous_file_name_match(
            review_row,
            duplicate_file_names=base_manifest_duplicate_file_names,
            by_relative_path=by_relative_path,
        ):
            merge_status = "ambiguous_file_name_match"
            reasons.append("add relative_path to disambiguate duplicate file_name")
        else:
            base_row = _matching_base_row(review_row, by_relative_path, by_file_name)
            matched_base_row = base_row is not None
            if base_row is None:
                merge_status = "unmatched_review_row"
                reasons.append("match row_id to base benchmark manifest")
            elif not _review_updates(review_row):
                merge_status = "skipped_empty_updates"
                reasons.append("fill at least one editable review field")

        _sources, source_reviewed = _extract_expected_manufacturing_sources(review_row)
        payloads, payload_reviewed = _extract_expected_manufacturing_payloads(
            review_row
        )
        detail_ready = payload_reviewed and _has_detail_payload(payloads)
        audit_rows.append(
            {
                "row_id": identifier,
                "file_name": _row_file_name(review_row),
                "label_cn": _row_label(review_row),
                "relative_path": str(review_row.get("relative_path") or "").strip(),
                "source_dir": str(review_row.get("source_dir") or "").strip(),
                "merge_status": merge_status,
                "merge_reasons": "; ".join(reasons),
                "matched_base_row": str(matched_base_row).lower(),
                "review_status": str(review_row.get("review_status") or "").strip(),
                "reviewer": _first_non_empty(review_row, REVIEWER_COLUMNS),
                "reviewed_at": _first_non_empty(review_row, REVIEWED_AT_COLUMNS),
                "source_ready": str(source_reviewed).lower(),
                "payload_ready": str(payload_reviewed).lower(),
                "detail_ready": str(detail_ready).lower(),
                "reviewed_manufacturing_evidence_sources": str(
                    review_row.get("reviewed_manufacturing_evidence_sources") or ""
                ),
                "reviewed_manufacturing_evidence_payload_json": str(
                    review_row.get("reviewed_manufacturing_evidence_payload_json") or ""
                ),
                "review_notes": str(review_row.get("review_notes") or ""),
            }
        )
    return audit_rows


def apply_reviewer_template_rows(
    manifest_rows: Iterable[Dict[str, str]],
    template_rows: Iterable[Dict[str, str]],
    *,
    require_reviewer_metadata: bool = False,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    applied_rows = [dict(row) for row in manifest_rows]
    template_materialized_rows = [dict(row) for row in template_rows]
    base_manifest_duplicate_identifiers = _duplicate_row_identifiers(applied_rows)
    base_manifest_duplicate_file_names = set(_duplicate_file_names(applied_rows))
    by_relative_path, by_file_name = _manifest_indices(applied_rows)

    template_row_count = 0
    approved_template_row_count = 0
    applied_row_count = 0
    skipped_no_review_content_row_count = 0
    skipped_unapproved_template_row_count = 0
    skipped_missing_metadata_row_count = 0
    unmatched_template_row_count = 0
    ambiguous_file_name_match_row_count = 0

    for template_row in template_materialized_rows:
        template_row_count += 1
        if not _row_has_review_content(template_row):
            skipped_no_review_content_row_count += 1
            continue
        if not _review_status_approved(template_row):
            skipped_unapproved_template_row_count += 1
            continue
        approved_template_row_count += 1
        if require_reviewer_metadata and not _reviewer_metadata_present(template_row):
            skipped_missing_metadata_row_count += 1
            continue
        if base_manifest_duplicate_identifiers:
            continue
        if _ambiguous_file_name_match(
            template_row,
            duplicate_file_names=base_manifest_duplicate_file_names,
            by_relative_path=by_relative_path,
        ):
            ambiguous_file_name_match_row_count += 1
            continue
        manifest_row = _matching_base_row(template_row, by_relative_path, by_file_name)
        if manifest_row is None:
            unmatched_template_row_count += 1
            continue
        if _apply_review_values(manifest_row, template_row):
            applied_row_count += 1

    blocking_reasons: List[str] = []
    if applied_row_count == 0:
        blocking_reasons.append("no_approved_template_rows_applied")
    if base_manifest_duplicate_identifiers:
        blocking_reasons.append("base_manifest_duplicate_rows")
    if ambiguous_file_name_match_row_count > 0:
        blocking_reasons.append("ambiguous_file_name_match_rows")

    return applied_rows, {
        "manifest_row_count": len(applied_rows),
        "base_manifest_duplicate_identity_count": len(
            base_manifest_duplicate_identifiers
        ),
        "base_manifest_duplicate_identifiers": (
            base_manifest_duplicate_identifiers[:20]
        ),
        "template_row_count": template_row_count,
        "approved_template_row_count": approved_template_row_count,
        "applied_row_count": applied_row_count,
        "skipped_no_review_content_row_count": skipped_no_review_content_row_count,
        "skipped_unapproved_template_row_count": (
            skipped_unapproved_template_row_count
        ),
        "skipped_missing_metadata_row_count": skipped_missing_metadata_row_count,
        "unmatched_template_row_count": unmatched_template_row_count,
        "ambiguous_file_name_match_row_count": ambiguous_file_name_match_row_count,
        "require_reviewer_metadata": bool(require_reviewer_metadata),
        "status": "applied" if not blocking_reasons else "blocked",
        "blocking_reasons": blocking_reasons,
    }


def _explicit_review_status_approved(row: Dict[str, str]) -> bool:
    status = str(row.get("review_status") or "").strip().lower()
    return status in APPROVED_REVIEW_STATUSES


def validate_reviewer_template_rows(
    rows: Iterable[Dict[str, str]],
    *,
    min_ready_rows: int = 30,
    require_reviewer_metadata: bool = False,
    base_rows: Optional[Iterable[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    base_match_required = base_rows is not None
    base_manifest_row_count = 0
    base_manifest_duplicate_identifiers: List[str] = []
    base_manifest_duplicate_file_names: set[str] = set()
    base_indices: Optional[
        Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]
    ] = None
    if base_rows is not None:
        materialized_base_rows = [dict(row) for row in base_rows]
        base_manifest_row_count = len(materialized_base_rows)
        base_manifest_duplicate_identifiers = _duplicate_row_identifiers(
            materialized_base_rows
        )
        base_manifest_duplicate_file_names = set(
            _duplicate_file_names(materialized_base_rows)
        )
        base_indices = _manifest_indices(materialized_base_rows)

    template_row_count = 0
    ready_template_row_count = 0
    approved_template_row_count = 0
    no_review_content_row_count = 0
    unapproved_template_row_count = 0
    reviewer_metadata_missing_row_count = 0
    source_missing_row_count = 0
    payload_missing_row_count = 0
    payload_detail_missing_row_count = 0
    duplicate_template_row_count = 0
    unmatched_template_row_count = 0
    ambiguous_file_name_match_row_count = 0
    seen_identifiers: set[str] = set()
    duplicate_identifiers: List[str] = []

    for row in rows:
        template_row_count += 1
        identifier = _row_identifier(row)
        duplicate_row = identifier in seen_identifiers
        if duplicate_row:
            duplicate_template_row_count += 1
            if identifier not in duplicate_identifiers:
                duplicate_identifiers.append(identifier)
        seen_identifiers.add(identifier)

        _sources, source_reviewed = _extract_expected_manufacturing_sources(row)
        payloads, payload_reviewed = _extract_expected_manufacturing_payloads(row)
        has_review_content = bool(source_reviewed or payload_reviewed)
        status_approved = _explicit_review_status_approved(row)
        metadata_present = _reviewer_metadata_present(row)
        detail_ready = payload_reviewed and _has_detail_payload(payloads)
        matched_manifest_row = True
        ambiguous_file_name_match = False
        if base_indices is not None:
            ambiguous_file_name_match = (
                not base_manifest_duplicate_identifiers
                and _ambiguous_file_name_match(
                    row,
                    duplicate_file_names=base_manifest_duplicate_file_names,
                    by_relative_path=base_indices[0],
                )
            )
            if ambiguous_file_name_match:
                matched_manifest_row = False
            else:
                matched_manifest_row = (
                    _matching_base_row(row, base_indices[0], base_indices[1])
                    is not None
                )

        if not has_review_content:
            no_review_content_row_count += 1
        if not source_reviewed:
            source_missing_row_count += 1
        if not payload_reviewed:
            payload_missing_row_count += 1
        elif not detail_ready:
            payload_detail_missing_row_count += 1
        if has_review_content and not status_approved:
            unapproved_template_row_count += 1
        if status_approved:
            approved_template_row_count += 1
        if (
            has_review_content
            and require_reviewer_metadata
            and not metadata_present
        ):
            reviewer_metadata_missing_row_count += 1

        if (
            source_reviewed
            and payload_reviewed
            and detail_ready
            and status_approved
            and not duplicate_row
            and matched_manifest_row
            and (not require_reviewer_metadata or metadata_present)
        ):
            ready_template_row_count += 1
        if base_match_required and ambiguous_file_name_match:
            ambiguous_file_name_match_row_count += 1
        elif base_match_required and not matched_manifest_row:
            unmatched_template_row_count += 1

    blocking_reasons: List[str] = []
    if ready_template_row_count < min_ready_rows:
        blocking_reasons.append("ready_template_row_count_below_minimum")
    if base_match_required and base_manifest_row_count == 0:
        blocking_reasons.append("base_manifest_empty")
    if base_manifest_duplicate_identifiers:
        blocking_reasons.append("base_manifest_duplicate_rows")
    if no_review_content_row_count > 0:
        blocking_reasons.append("no_review_content_rows")
    if unapproved_template_row_count > 0:
        blocking_reasons.append("unapproved_template_rows")
    if require_reviewer_metadata and reviewer_metadata_missing_row_count > 0:
        blocking_reasons.append("reviewer_metadata_missing")
    if source_missing_row_count > 0:
        blocking_reasons.append("source_labels_missing")
    if payload_missing_row_count > 0:
        blocking_reasons.append("payload_labels_missing")
    if payload_detail_missing_row_count > 0:
        blocking_reasons.append("payload_detail_labels_missing")
    if duplicate_template_row_count > 0:
        blocking_reasons.append("duplicate_template_rows")
    if unmatched_template_row_count > 0:
        blocking_reasons.append("unmatched_template_rows")
    if ambiguous_file_name_match_row_count > 0:
        blocking_reasons.append("ambiguous_file_name_match_rows")

    return {
        "template_row_count": template_row_count,
        "base_manifest_match_required": base_match_required,
        "base_manifest_row_count": base_manifest_row_count,
        "base_manifest_duplicate_identity_count": len(
            base_manifest_duplicate_identifiers
        ),
        "base_manifest_duplicate_identifiers": (
            base_manifest_duplicate_identifiers[:20]
        ),
        "min_ready_rows": min_ready_rows,
        "approved_review_statuses": sorted(APPROVED_REVIEW_STATUSES),
        "approved_template_row_count": approved_template_row_count,
        "ready_template_row_count": ready_template_row_count,
        "no_review_content_row_count": no_review_content_row_count,
        "unapproved_template_row_count": unapproved_template_row_count,
        "require_reviewer_metadata": bool(require_reviewer_metadata),
        "reviewer_metadata_missing_row_count": (
            reviewer_metadata_missing_row_count
        ),
        "source_missing_row_count": source_missing_row_count,
        "payload_missing_row_count": payload_missing_row_count,
        "payload_detail_missing_row_count": payload_detail_missing_row_count,
        "duplicate_template_row_count": duplicate_template_row_count,
        "unmatched_template_row_count": unmatched_template_row_count,
        "ambiguous_file_name_match_row_count": (
            ambiguous_file_name_match_row_count
        ),
        "duplicate_identifiers": duplicate_identifiers[:20],
        "status": "ready" if not blocking_reasons else "blocked",
        "blocking_reasons": blocking_reasons,
    }


def _reviewer_template_preflight_reasons(
    row: Dict[str, str],
    *,
    duplicate: bool = False,
    matched_manifest_row: Optional[bool] = None,
    ambiguous_file_name_match: bool = False,
    require_reviewer_metadata: bool = False,
) -> List[str]:
    _sources, source_reviewed = _extract_expected_manufacturing_sources(row)
    payloads, payload_reviewed = _extract_expected_manufacturing_payloads(row)
    has_review_content = bool(source_reviewed or payload_reviewed)
    reasons: List[str] = []

    if duplicate:
        reasons.append("deduplicate row_id")
    if ambiguous_file_name_match:
        reasons.append("add relative_path to disambiguate duplicate file_name")
    elif matched_manifest_row is False:
        reasons.append("match row identity to review manifest")
    if not source_reviewed:
        reasons.append("fill reviewed_manufacturing_evidence_sources")
    if not payload_reviewed:
        reasons.append("fill reviewed_manufacturing_evidence_payload_json")
    elif not _has_detail_payload(payloads):
        reasons.append("add details.* payload labels")
    if has_review_content and not _explicit_review_status_approved(row):
        reasons.append("set approved review_status")
    if (
        has_review_content
        and require_reviewer_metadata
        and not _reviewer_metadata_present(row)
    ):
        reasons.append("fill reviewer and reviewed_at")
    return reasons


def validate_review_manifest_rows(
    rows: Iterable[Dict[str, str]],
    *,
    min_reviewed_samples: int = 30,
    require_reviewer_metadata: bool = False,
) -> Dict[str, Any]:
    row_count = 0
    approved_review_sample_count = 0
    unapproved_review_sample_count = 0
    reviewer_metadata_missing_sample_count = 0
    source_reviewed_sample_count = 0
    payload_reviewed_sample_count = 0
    payload_detail_reviewed_sample_count = 0
    payload_expected_field_total = 0
    payload_detail_expected_field_total = 0
    blocking_reasons: List[str] = []

    for row in rows:
        row_count += 1
        _sources, source_reviewed = _extract_expected_manufacturing_sources(row)
        payloads, payload_reviewed = _extract_expected_manufacturing_payloads(row)
        has_review_content = bool(source_reviewed or payload_reviewed)
        status_approved = _review_status_approved(row)
        metadata_present = _reviewer_metadata_present(row)
        if has_review_content and not status_approved:
            unapproved_review_sample_count += 1
            continue
        if has_review_content and require_reviewer_metadata and not metadata_present:
            reviewer_metadata_missing_sample_count += 1
            continue
        if has_review_content and status_approved:
            approved_review_sample_count += 1
        if source_reviewed:
            source_reviewed_sample_count += 1
        if payload_reviewed:
            payload_reviewed_sample_count += 1
            payload_expected_field_total += _payload_field_count(payloads)
            payload_detail_expected_field_total += _detail_field_count(payloads)
        if payload_reviewed and _has_detail_payload(payloads):
            payload_detail_reviewed_sample_count += 1

    if source_reviewed_sample_count < min_reviewed_samples:
        blocking_reasons.append("source_reviewed_sample_count_below_minimum")
    if payload_reviewed_sample_count < min_reviewed_samples:
        blocking_reasons.append("payload_reviewed_sample_count_below_minimum")
    if payload_detail_reviewed_sample_count < min_reviewed_samples:
        blocking_reasons.append("payload_detail_reviewed_sample_count_below_minimum")
    if require_reviewer_metadata and reviewer_metadata_missing_sample_count > 0:
        blocking_reasons.append("reviewer_metadata_missing")

    return {
        "row_count": row_count,
        "min_reviewed_samples": min_reviewed_samples,
        "approved_review_statuses": sorted(APPROVED_REVIEW_STATUSES),
        "approved_review_sample_count": approved_review_sample_count,
        "unapproved_review_sample_count": unapproved_review_sample_count,
        "require_reviewer_metadata": bool(require_reviewer_metadata),
        "reviewer_metadata_missing_sample_count": (
            reviewer_metadata_missing_sample_count
        ),
        "source_reviewed_sample_count": source_reviewed_sample_count,
        "payload_reviewed_sample_count": payload_reviewed_sample_count,
        "payload_detail_reviewed_sample_count": payload_detail_reviewed_sample_count,
        "payload_expected_field_total": payload_expected_field_total,
        "payload_detail_expected_field_total": payload_detail_expected_field_total,
        "source_reviewed_ready": source_reviewed_sample_count >= min_reviewed_samples,
        "payload_reviewed_ready": payload_reviewed_sample_count >= min_reviewed_samples,
        "payload_detail_reviewed_ready": (
            payload_detail_reviewed_sample_count >= min_reviewed_samples
        ),
        "status": "release_label_ready" if not blocking_reasons else "blocked",
        "blocking_reasons": blocking_reasons,
    }


def _markdown_cell(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    return text.replace("|", "\\|").replace("\n", " ")


def _row_identifier(row: Dict[str, str]) -> str:
    return str(row.get("relative_path") or _row_file_name(row) or "<unknown>").strip()


def _review_gap_reasons(
    row: Dict[str, str],
    *,
    require_reviewer_metadata: bool = False,
) -> List[str]:
    _sources, source_reviewed = _extract_expected_manufacturing_sources(row)
    payloads, payload_reviewed = _extract_expected_manufacturing_payloads(row)
    has_review_content = bool(source_reviewed or payload_reviewed)
    reasons: List[str] = []

    if not source_reviewed:
        reasons.append("fill reviewed_manufacturing_evidence_sources")
    if not payload_reviewed:
        reasons.append("fill reviewed_manufacturing_evidence_payload_json")
    elif not _has_detail_payload(payloads):
        reasons.append("add details.* payload labels")
    if has_review_content and not _review_status_approved(row):
        reasons.append("set approved review_status")
    if (
        has_review_content
        and require_reviewer_metadata
        and not _reviewer_metadata_present(row)
    ):
        reasons.append("fill reviewer and reviewed_at")
    return reasons


def _row_counts_for_release(
    row: Dict[str, str],
    *,
    require_reviewer_metadata: bool = False,
) -> Tuple[bool, bool, bool]:
    _sources, source_reviewed = _extract_expected_manufacturing_sources(row)
    payloads, payload_reviewed = _extract_expected_manufacturing_payloads(row)
    has_review_content = bool(source_reviewed or payload_reviewed)
    if has_review_content and not _review_status_approved(row):
        return False, False, False
    if (
        has_review_content
        and require_reviewer_metadata
        and not _reviewer_metadata_present(row)
    ):
        return False, False, False
    return (
        source_reviewed,
        payload_reviewed,
        payload_reviewed and _has_detail_payload(payloads),
    )


def build_review_progress_markdown(
    rows: Iterable[Dict[str, str]],
    summary: Dict[str, Any],
    *,
    max_gap_rows: int = 30,
    require_reviewer_metadata: bool = False,
) -> str:
    materialized_rows = [dict(row) for row in rows]
    min_reviewed_samples = int(summary.get("min_reviewed_samples") or 0)
    source_count = int(summary.get("source_reviewed_sample_count") or 0)
    payload_count = int(summary.get("payload_reviewed_sample_count") or 0)
    detail_count = int(summary.get("payload_detail_reviewed_sample_count") or 0)

    lines = [
        "# Manufacturing Evidence Review Progress",
        "",
        "## Status",
        "",
        f"- status: `{summary.get('status', 'unknown')}`",
        f"- row_count: `{summary.get('row_count', len(materialized_rows))}`",
        f"- min_reviewed_samples: `{min_reviewed_samples}`",
        (
            "- require_reviewer_metadata: "
            f"`{str(summary.get('require_reviewer_metadata', False)).lower()}`"
        ),
        "",
        "## Release Label Counts",
        "",
        "| Evidence class | Reviewed | Required | Remaining |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| Source labels | {source_count} | {min_reviewed_samples} | "
            f"{max(0, min_reviewed_samples - source_count)} |"
        ),
        (
            f"| Payload labels | {payload_count} | {min_reviewed_samples} | "
            f"{max(0, min_reviewed_samples - payload_count)} |"
        ),
        (
            f"| Detail labels | {detail_count} | {min_reviewed_samples} | "
            f"{max(0, min_reviewed_samples - detail_count)} |"
        ),
        "",
        "## Blocking Reasons",
        "",
    ]

    blocking_reasons = list(summary.get("blocking_reasons") or [])
    if blocking_reasons:
        lines.extend(f"- `{_markdown_cell(reason)}`" for reason in blocking_reasons)
    else:
        lines.append("- None")

    label_counts: Dict[str, Dict[str, int]] = {}
    gap_rows: List[Tuple[str, str, str, List[str]]] = []
    for row in materialized_rows:
        label = _row_label(row) or "<unknown>"
        counts = label_counts.setdefault(
            label,
            {"rows": 0, "source": 0, "payload": 0, "detail": 0},
        )
        counts["rows"] += 1
        source_ok, payload_ok, detail_ok = _row_counts_for_release(
            row,
            require_reviewer_metadata=require_reviewer_metadata,
        )
        counts["source"] += int(source_ok)
        counts["payload"] += int(payload_ok)
        counts["detail"] += int(detail_ok)

        reasons = _review_gap_reasons(
            row,
            require_reviewer_metadata=require_reviewer_metadata,
        )
        if reasons:
            gap_rows.append(
                (
                    _row_identifier(row),
                    label,
                    str(row.get("review_status") or "").strip() or "<missing>",
                    reasons,
                )
            )

    lines.extend(
        [
            "",
            "## Label Progress",
            "",
            "| Label | Rows | Source | Payload | Detail |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, counts in sorted(
        label_counts.items(),
        key=lambda item: (-item[1]["rows"], item[0]),
    ):
        lines.append(
            "| "
            f"{_markdown_cell(label)} | {counts['rows']} | {counts['source']} | "
            f"{counts['payload']} | {counts['detail']} |"
        )

    lines.extend(
        [
            "",
            "## Next Review Rows",
            "",
            "| Row | Label | Review status | Gaps |",
            "| --- | --- | --- | --- |",
        ]
    )
    for identifier, label, status, reasons in gap_rows[: max(0, max_gap_rows)]:
        lines.append(
            "| "
            f"{_markdown_cell(identifier)} | {_markdown_cell(label)} | "
            f"{_markdown_cell(status)} | {_markdown_cell('; '.join(reasons))} |"
        )
    if not gap_rows:
        lines.append("| None | - | - | - |")
    if len(gap_rows) > max_gap_rows:
        lines.extend(
            [
                "",
                f"_Additional gap rows omitted: {len(gap_rows) - max_gap_rows}_",
            ]
        )

    return "\n".join(lines) + "\n"


def build_review_gap_rows(
    rows: Iterable[Dict[str, str]],
    *,
    require_reviewer_metadata: bool = False,
) -> List[Dict[str, str]]:
    gap_rows: List[Dict[str, str]] = []
    for row in rows:
        reasons = _review_gap_reasons(
            row,
            require_reviewer_metadata=require_reviewer_metadata,
        )
        if not reasons:
            continue
        source_ready, payload_ready, detail_ready = _row_counts_for_release(
            row,
            require_reviewer_metadata=require_reviewer_metadata,
        )
        gap_rows.append(
            {
                "row_id": _row_identifier(row),
                "file_name": _row_file_name(row),
                "label_cn": _row_label(row),
                "relative_path": str(row.get("relative_path") or "").strip(),
                "source_dir": str(row.get("source_dir") or "").strip(),
                "review_status": str(row.get("review_status") or "").strip(),
                "reviewer": _first_non_empty(row, REVIEWER_COLUMNS),
                "reviewed_at": _first_non_empty(row, REVIEWED_AT_COLUMNS),
                "gap_reasons": "; ".join(reasons),
                "gap_reason_count": str(len(reasons)),
                "source_ready": str(source_ready).lower(),
                "payload_ready": str(payload_ready).lower(),
                "detail_ready": str(detail_ready).lower(),
                "suggested_manufacturing_evidence_sources": str(
                    row.get("suggested_manufacturing_evidence_sources") or ""
                ),
                "suggested_manufacturing_evidence_payload_json": str(
                    row.get("suggested_manufacturing_evidence_payload_json") or ""
                ),
                "reviewed_manufacturing_evidence_sources": str(
                    row.get("reviewed_manufacturing_evidence_sources") or ""
                ),
                "reviewed_manufacturing_evidence_payload_json": str(
                    row.get("reviewed_manufacturing_evidence_payload_json") or ""
                ),
                "review_notes": str(row.get("review_notes") or ""),
            }
        )
    return gap_rows


def build_review_context_rows(
    rows: Iterable[Dict[str, str]],
    *,
    require_reviewer_metadata: bool = False,
) -> List[Dict[str, str]]:
    context_rows: List[Dict[str, str]] = []
    for row in rows:
        reasons = _review_gap_reasons(
            row,
            require_reviewer_metadata=require_reviewer_metadata,
        )
        if not reasons:
            continue

        source_ready, payload_ready, detail_ready = _row_counts_for_release(
            row,
            require_reviewer_metadata=require_reviewer_metadata,
        )
        suggested_sources, _reviewed = _parse_manufacturing_source_tokens(
            row.get("suggested_manufacturing_evidence_sources")
        )
        suggested_payloads = _payloads_from_json_cell(
            row.get("suggested_manufacturing_evidence_payload_json")
        )
        suggested_sources = tuple(
            dict.fromkeys([*suggested_sources, *suggested_payloads.keys()])
        )
        actual_evidence = _actual_evidence_items_from_row(row)
        actual_sources = _source_tokens_from_evidence(actual_evidence)
        actual_detail_keys = _detail_keys_from_evidence(actual_evidence)
        actual_evidence_json = str(row.get("actual_manufacturing_evidence") or "")
        if not actual_evidence_json and actual_evidence:
            actual_evidence_json = _json_cell(actual_evidence)

        context_rows.append(
            {
                "row_id": _row_identifier(row),
                "file_name": _row_file_name(row),
                "label_cn": _row_label(row),
                "relative_path": str(row.get("relative_path") or "").strip(),
                "source_dir": str(row.get("source_dir") or "").strip(),
                "review_status": str(row.get("review_status") or "").strip(),
                "reviewer": _first_non_empty(row, REVIEWER_COLUMNS),
                "reviewed_at": _first_non_empty(row, REVIEWED_AT_COLUMNS),
                "gap_reasons": "; ".join(reasons),
                "gap_reason_count": str(len(reasons)),
                "source_ready": str(source_ready).lower(),
                "payload_ready": str(payload_ready).lower(),
                "detail_ready": str(detail_ready).lower(),
                "suggested_manufacturing_evidence_sources": ";".join(
                    suggested_sources
                ),
                "suggested_source_count": str(len(suggested_sources)),
                "suggested_payload_field_count": str(
                    _payload_field_count(suggested_payloads)
                ),
                "suggested_detail_field_count": str(
                    _detail_field_count(suggested_payloads)
                ),
                "suggested_payload_fields": ";".join(
                    _payload_field_names(suggested_payloads)
                ),
                "suggested_manufacturing_evidence_payload_json": str(
                    row.get("suggested_manufacturing_evidence_payload_json") or ""
                ),
                "actual_evidence_item_count": str(len(actual_evidence)),
                "actual_evidence_sources": ";".join(actual_sources),
                "actual_evidence_summary": _actual_evidence_summary(actual_evidence),
                "actual_evidence_detail_keys": ";".join(actual_detail_keys),
                "actual_manufacturing_evidence": actual_evidence_json,
                "reviewed_manufacturing_evidence_sources": str(
                    row.get("reviewed_manufacturing_evidence_sources") or ""
                ),
                "reviewed_manufacturing_evidence_payload_json": str(
                    row.get("reviewed_manufacturing_evidence_payload_json") or ""
                ),
                "review_notes": str(row.get("review_notes") or ""),
            }
        )
    return context_rows


def _gap_reason_present(reasons: Iterable[str], expected: str) -> str:
    return str(any(reason == expected for reason in reasons)).lower()


def build_review_batch_rows(
    rows: Iterable[Dict[str, str]],
    *,
    max_rows_per_label: int = 5,
    require_reviewer_metadata: bool = False,
) -> List[Dict[str, str]]:
    label_buckets: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        reasons = _review_gap_reasons(
            row,
            require_reviewer_metadata=require_reviewer_metadata,
        )
        if not reasons:
            continue

        suggested_sources, _reviewed = _parse_manufacturing_source_tokens(
            row.get("suggested_manufacturing_evidence_sources")
        )
        actual_evidence = _actual_evidence_items_from_row(row)
        actual_sources = _source_tokens_from_evidence(actual_evidence)
        label = _row_label(row) or "<unknown>"
        label_buckets.setdefault(label, []).append(
            {
                "row_id": _row_identifier(row),
                "file_name": _row_file_name(row),
                "label_cn": label,
                "relative_path": str(row.get("relative_path") or "").strip(),
                "source_dir": str(row.get("source_dir") or "").strip(),
                "review_status": str(row.get("review_status") or "").strip(),
                "gap_reasons": "; ".join(reasons),
                "gap_reason_count": str(len(reasons)),
                "source_gap": _gap_reason_present(
                    reasons,
                    "fill reviewed_manufacturing_evidence_sources",
                ),
                "payload_gap": _gap_reason_present(
                    reasons,
                    "fill reviewed_manufacturing_evidence_payload_json",
                ),
                "detail_gap": _gap_reason_present(
                    reasons,
                    "add details.* payload labels",
                ),
                "approval_gap": _gap_reason_present(
                    reasons,
                    "set approved review_status",
                ),
                "metadata_gap": _gap_reason_present(
                    reasons,
                    "fill reviewer and reviewed_at",
                ),
                "suggested_manufacturing_evidence_sources": ";".join(
                    suggested_sources
                ),
                "actual_evidence_sources": ";".join(actual_sources),
                "reviewer": _first_non_empty(row, REVIEWER_COLUMNS),
                "reviewed_at": _first_non_empty(row, REVIEWED_AT_COLUMNS),
                "review_notes": str(row.get("review_notes") or ""),
            }
        )

    batch_rows: List[Dict[str, str]] = []
    for batch_index, (label, bucket_rows) in enumerate(
        sorted(
            label_buckets.items(),
            key=lambda item: (-len(item[1]), item[0]),
        ),
        start=1,
    ):
        sorted_rows = sorted(
            bucket_rows,
            key=lambda row: (-int(row["gap_reason_count"]), row["row_id"]),
        )
        selected_rows = sorted_rows[: max(0, max_rows_per_label)]
        for batch_rank, row in enumerate(selected_rows, start=1):
            batch_rows.append(
                {
                    "review_batch": f"batch_{batch_index:03d}",
                    "batch_rank": str(batch_rank),
                    "label_gap_row_count": str(len(bucket_rows)),
                    **row,
                }
            )
    return batch_rows


def build_reviewer_template_rows(
    rows: Iterable[Dict[str, str]],
    *,
    require_reviewer_metadata: bool = False,
) -> List[Dict[str, str]]:
    template_rows: List[Dict[str, str]] = []
    for gap in build_review_gap_rows(
        rows,
        require_reviewer_metadata=require_reviewer_metadata,
    ):
        template_rows.append(
            {
                "row_id": gap["row_id"],
                "file_name": gap["file_name"],
                "label_cn": gap["label_cn"],
                "relative_path": gap["relative_path"],
                "source_dir": gap["source_dir"],
                "review_status": gap["review_status"] or "needs_human_review",
                "reviewer": gap["reviewer"],
                "reviewed_at": gap["reviewed_at"],
                "reviewed_manufacturing_evidence_sources": gap[
                    "reviewed_manufacturing_evidence_sources"
                ],
                "reviewed_manufacturing_evidence_payload_json": gap[
                    "reviewed_manufacturing_evidence_payload_json"
                ],
                "review_notes": gap["review_notes"],
                "suggested_manufacturing_evidence_sources": gap[
                    "suggested_manufacturing_evidence_sources"
                ],
                "suggested_manufacturing_evidence_payload_json": gap[
                    "suggested_manufacturing_evidence_payload_json"
                ],
                "gap_reasons": gap["gap_reasons"],
            }
        )
    return template_rows


def build_review_batch_template_rows(
    rows: Iterable[Dict[str, str]],
    *,
    max_rows_per_label: int = 5,
    require_reviewer_metadata: bool = False,
) -> List[Dict[str, str]]:
    materialized_rows = [dict(row) for row in rows]
    templates_by_row_id: Dict[str, List[Dict[str, str]]] = {}
    for template_row in build_reviewer_template_rows(
        materialized_rows,
        require_reviewer_metadata=require_reviewer_metadata,
    ):
        templates_by_row_id.setdefault(template_row["row_id"], []).append(template_row)

    batch_template_rows: List[Dict[str, str]] = []
    for batch_row in build_review_batch_rows(
        materialized_rows,
        max_rows_per_label=max_rows_per_label,
        require_reviewer_metadata=require_reviewer_metadata,
    ):
        matching_templates = templates_by_row_id.get(batch_row["row_id"]) or []
        if not matching_templates:
            continue
        template_row = matching_templates.pop(0)
        batch_template_rows.append(
            {
                "review_batch": batch_row["review_batch"],
                "batch_rank": batch_row["batch_rank"],
                "label_gap_row_count": batch_row["label_gap_row_count"],
                **template_row,
            }
        )
    return batch_template_rows


def build_reviewer_template_preflight_markdown(
    rows: Iterable[Dict[str, str]],
    summary: Dict[str, Any],
    *,
    max_blocking_rows: int = 30,
    require_reviewer_metadata: bool = False,
    base_rows: Optional[Iterable[Dict[str, str]]] = None,
) -> str:
    materialized_rows = [dict(row) for row in rows]
    base_indices: Optional[
        Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]
    ] = None
    base_manifest_duplicate_identifiers: List[str] = []
    base_manifest_duplicate_file_names: set[str] = set()
    if base_rows is not None:
        materialized_base_rows = [dict(row) for row in base_rows]
        base_indices = _manifest_indices(materialized_base_rows)
        base_manifest_duplicate_identifiers = _duplicate_row_identifiers(
            materialized_base_rows
        )
        base_manifest_duplicate_file_names = set(
            _duplicate_file_names(materialized_base_rows)
        )
    duplicate_counts: Dict[str, int] = {}
    for row in materialized_rows:
        identifier = _row_identifier(row)
        duplicate_counts[identifier] = duplicate_counts.get(identifier, 0) + 1

    lines = [
        "# Manufacturing Reviewer Template Preflight",
        "",
        "## Status",
        "",
        f"- status: `{summary.get('status', 'unknown')}`",
        f"- template_row_count: `{summary.get('template_row_count', len(materialized_rows))}`",
        f"- ready_template_row_count: `{summary.get('ready_template_row_count', 0)}`",
        f"- min_ready_rows: `{summary.get('min_ready_rows', 0)}`",
        (
            "- base_manifest_match_required: "
            f"`{str(summary.get('base_manifest_match_required', False)).lower()}`"
        ),
        f"- base_manifest_row_count: `{summary.get('base_manifest_row_count', 0)}`",
        (
            "- base_manifest_duplicate_identity_count: "
            f"`{summary.get('base_manifest_duplicate_identity_count', 0)}`"
        ),
        (
            "- require_reviewer_metadata: "
            f"`{str(summary.get('require_reviewer_metadata', False)).lower()}`"
        ),
        "",
        "## Issue Counts",
        "",
        "| Issue | Count |",
        "| --- | ---: |",
        f"| Approved rows | {summary.get('approved_template_row_count', 0)} |",
        f"| No review content | {summary.get('no_review_content_row_count', 0)} |",
        f"| Unapproved rows | {summary.get('unapproved_template_row_count', 0)} |",
        (
            "| Missing reviewer metadata | "
            f"{summary.get('reviewer_metadata_missing_row_count', 0)} |"
        ),
        (
            "| Unmatched manifest rows | "
            f"{summary.get('unmatched_template_row_count', 0)} |"
        ),
        (
            "| Ambiguous file-name fallback rows | "
            f"{summary.get('ambiguous_file_name_match_row_count', 0)} |"
        ),
        (
            "| Duplicate base manifest row identities | "
            f"{summary.get('base_manifest_duplicate_identity_count', 0)} |"
        ),
        f"| Missing source labels | {summary.get('source_missing_row_count', 0)} |",
        f"| Missing payload labels | {summary.get('payload_missing_row_count', 0)} |",
        (
            "| Missing detail payload labels | "
            f"{summary.get('payload_detail_missing_row_count', 0)} |"
        ),
        f"| Duplicate row identities | {summary.get('duplicate_template_row_count', 0)} |",
        "",
        "## Blocking Reasons",
        "",
    ]

    blocking_reasons = list(summary.get("blocking_reasons") or [])
    if blocking_reasons:
        lines.extend(f"- `{_markdown_cell(reason)}`" for reason in blocking_reasons)
    else:
        lines.append("- None")

    duplicate_identifiers = list(summary.get("duplicate_identifiers") or [])
    if duplicate_identifiers:
        lines.extend(["", "## Duplicate Row IDs", ""])
        lines.extend(f"- `{_markdown_cell(identifier)}`" for identifier in duplicate_identifiers)

    base_duplicate_identifiers = list(
        summary.get("base_manifest_duplicate_identifiers") or []
    )
    if base_duplicate_identifiers:
        lines.extend(["", "## Duplicate Base Manifest Row IDs", ""])
        lines.extend(
            f"- `{_markdown_cell(identifier)}`"
            for identifier in base_duplicate_identifiers
        )

    blocking_rows: List[Tuple[str, str, str, List[str]]] = []
    seen_identifiers: set[str] = set()
    for row in materialized_rows:
        identifier = _row_identifier(row)
        duplicate = identifier in seen_identifiers or duplicate_counts.get(identifier, 0) > 1
        seen_identifiers.add(identifier)
        matched_manifest_row: Optional[bool] = None
        ambiguous_file_name_match = False
        if base_indices is not None:
            ambiguous_file_name_match = (
                not base_manifest_duplicate_identifiers
                and _ambiguous_file_name_match(
                    row,
                    duplicate_file_names=base_manifest_duplicate_file_names,
                    by_relative_path=base_indices[0],
                )
            )
            if ambiguous_file_name_match:
                matched_manifest_row = False
            else:
                matched_manifest_row = (
                    _matching_base_row(row, base_indices[0], base_indices[1])
                    is not None
                )
        reasons = _reviewer_template_preflight_reasons(
            row,
            duplicate=duplicate,
            matched_manifest_row=matched_manifest_row,
            ambiguous_file_name_match=ambiguous_file_name_match,
            require_reviewer_metadata=require_reviewer_metadata,
        )
        if reasons:
            blocking_rows.append(
                (
                    identifier,
                    _row_label(row) or "<unknown>",
                    str(row.get("review_status") or "").strip() or "<missing>",
                    reasons,
                )
            )

    lines.extend(
        [
            "",
            "## Next Rows To Fix",
            "",
            "| Row | Label | Review status | Fixes |",
            "| --- | --- | --- | --- |",
        ]
    )
    for identifier, label, status, reasons in blocking_rows[: max(0, max_blocking_rows)]:
        lines.append(
            "| "
            f"{_markdown_cell(identifier)} | {_markdown_cell(label)} | "
            f"{_markdown_cell(status)} | {_markdown_cell('; '.join(reasons))} |"
        )
    if not blocking_rows:
        lines.append("| None | - | - | - |")
    if len(blocking_rows) > max_blocking_rows:
        lines.extend(
            [
                "",
                f"_Additional blocking rows omitted: {len(blocking_rows) - max_blocking_rows}_",
            ]
        )

    return "\n".join(lines) + "\n"


def build_review_assignment_markdown(
    rows: Iterable[Dict[str, str]],
    summary: Dict[str, Any],
    *,
    max_rows_per_label: int = 5,
    require_reviewer_metadata: bool = False,
) -> str:
    gap_rows = build_review_gap_rows(
        rows,
        require_reviewer_metadata=require_reviewer_metadata,
    )
    min_reviewed_samples = int(summary.get("min_reviewed_samples") or 0)
    source_count = int(summary.get("source_reviewed_sample_count") or 0)
    payload_count = int(summary.get("payload_reviewed_sample_count") or 0)
    detail_count = int(summary.get("payload_detail_reviewed_sample_count") or 0)
    buckets: Dict[str, Dict[str, Any]] = {}
    for gap in gap_rows:
        label = gap.get("label_cn") or "<unknown>"
        bucket = buckets.setdefault(
            label,
            {
                "rows": [],
                "source": 0,
                "payload": 0,
                "detail": 0,
                "approval": 0,
                "metadata": 0,
            },
        )
        bucket["rows"].append(gap)
        reasons = str(gap.get("gap_reasons") or "")
        if "fill reviewed_manufacturing_evidence_sources" in reasons:
            bucket["source"] += 1
        if "fill reviewed_manufacturing_evidence_payload_json" in reasons:
            bucket["payload"] += 1
        if "add details.* payload labels" in reasons:
            bucket["detail"] += 1
        if "set approved review_status" in reasons:
            bucket["approval"] += 1
        if "fill reviewer and reviewed_at" in reasons:
            bucket["metadata"] += 1

    lines = [
        "# Manufacturing Review Assignment Plan",
        "",
        "## Status",
        "",
        f"- validation_status: `{summary.get('status', 'unknown')}`",
        f"- gap_row_count: `{len(gap_rows)}`",
        f"- min_reviewed_samples: `{min_reviewed_samples}`",
        (
            "- require_reviewer_metadata: "
            f"`{str(summary.get('require_reviewer_metadata', False)).lower()}`"
        ),
        "",
        "## Remaining Release Labels",
        "",
        "| Evidence class | Reviewed | Required | Remaining |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| Source labels | {source_count} | {min_reviewed_samples} | "
            f"{max(0, min_reviewed_samples - source_count)} |"
        ),
        (
            f"| Payload labels | {payload_count} | {min_reviewed_samples} | "
            f"{max(0, min_reviewed_samples - payload_count)} |"
        ),
        (
            f"| Detail labels | {detail_count} | {min_reviewed_samples} | "
            f"{max(0, min_reviewed_samples - detail_count)} |"
        ),
        "",
        "## Assignment Buckets",
        "",
        "| Label | Gap rows | Source | Payload | Detail | Approval | Metadata |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    if buckets:
        for label, bucket in sorted(
            buckets.items(),
            key=lambda item: (-len(item[1]["rows"]), item[0]),
        ):
            lines.append(
                "| "
                f"{_markdown_cell(label)} | {len(bucket['rows'])} | "
                f"{bucket['source']} | {bucket['payload']} | {bucket['detail']} | "
                f"{bucket['approval']} | {bucket['metadata']} |"
            )
    else:
        lines.append("| None | 0 | 0 | 0 | 0 | 0 | 0 |")

    lines.extend(["", "## Suggested Review Batches", ""])
    if not buckets:
        lines.append("- None")
    for label, bucket in sorted(
        buckets.items(),
        key=lambda item: (-len(item[1]["rows"]), item[0]),
    ):
        rows_for_label = bucket["rows"][: max(0, max_rows_per_label)]
        lines.extend(
            [
                f"### {_markdown_cell(label)}",
                "",
                "| Row | Status | Gaps | Suggested sources |",
                "| --- | --- | --- | --- |",
            ]
        )
        for gap in rows_for_label:
            lines.append(
                "| "
                f"{_markdown_cell(gap.get('row_id'))} | "
                f"{_markdown_cell(gap.get('review_status') or '<missing>')} | "
                f"{_markdown_cell(gap.get('gap_reasons'))} | "
                f"{_markdown_cell(gap.get('suggested_manufacturing_evidence_sources'))} |"
            )
        if len(bucket["rows"]) > max_rows_per_label:
            lines.extend(
                [
                    "",
                    (
                        f"_Additional rows omitted for {_markdown_cell(label)}: "
                        f"{len(bucket['rows']) - max_rows_per_label}_"
                    ),
                ]
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_reviewer_template_preflight_gap_rows(
    rows: Iterable[Dict[str, str]],
    *,
    require_reviewer_metadata: bool = False,
    base_rows: Optional[Iterable[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    materialized_rows = [dict(row) for row in rows]
    base_indices: Optional[
        Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]
    ] = None
    base_manifest_duplicate_identifiers: List[str] = []
    base_manifest_duplicate_file_names: set[str] = set()
    if base_rows is not None:
        materialized_base_rows = [dict(row) for row in base_rows]
        base_indices = _manifest_indices(materialized_base_rows)
        base_manifest_duplicate_identifiers = _duplicate_row_identifiers(
            materialized_base_rows
        )
        base_manifest_duplicate_file_names = set(
            _duplicate_file_names(materialized_base_rows)
        )
    duplicate_counts: Dict[str, int] = {}
    for row in materialized_rows:
        identifier = _row_identifier(row)
        duplicate_counts[identifier] = duplicate_counts.get(identifier, 0) + 1

    gap_rows: List[Dict[str, str]] = []
    seen_identifiers: set[str] = set()
    for row in materialized_rows:
        identifier = _row_identifier(row)
        duplicate = (
            identifier in seen_identifiers or duplicate_counts.get(identifier, 0) > 1
        )
        seen_identifiers.add(identifier)
        matched_manifest_row: Optional[bool] = None
        ambiguous_file_name_match = False
        if base_indices is not None:
            ambiguous_file_name_match = (
                not base_manifest_duplicate_identifiers
                and _ambiguous_file_name_match(
                    row,
                    duplicate_file_names=base_manifest_duplicate_file_names,
                    by_relative_path=base_indices[0],
                )
            )
            if ambiguous_file_name_match:
                matched_manifest_row = False
            else:
                matched_manifest_row = (
                    _matching_base_row(row, base_indices[0], base_indices[1])
                    is not None
                )
        reasons = _reviewer_template_preflight_reasons(
            row,
            duplicate=duplicate,
            matched_manifest_row=matched_manifest_row,
            ambiguous_file_name_match=ambiguous_file_name_match,
            require_reviewer_metadata=require_reviewer_metadata,
        )
        if not reasons:
            continue

        _sources, source_reviewed = _extract_expected_manufacturing_sources(row)
        payloads, payload_reviewed = _extract_expected_manufacturing_payloads(row)
        detail_ready = payload_reviewed and _has_detail_payload(payloads)
        gap_rows.append(
            {
                "row_id": identifier,
                "file_name": _row_file_name(row),
                "label_cn": _row_label(row),
                "relative_path": str(row.get("relative_path") or "").strip(),
                "source_dir": str(row.get("source_dir") or "").strip(),
                "review_status": str(row.get("review_status") or "").strip(),
                "reviewer": _first_non_empty(row, REVIEWER_COLUMNS),
                "reviewed_at": _first_non_empty(row, REVIEWED_AT_COLUMNS),
                "preflight_reasons": "; ".join(reasons),
                "preflight_reason_count": str(len(reasons)),
                "duplicate_row": str(duplicate).lower(),
                "matched_manifest_row": (
                    "not_checked"
                    if matched_manifest_row is None
                    else str(matched_manifest_row).lower()
                ),
                "ambiguous_file_name_match": (
                    "not_checked"
                    if matched_manifest_row is None
                    else str(ambiguous_file_name_match).lower()
                ),
                "source_ready": str(source_reviewed).lower(),
                "payload_ready": str(payload_reviewed).lower(),
                "detail_ready": str(detail_ready).lower(),
                "reviewed_manufacturing_evidence_sources": str(
                    row.get("reviewed_manufacturing_evidence_sources") or ""
                ),
                "reviewed_manufacturing_evidence_payload_json": str(
                    row.get("reviewed_manufacturing_evidence_payload_json") or ""
                ),
                "review_notes": str(row.get("review_notes") or ""),
                "suggested_manufacturing_evidence_sources": str(
                    row.get("suggested_manufacturing_evidence_sources") or ""
                ),
                "suggested_manufacturing_evidence_payload_json": str(
                    row.get("suggested_manufacturing_evidence_payload_json") or ""
                ),
            }
        )
    return gap_rows


def build_reviewer_template_apply_audit_rows(
    manifest_rows: Iterable[Dict[str, str]],
    template_rows: Iterable[Dict[str, str]],
    *,
    require_reviewer_metadata: bool = False,
) -> List[Dict[str, str]]:
    materialized_manifest_rows = [dict(row) for row in manifest_rows]
    base_manifest_duplicate_identifiers = _duplicate_row_identifiers(
        materialized_manifest_rows
    )
    base_manifest_duplicate_file_names = set(
        _duplicate_file_names(materialized_manifest_rows)
    )
    by_relative_path, by_file_name = _manifest_indices(materialized_manifest_rows)
    audit_rows: List[Dict[str, str]] = []

    for template_row in template_rows:
        identifier = _row_identifier(template_row)
        reasons: List[str] = []
        matched_manifest_row = False
        apply_status = "applied"

        if not _row_has_review_content(template_row):
            apply_status = "skipped_no_review_content"
            reasons.append("fill reviewed source or payload fields")
        elif not _review_status_approved(template_row):
            apply_status = "skipped_unapproved_template"
            reasons.append("set approved review_status")
        elif (
            require_reviewer_metadata
            and not _reviewer_metadata_present(template_row)
        ):
            apply_status = "skipped_missing_metadata"
            reasons.append("fill reviewer and reviewed_at")
        elif base_manifest_duplicate_identifiers:
            apply_status = "blocked_duplicate_base_manifest"
            reasons.append("deduplicate base review manifest")
        elif _ambiguous_file_name_match(
            template_row,
            duplicate_file_names=base_manifest_duplicate_file_names,
            by_relative_path=by_relative_path,
        ):
            apply_status = "ambiguous_file_name_match"
            reasons.append("add relative_path to disambiguate duplicate file_name")
        else:
            manifest_row = _matching_base_row(
                template_row,
                by_relative_path,
                by_file_name,
            )
            matched_manifest_row = manifest_row is not None
            if manifest_row is None:
                apply_status = "unmatched_template_row"
                reasons.append("match row_id to review manifest")
            elif not _review_updates(template_row):
                apply_status = "skipped_empty_updates"
                reasons.append("fill at least one editable review field")

        _sources, source_reviewed = _extract_expected_manufacturing_sources(
            template_row
        )
        payloads, payload_reviewed = _extract_expected_manufacturing_payloads(
            template_row
        )
        detail_ready = payload_reviewed and _has_detail_payload(payloads)
        audit_rows.append(
            {
                "row_id": identifier,
                "file_name": _row_file_name(template_row),
                "label_cn": _row_label(template_row),
                "relative_path": str(template_row.get("relative_path") or "").strip(),
                "source_dir": str(template_row.get("source_dir") or "").strip(),
                "apply_status": apply_status,
                "apply_reasons": "; ".join(reasons),
                "matched_manifest_row": str(matched_manifest_row).lower(),
                "review_status": str(template_row.get("review_status") or "").strip(),
                "reviewer": _first_non_empty(template_row, REVIEWER_COLUMNS),
                "reviewed_at": _first_non_empty(template_row, REVIEWED_AT_COLUMNS),
                "source_ready": str(source_reviewed).lower(),
                "payload_ready": str(payload_reviewed).lower(),
                "detail_ready": str(detail_ready).lower(),
                "reviewed_manufacturing_evidence_sources": str(
                    template_row.get("reviewed_manufacturing_evidence_sources") or ""
                ),
                "reviewed_manufacturing_evidence_payload_json": str(
                    template_row.get("reviewed_manufacturing_evidence_payload_json")
                    or ""
                ),
                "review_notes": str(template_row.get("review_notes") or ""),
            }
        )
    return audit_rows


def build_review_handoff_markdown(
    summary: Dict[str, Any],
    *,
    manifest_path: str = "",
    summary_json_path: str = "",
    progress_md_path: str = "",
    gap_csv_path: str = "",
    context_csv_path: str = "",
    batch_csv_path: str = "",
    batch_template_csv_path: str = "",
    assignment_md_path: str = "",
    reviewer_template_csv_path: str = "",
    reviewer_template_preflight_md_path: str = "",
    reviewer_template_preflight_gap_csv_path: str = "",
    reviewer_template_preflight_min_ready_rows: Optional[int] = None,
) -> str:
    min_reviewed_samples = int(summary.get("min_reviewed_samples") or 0)
    preflight_min_ready_rows = (
        int(reviewer_template_preflight_min_ready_rows)
        if reviewer_template_preflight_min_ready_rows is not None
        else min_reviewed_samples
    )
    source_count = int(summary.get("source_reviewed_sample_count") or 0)
    payload_count = int(summary.get("payload_reviewed_sample_count") or 0)
    detail_count = int(summary.get("payload_detail_reviewed_sample_count") or 0)
    require_metadata = bool(summary.get("require_reviewer_metadata", False))
    template_path = (
        batch_template_csv_path
        or reviewer_template_csv_path
        or "<filled-reviewer-template.csv>"
    )
    preflight_md_path = reviewer_template_preflight_md_path or "<preflight-report.md>"
    preflight_gap_csv_path = (
        reviewer_template_preflight_gap_csv_path or "<preflight-gaps.csv>"
    )
    preflight_summary_path = "<preflight-summary.json>"
    applied_manifest_path = "<applied-review-manifest.csv>"
    apply_summary_path = "<apply-summary.json>"

    command_parts = [
        "python3 scripts/build_manufacturing_review_manifest.py",
        f"--validate-reviewer-template {template_path}",
        f"--summary-json {preflight_summary_path}",
        f"--reviewer-template-preflight-md {preflight_md_path}",
        f"--reviewer-template-preflight-gap-csv {preflight_gap_csv_path}",
        f"--min-reviewed-samples {preflight_min_ready_rows}",
    ]
    if manifest_path:
        command_parts.append(f"--base-manifest {manifest_path}")
    apply_parts = [
        "python3 scripts/build_manufacturing_review_manifest.py",
        f"--apply-reviewer-template {template_path}",
        f"--base-manifest {manifest_path or '<review-manifest.csv>'}",
        f"--output-csv {applied_manifest_path}",
        f"--summary-json {apply_summary_path}",
        f"--min-reviewed-samples {min_reviewed_samples}",
    ]
    if require_metadata:
        command_parts.append("--require-reviewer-metadata")
        apply_parts.append("--require-reviewer-metadata")
    command_separator = " \\\n  "

    artifact_rows = [
        ("Review manifest", manifest_path),
        ("Validation summary", summary_json_path),
        ("Progress report", progress_md_path),
        ("Gap CSV", gap_csv_path),
        ("Context CSV", context_csv_path),
        ("Batch CSV", batch_csv_path),
        ("Batch reviewer template", batch_template_csv_path),
        ("Assignment plan", assignment_md_path),
        ("Reviewer fill template", reviewer_template_csv_path),
        ("Preflight report", reviewer_template_preflight_md_path),
        ("Preflight gap CSV", reviewer_template_preflight_gap_csv_path),
    ]

    lines = [
        "# Manufacturing Review Handoff",
        "",
        "## Status",
        "",
        f"- validation_status: `{summary.get('status', 'unknown')}`",
        f"- row_count: `{summary.get('row_count', 0)}`",
        f"- min_reviewed_samples: `{min_reviewed_samples}`",
        f"- require_reviewer_metadata: `{str(require_metadata).lower()}`",
        "",
        "## Release Label Closeout",
        "",
        "| Evidence class | Reviewed | Required | Remaining | Ready |",
        "| --- | ---: | ---: | ---: | --- |",
        (
            f"| Source labels | {source_count} | {min_reviewed_samples} | "
            f"{max(0, min_reviewed_samples - source_count)} | "
            f"{str(bool(summary.get('source_reviewed_ready'))).lower()} |"
        ),
        (
            f"| Payload labels | {payload_count} | {min_reviewed_samples} | "
            f"{max(0, min_reviewed_samples - payload_count)} | "
            f"{str(bool(summary.get('payload_reviewed_ready'))).lower()} |"
        ),
        (
            f"| Detail labels | {detail_count} | {min_reviewed_samples} | "
            f"{max(0, min_reviewed_samples - detail_count)} | "
            f"{str(bool(summary.get('payload_detail_reviewed_ready'))).lower()} |"
        ),
        "",
        "## Artifact Map",
        "",
        "| Artifact | Path |",
        "| --- | --- |",
    ]

    for label, path in artifact_rows:
        lines.append(
            "| "
            f"{_markdown_cell(label)} | {_markdown_cell(path or '<not configured>')} |"
        )

    lines.extend(
        [
            "",
            "## Reviewer Workflow",
            "",
            (
                "1. Open the assignment plan, gap CSV, context CSV, and batch CSV "
                "to choose review rows."
            ),
            (
                "2. Fill the batch reviewer template or full reviewer template "
                "with domain-approved labels only."
            ),
            "3. Do not copy suggestions into reviewed fields without human review.",
            (
                "4. Set an approved review status only after source, payload, "
                "and detail labels are checked."
            ),
            "5. Run template preflight before applying the filled template.",
            "",
            "```bash",
            command_separator.join(command_parts),
            "```",
            "",
            "6. Apply only a preflight-ready template back into the review manifest.",
            "",
            "```bash",
            command_separator.join(apply_parts),
            "```",
            "",
            "## Blocking Reasons",
            "",
        ]
    )

    blocking_reasons = list(summary.get("blocking_reasons") or [])
    if blocking_reasons:
        lines.extend(f"- `{_markdown_cell(reason)}`" for reason in blocking_reasons)
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def _write_progress_markdown(
    path: Optional[Path],
    rows: Iterable[Dict[str, str]],
    summary: Dict[str, Any],
    *,
    max_gap_rows: int,
    require_reviewer_metadata: bool,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_review_progress_markdown(
            rows,
            summary,
            max_gap_rows=max_gap_rows,
            require_reviewer_metadata=require_reviewer_metadata,
        ),
        encoding="utf-8",
    )


def _write_assignment_markdown(
    path: Optional[Path],
    rows: Iterable[Dict[str, str]],
    summary: Dict[str, Any],
    *,
    max_rows_per_label: int,
    require_reviewer_metadata: bool,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_review_assignment_markdown(
            rows,
            summary,
            max_rows_per_label=max_rows_per_label,
            require_reviewer_metadata=require_reviewer_metadata,
        ),
        encoding="utf-8",
    )


def _write_gap_csv(
    path: Optional[Path],
    rows: Iterable[Dict[str, str]],
    *,
    require_reviewer_metadata: bool,
) -> None:
    if path is None:
        return
    _write_csv(
        path,
        build_review_gap_rows(
            rows,
            require_reviewer_metadata=require_reviewer_metadata,
        ),
        REVIEW_GAP_COLUMNS,
    )


def _write_review_context_csv(
    path: Optional[Path],
    rows: Iterable[Dict[str, str]],
    *,
    require_reviewer_metadata: bool,
) -> None:
    if path is None:
        return
    _write_csv(
        path,
        build_review_context_rows(
            rows,
            require_reviewer_metadata=require_reviewer_metadata,
        ),
        REVIEW_CONTEXT_COLUMNS,
    )


def _write_review_batch_csv(
    path: Optional[Path],
    rows: Iterable[Dict[str, str]],
    *,
    max_rows_per_label: int,
    require_reviewer_metadata: bool,
) -> None:
    if path is None:
        return
    _write_csv(
        path,
        build_review_batch_rows(
            rows,
            max_rows_per_label=max_rows_per_label,
            require_reviewer_metadata=require_reviewer_metadata,
        ),
        REVIEW_BATCH_COLUMNS,
    )


def _write_review_batch_template_csv(
    path: Optional[Path],
    rows: Iterable[Dict[str, str]],
    *,
    max_rows_per_label: int,
    require_reviewer_metadata: bool,
) -> None:
    if path is None:
        return
    _write_csv(
        path,
        build_review_batch_template_rows(
            rows,
            max_rows_per_label=max_rows_per_label,
            require_reviewer_metadata=require_reviewer_metadata,
        ),
        REVIEW_BATCH_TEMPLATE_COLUMNS,
    )


def _write_reviewer_template_csv(
    path: Optional[Path],
    rows: Iterable[Dict[str, str]],
    *,
    require_reviewer_metadata: bool,
) -> None:
    if path is None:
        return
    _write_csv(
        path,
        build_reviewer_template_rows(
            rows,
            require_reviewer_metadata=require_reviewer_metadata,
        ),
        REVIEWER_TEMPLATE_COLUMNS,
    )


def _write_reviewer_template_preflight_markdown(
    path: Optional[Path],
    rows: Iterable[Dict[str, str]],
    summary: Dict[str, Any],
    *,
    max_blocking_rows: int,
    require_reviewer_metadata: bool,
    base_rows: Optional[Iterable[Dict[str, str]]] = None,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_reviewer_template_preflight_markdown(
            rows,
            summary,
            max_blocking_rows=max_blocking_rows,
            require_reviewer_metadata=require_reviewer_metadata,
            base_rows=base_rows,
        ),
        encoding="utf-8",
    )


def _write_reviewer_template_preflight_gap_csv(
    path: Optional[Path],
    rows: Iterable[Dict[str, str]],
    *,
    require_reviewer_metadata: bool,
    base_rows: Optional[Iterable[Dict[str, str]]] = None,
) -> None:
    if path is None:
        return
    _write_csv(
        path,
        build_reviewer_template_preflight_gap_rows(
            rows,
            require_reviewer_metadata=require_reviewer_metadata,
            base_rows=base_rows,
        ),
        REVIEWER_TEMPLATE_PREFLIGHT_GAP_COLUMNS,
    )


def _write_reviewer_template_apply_audit_csv(
    path: Optional[Path],
    manifest_rows: Iterable[Dict[str, str]],
    template_rows: Iterable[Dict[str, str]],
    *,
    require_reviewer_metadata: bool,
) -> None:
    if path is None:
        return
    _write_csv(
        path,
        build_reviewer_template_apply_audit_rows(
            manifest_rows,
            template_rows,
            require_reviewer_metadata=require_reviewer_metadata,
        ),
        REVIEWER_TEMPLATE_APPLY_AUDIT_COLUMNS,
    )


def _write_review_manifest_merge_audit_csv(
    path: Optional[Path],
    base_rows: Iterable[Dict[str, str]],
    review_rows: Iterable[Dict[str, str]],
    *,
    require_reviewer_metadata: bool,
) -> None:
    if path is None:
        return
    _write_csv(
        path,
        build_review_manifest_merge_audit_rows(
            base_rows,
            review_rows,
            require_reviewer_metadata=require_reviewer_metadata,
        ),
        REVIEW_MANIFEST_MERGE_AUDIT_COLUMNS,
    )


def _write_handoff_markdown(
    path: Optional[Path],
    summary: Dict[str, Any],
    *,
    manifest_path: str,
    summary_json_path: str,
    progress_md_path: str,
    gap_csv_path: str,
    context_csv_path: str,
    batch_csv_path: str,
    batch_template_csv_path: str,
    assignment_md_path: str,
    reviewer_template_csv_path: str,
    reviewer_template_preflight_md_path: str,
    reviewer_template_preflight_gap_csv_path: str,
    reviewer_template_preflight_min_ready_rows: Optional[int],
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_review_handoff_markdown(
            summary,
            manifest_path=manifest_path,
            summary_json_path=summary_json_path,
            progress_md_path=progress_md_path,
            gap_csv_path=gap_csv_path,
            context_csv_path=context_csv_path,
            batch_csv_path=batch_csv_path,
            batch_template_csv_path=batch_template_csv_path,
            assignment_md_path=assignment_md_path,
            reviewer_template_csv_path=reviewer_template_csv_path,
            reviewer_template_preflight_md_path=reviewer_template_preflight_md_path,
            reviewer_template_preflight_gap_csv_path=(
                reviewer_template_preflight_gap_csv_path
            ),
            reviewer_template_preflight_min_ready_rows=(
                reviewer_template_preflight_min_ready_rows
            ),
        ),
        encoding="utf-8",
    )


def _write_summary(path: Optional[Path], summary: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _output_columns(
    base_columns: Iterable[str],
    rows: Iterable[Dict[str, str]],
    extra_columns: Iterable[str],
) -> List[str]:
    columns = list(dict.fromkeys(base_columns))
    if not columns:
        for row in rows:
            for column in row:
                if column not in columns:
                    columns.append(column)
    for column in extra_columns:
        if column not in columns:
            columns.append(column)
    return columns


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a manufacturing evidence review manifest from benchmark results "
            "or validate and merge reviewed manufacturing labels."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-results-csv", help="Benchmark results CSV to template.")
    mode.add_argument("--validate-manifest", help="Reviewed manifest CSV to validate.")
    mode.add_argument(
        "--merge-approved-review-manifest",
        help="Reviewed manifest CSV with approved labels to merge into a base manifest.",
    )
    mode.add_argument(
        "--apply-reviewer-template",
        help="Reviewer template CSV with approved rows to apply into a review manifest.",
    )
    mode.add_argument(
        "--validate-reviewer-template",
        help="Reviewer template CSV to preflight before applying.",
    )
    parser.add_argument(
        "--base-manifest",
        help=(
            "Base CSV used with --merge-approved-review-manifest or "
            "--apply-reviewer-template."
        ),
    )
    parser.add_argument("--output-csv", help="Review manifest CSV output path.")
    parser.add_argument("--summary-json", help="Optional summary JSON output path.")
    parser.add_argument("--progress-md", help="Optional review progress Markdown path.")
    parser.add_argument("--gap-csv", help="Optional review gap CSV path.")
    parser.add_argument("--review-context-csv", help="Optional review context CSV path.")
    parser.add_argument("--review-batch-csv", help="Optional review batch CSV path.")
    parser.add_argument(
        "--review-batch-template-csv",
        help="Optional label-balanced reviewer batch template CSV path.",
    )
    parser.add_argument("--assignment-md", help="Optional review assignment Markdown path.")
    parser.add_argument("--reviewer-template-csv", help="Optional reviewer fill template CSV path.")
    parser.add_argument("--handoff-md", help="Optional review handoff Markdown path.")
    parser.add_argument(
        "--reviewer-template-preflight-md",
        help="Optional reviewer template preflight Markdown path.",
    )
    parser.add_argument(
        "--reviewer-template-preflight-gap-csv",
        help="Optional reviewer template preflight gap CSV path.",
    )
    parser.add_argument(
        "--reviewer-template-preflight-min-ready-rows",
        type=int,
        help=(
            "Minimum ready rows to show in handoff preflight commands. "
            "Defaults to --min-reviewed-samples."
        ),
    )
    parser.add_argument(
        "--reviewer-template-apply-audit-csv",
        help="Optional reviewer template apply audit CSV path.",
    )
    parser.add_argument(
        "--review-manifest-merge-audit-csv",
        help="Optional reviewed manifest merge audit CSV path.",
    )
    parser.add_argument("--max-progress-rows", type=int, default=30)
    parser.add_argument("--max-assignment-rows-per-label", type=int, default=5)
    parser.add_argument("--max-batch-rows-per-label", type=int, default=5)
    parser.add_argument("--max-preflight-rows", type=int, default=30)
    parser.add_argument("--min-reviewed-samples", type=int, default=30)
    parser.add_argument(
        "--prefill-reviewed-from-suggestions",
        action="store_true",
        help="Copy suggestions into reviewed columns for controlled bootstrap runs.",
    )
    parser.add_argument(
        "--fail-under-minimum",
        action="store_true",
        help="Return non-zero when reviewed sample counts are below thresholds.",
    )
    parser.add_argument(
        "--require-reviewer-metadata",
        action="store_true",
        help="Require reviewer and reviewed_at metadata on approved reviewed rows.",
    )
    args = parser.parse_args(argv)

    if args.from_results_csv:
        input_path = Path(args.from_results_csv)
        output_path = Path(args.output_csv) if args.output_csv else None
        if output_path is None:
            raise SystemExit("--output-csv is required with --from-results-csv")
        rows = _read_csv(input_path)
        review_rows = build_review_rows(
            rows,
            prefill_reviewed_from_suggestions=bool(args.prefill_reviewed_from_suggestions),
        )
        _write_csv(output_path, review_rows, REVIEW_MANIFEST_COLUMNS)
        validation = validate_review_manifest_rows(
            review_rows,
            min_reviewed_samples=int(args.min_reviewed_samples),
            require_reviewer_metadata=bool(args.require_reviewer_metadata),
        )
        summary = {
            "mode": "build",
            "input_csv": str(input_path),
            "output_csv": str(output_path),
            "prefill_reviewed_from_suggestions": bool(
                args.prefill_reviewed_from_suggestions
            ),
            "suggested_sample_count": len(review_rows),
            **validation,
        }
        _write_summary(Path(args.summary_json) if args.summary_json else None, summary)
        _write_progress_markdown(
            Path(args.progress_md) if args.progress_md else None,
            review_rows,
            summary,
            max_gap_rows=int(args.max_progress_rows),
            require_reviewer_metadata=bool(args.require_reviewer_metadata),
        )
        _write_gap_csv(
            Path(args.gap_csv) if args.gap_csv else None,
            review_rows,
            require_reviewer_metadata=bool(args.require_reviewer_metadata),
        )
        _write_review_context_csv(
            Path(args.review_context_csv) if args.review_context_csv else None,
            review_rows,
            require_reviewer_metadata=bool(args.require_reviewer_metadata),
        )
        _write_review_batch_csv(
            Path(args.review_batch_csv) if args.review_batch_csv else None,
            review_rows,
            max_rows_per_label=int(args.max_batch_rows_per_label),
            require_reviewer_metadata=bool(args.require_reviewer_metadata),
        )
        _write_review_batch_template_csv(
            (
                Path(args.review_batch_template_csv)
                if args.review_batch_template_csv
                else None
            ),
            review_rows,
            max_rows_per_label=int(args.max_batch_rows_per_label),
            require_reviewer_metadata=bool(args.require_reviewer_metadata),
        )
        _write_reviewer_template_csv(
            Path(args.reviewer_template_csv) if args.reviewer_template_csv else None,
            review_rows,
            require_reviewer_metadata=bool(args.require_reviewer_metadata),
        )
        _write_assignment_markdown(
            Path(args.assignment_md) if args.assignment_md else None,
            review_rows,
            summary,
            max_rows_per_label=int(args.max_assignment_rows_per_label),
            require_reviewer_metadata=bool(args.require_reviewer_metadata),
        )
        _write_handoff_markdown(
            Path(args.handoff_md) if args.handoff_md else None,
            summary,
            manifest_path=str(output_path),
            summary_json_path=str(args.summary_json or ""),
            progress_md_path=str(args.progress_md or ""),
            gap_csv_path=str(args.gap_csv or ""),
            context_csv_path=str(args.review_context_csv or ""),
            batch_csv_path=str(args.review_batch_csv or ""),
            batch_template_csv_path=str(args.review_batch_template_csv or ""),
            assignment_md_path=str(args.assignment_md or ""),
            reviewer_template_csv_path=str(args.reviewer_template_csv or ""),
            reviewer_template_preflight_md_path=str(
                args.reviewer_template_preflight_md or ""
            ),
            reviewer_template_preflight_gap_csv_path=str(
                args.reviewer_template_preflight_gap_csv or ""
            ),
            reviewer_template_preflight_min_ready_rows=(
                args.reviewer_template_preflight_min_ready_rows
            ),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.fail_under_minimum and summary["status"] != "release_label_ready":
            return 1
        return 0

    if args.merge_approved_review_manifest:
        base_manifest_path = Path(args.base_manifest) if args.base_manifest else None
        review_manifest_path = Path(args.merge_approved_review_manifest)
        output_path = Path(args.output_csv) if args.output_csv else None
        if base_manifest_path is None:
            raise SystemExit(
                "--base-manifest is required with --merge-approved-review-manifest"
            )
        if output_path is None:
            raise SystemExit(
                "--output-csv is required with --merge-approved-review-manifest"
            )
        base_rows, base_columns = _read_csv_with_columns(base_manifest_path)
        review_rows = _read_csv(review_manifest_path)
        merged_rows, merge_summary = merge_approved_review_rows(
            base_rows,
            review_rows,
            require_reviewer_metadata=bool(args.require_reviewer_metadata),
        )
        _write_csv(
            output_path,
            merged_rows,
            _output_columns(base_columns, merged_rows, MERGED_REVIEW_COLUMNS),
        )
        summary = {
            "mode": "merge",
            "base_manifest": str(base_manifest_path),
            "review_manifest": str(review_manifest_path),
            "output_csv": str(output_path),
            **merge_summary,
        }
        _write_summary(Path(args.summary_json) if args.summary_json else None, summary)
        _write_review_manifest_merge_audit_csv(
            (
                Path(args.review_manifest_merge_audit_csv)
                if args.review_manifest_merge_audit_csv
                else None
            ),
            base_rows,
            review_rows,
            require_reviewer_metadata=bool(args.require_reviewer_metadata),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.fail_under_minimum and summary["status"] != "merged":
            return 1
        return 0

    if args.apply_reviewer_template:
        base_manifest_path = Path(args.base_manifest) if args.base_manifest else None
        template_path = Path(args.apply_reviewer_template)
        output_path = Path(args.output_csv) if args.output_csv else None
        if base_manifest_path is None:
            raise SystemExit("--base-manifest is required with --apply-reviewer-template")
        if output_path is None:
            raise SystemExit("--output-csv is required with --apply-reviewer-template")
        base_rows, base_columns = _read_csv_with_columns(base_manifest_path)
        template_rows = _read_csv(template_path)
        applied_rows, apply_summary = apply_reviewer_template_rows(
            base_rows,
            template_rows,
            require_reviewer_metadata=bool(args.require_reviewer_metadata),
        )
        post_apply_validation = validate_review_manifest_rows(
            applied_rows,
            min_reviewed_samples=int(args.min_reviewed_samples),
            require_reviewer_metadata=bool(args.require_reviewer_metadata),
        )
        _write_csv(
            output_path,
            applied_rows,
            _output_columns(base_columns, applied_rows, MERGED_REVIEW_COLUMNS),
        )
        summary = {
            "mode": "apply_reviewer_template",
            "base_manifest": str(base_manifest_path),
            "reviewer_template": str(template_path),
            "output_csv": str(output_path),
            **apply_summary,
            "post_apply_validation": post_apply_validation,
        }
        _write_summary(Path(args.summary_json) if args.summary_json else None, summary)
        _write_reviewer_template_apply_audit_csv(
            (
                Path(args.reviewer_template_apply_audit_csv)
                if args.reviewer_template_apply_audit_csv
                else None
            ),
            base_rows,
            template_rows,
            require_reviewer_metadata=bool(args.require_reviewer_metadata),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.fail_under_minimum:
            if post_apply_validation["status"] != "release_label_ready":
                return 1
        elif summary["status"] != "applied":
            return 1
        return 0

    if args.validate_reviewer_template:
        template_path = Path(args.validate_reviewer_template)
        template_rows = _read_csv(template_path)
        base_manifest_path = Path(args.base_manifest) if args.base_manifest else None
        base_rows = _read_csv(base_manifest_path) if base_manifest_path else None
        summary = {
            "mode": "validate_reviewer_template",
            "reviewer_template": str(template_path),
            "base_manifest": str(base_manifest_path or ""),
            **validate_reviewer_template_rows(
                template_rows,
                min_ready_rows=int(args.min_reviewed_samples),
                require_reviewer_metadata=bool(args.require_reviewer_metadata),
                base_rows=base_rows,
            ),
        }
        _write_summary(Path(args.summary_json) if args.summary_json else None, summary)
        _write_reviewer_template_preflight_markdown(
            (
                Path(args.reviewer_template_preflight_md)
                if args.reviewer_template_preflight_md
                else None
            ),
            template_rows,
            summary,
            max_blocking_rows=int(args.max_preflight_rows),
            require_reviewer_metadata=bool(args.require_reviewer_metadata),
            base_rows=base_rows,
        )
        _write_reviewer_template_preflight_gap_csv(
            (
                Path(args.reviewer_template_preflight_gap_csv)
                if args.reviewer_template_preflight_gap_csv
                else None
            ),
            template_rows,
            require_reviewer_metadata=bool(args.require_reviewer_metadata),
            base_rows=base_rows,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.fail_under_minimum and summary["status"] != "ready":
            return 1
        return 0

    manifest_path = Path(args.validate_manifest)
    rows = _read_csv(manifest_path)
    summary = {
        "mode": "validate",
        "manifest": str(manifest_path),
        **validate_review_manifest_rows(
            rows,
            min_reviewed_samples=int(args.min_reviewed_samples),
            require_reviewer_metadata=bool(args.require_reviewer_metadata),
        ),
    }
    _write_summary(Path(args.summary_json) if args.summary_json else None, summary)
    _write_progress_markdown(
        Path(args.progress_md) if args.progress_md else None,
        rows,
        summary,
        max_gap_rows=int(args.max_progress_rows),
        require_reviewer_metadata=bool(args.require_reviewer_metadata),
    )
    _write_gap_csv(
        Path(args.gap_csv) if args.gap_csv else None,
        rows,
        require_reviewer_metadata=bool(args.require_reviewer_metadata),
    )
    _write_review_context_csv(
        Path(args.review_context_csv) if args.review_context_csv else None,
        rows,
        require_reviewer_metadata=bool(args.require_reviewer_metadata),
    )
    _write_review_batch_csv(
        Path(args.review_batch_csv) if args.review_batch_csv else None,
        rows,
        max_rows_per_label=int(args.max_batch_rows_per_label),
        require_reviewer_metadata=bool(args.require_reviewer_metadata),
    )
    _write_review_batch_template_csv(
        (
            Path(args.review_batch_template_csv)
            if args.review_batch_template_csv
            else None
        ),
        rows,
        max_rows_per_label=int(args.max_batch_rows_per_label),
        require_reviewer_metadata=bool(args.require_reviewer_metadata),
    )
    _write_reviewer_template_csv(
        Path(args.reviewer_template_csv) if args.reviewer_template_csv else None,
        rows,
        require_reviewer_metadata=bool(args.require_reviewer_metadata),
    )
    _write_assignment_markdown(
        Path(args.assignment_md) if args.assignment_md else None,
        rows,
        summary,
        max_rows_per_label=int(args.max_assignment_rows_per_label),
        require_reviewer_metadata=bool(args.require_reviewer_metadata),
    )
    _write_handoff_markdown(
        Path(args.handoff_md) if args.handoff_md else None,
        summary,
        manifest_path=str(manifest_path),
        summary_json_path=str(args.summary_json or ""),
        progress_md_path=str(args.progress_md or ""),
        gap_csv_path=str(args.gap_csv or ""),
        context_csv_path=str(args.review_context_csv or ""),
        batch_csv_path=str(args.review_batch_csv or ""),
        batch_template_csv_path=str(args.review_batch_template_csv or ""),
        assignment_md_path=str(args.assignment_md or ""),
        reviewer_template_csv_path=str(args.reviewer_template_csv or ""),
        reviewer_template_preflight_md_path=str(
            args.reviewer_template_preflight_md or ""
        ),
        reviewer_template_preflight_gap_csv_path=str(
            args.reviewer_template_preflight_gap_csv or ""
        ),
        reviewer_template_preflight_min_ready_rows=(
            args.reviewer_template_preflight_min_ready_rows
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.fail_under_minimum and summary["status"] != "release_label_ready":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
