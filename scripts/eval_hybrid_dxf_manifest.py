#!/usr/bin/env python3
"""Evaluate DXF API / hybrid classification against a labeled manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MANUFACTURING_EVIDENCE_REQUIRED_SOURCES = (
    "dfm",
    "manufacturing_process",
    "manufacturing_cost",
    "manufacturing_decision",
)

MANUFACTURING_EVIDENCE_SOURCE_FLAGS = {
    "dfm": "manufacturing_evidence_has_dfm",
    "manufacturing_process": "manufacturing_evidence_has_process",
    "manufacturing_cost": "manufacturing_evidence_has_cost",
    "manufacturing_decision": "manufacturing_evidence_has_decision",
}

MANUFACTURING_EXPECTED_SOURCE_COLUMNS = (
    "expected_manufacturing_evidence_sources",
    "expected_manufacturing_sources",
    "reviewed_manufacturing_evidence_sources",
    "reviewed_manufacturing_sources",
)

MANUFACTURING_SOURCE_ALIASES = {
    "dfm": "dfm",
    "manufacturability": "dfm",
    "manufacturability_check": "dfm",
    "process": "manufacturing_process",
    "manufacturing_process": "manufacturing_process",
    "process_recommendation": "manufacturing_process",
    "cost": "manufacturing_cost",
    "manufacturing_cost": "manufacturing_cost",
    "cost_estimate": "manufacturing_cost",
    "decision": "manufacturing_decision",
    "manufacturing_decision": "manufacturing_decision",
    "manufacturing_summary": "manufacturing_decision",
}

MANUFACTURING_SOURCE_NONE_TOKENS = {
    "none",
    "no_evidence",
    "no_manufacturing_evidence",
    "empty",
    "n/a",
    "na",
}

MANUFACTURING_PAYLOAD_FIELDS = ("kind", "label", "status")
MANUFACTURING_PAYLOAD_DETAIL_PREFIX = "details."

MANUFACTURING_PAYLOAD_JSON_COLUMNS = (
    "expected_manufacturing_evidence_payload_json",
    "expected_manufacturing_payload_json",
    "reviewed_manufacturing_evidence_payload_json",
    "reviewed_manufacturing_payload_json",
)

MANUFACTURING_PAYLOAD_SOURCE_COLUMN_ALIASES = {
    "dfm": ("dfm", "manufacturing_dfm"),
    "manufacturing_process": ("process", "manufacturing_process"),
    "manufacturing_cost": ("cost", "manufacturing_cost"),
    "manufacturing_decision": ("decision", "manufacturing_decision"),
}


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
    expected_manufacturing_evidence_sources: Tuple[str, ...] = ()
    expected_manufacturing_evidence_payloads: Dict[str, Dict[str, str]] = field(
        default_factory=dict
    )
    manufacturing_evidence_reviewed: bool = False
    manufacturing_payload_reviewed: bool = False


def _normalize_manufacturing_source_token(value: Any) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        return ""
    normalized = cleaned.lower().replace("-", "_").replace(" ", "_")
    return MANUFACTURING_SOURCE_ALIASES.get(normalized, normalized)


def _parse_manufacturing_source_tokens(value: Any) -> Tuple[Tuple[str, ...], bool]:
    if value is None:
        return (), False
    if isinstance(value, list):
        raw_tokens = [str(item or "").strip() for item in value]
    else:
        text = str(value or "").strip()
        if not text:
            return (), False
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, list):
            raw_tokens = [str(item or "").strip() for item in decoded]
        else:
            raw_tokens = [
                token.strip()
                for token in re.split(r"[;,|]", text)
                if token.strip()
            ]

    reviewed = bool(raw_tokens)
    sources: List[str] = []
    for token in raw_tokens:
        normalized = _normalize_manufacturing_source_token(token)
        if not normalized or normalized in MANUFACTURING_SOURCE_NONE_TOKENS:
            continue
        sources.append(normalized)
    return tuple(dict.fromkeys(sources)), reviewed


def _extract_expected_manufacturing_sources(
    row: Dict[str, str]
) -> Tuple[Tuple[str, ...], bool]:
    for column in MANUFACTURING_EXPECTED_SOURCE_COLUMNS:
        if column not in row:
            continue
        sources, reviewed = _parse_manufacturing_source_tokens(row.get(column))
        if reviewed:
            return sources, True
    return (), False


def _clean_expected_payload_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in MANUFACTURING_SOURCE_NONE_TOKENS:
        return ""
    return text


def _flatten_expected_detail_fields(
    value: Any,
    *,
    prefix: str = MANUFACTURING_PAYLOAD_DETAIL_PREFIX,
) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    flattened: Dict[str, str] = {}
    for key, raw_value in value.items():
        cleaned_key = str(key or "").strip()
        if not cleaned_key:
            continue
        field_name = f"{prefix}{cleaned_key}"
        if isinstance(raw_value, dict):
            flattened.update(
                _flatten_expected_detail_fields(
                    raw_value,
                    prefix=f"{field_name}.",
                )
            )
            continue
        cleaned_value = _clean_expected_payload_value(raw_value)
        if cleaned_value:
            flattened[field_name] = cleaned_value
    return flattened


def _normalize_expected_payload_fields(value: Any) -> Dict[str, str]:
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


def _merge_expected_payload(
    payloads: Dict[str, Dict[str, str]],
    *,
    source: Any,
    value: Any,
) -> None:
    normalized_source = _normalize_manufacturing_source_token(source)
    if not normalized_source:
        return
    normalized_fields = _normalize_expected_payload_fields(value)
    if not normalized_fields:
        return
    payloads.setdefault(normalized_source, {}).update(normalized_fields)


def _extract_expected_manufacturing_payloads(
    row: Dict[str, str]
) -> Tuple[Dict[str, Dict[str, str]], bool]:
    payloads: Dict[str, Dict[str, str]] = {}

    for column in MANUFACTURING_PAYLOAD_JSON_COLUMNS:
        raw_value = str(row.get(column) or "").strip()
        if not raw_value:
            continue
        try:
            decoded = json.loads(raw_value)
        except json.JSONDecodeError:
            decoded = {}
        if not isinstance(decoded, dict):
            continue
        for source, value in decoded.items():
            _merge_expected_payload(payloads, source=source, value=value)

    for source, aliases in MANUFACTURING_PAYLOAD_SOURCE_COLUMN_ALIASES.items():
        for field_name in MANUFACTURING_PAYLOAD_FIELDS:
            for alias in aliases:
                column = f"expected_{alias}_{field_name}"
                cleaned = _clean_expected_payload_value(row.get(column))
                if cleaned:
                    payloads.setdefault(source, {})[field_name] = cleaned
        for alias in aliases:
            detail_prefix = f"expected_{alias}_detail_"
            for column, raw_value in row.items():
                if not column.startswith(detail_prefix):
                    continue
                detail_path = column[len(detail_prefix) :].strip().replace("__", ".")
                if not detail_path:
                    continue
                cleaned = _clean_expected_payload_value(raw_value)
                if cleaned:
                    payloads.setdefault(source, {})[
                        f"{MANUFACTURING_PAYLOAD_DETAIL_PREFIX}{detail_path}"
                    ] = cleaned

    return payloads, bool(payloads)


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
            expected_sources, evidence_reviewed = _extract_expected_manufacturing_sources(
                row
            )
            expected_payloads, payload_reviewed = _extract_expected_manufacturing_payloads(
                row
            )

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
                    expected_manufacturing_evidence_sources=expected_sources,
                    expected_manufacturing_evidence_payloads=expected_payloads,
                    manufacturing_evidence_reviewed=evidence_reviewed,
                    manufacturing_payload_reviewed=payload_reviewed,
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


def _json_list_cell(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _string_tokens(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    tokens: List[str] = []
    for item in value:
        token = str(item or "").strip()
        if token:
            tokens.append(token)
    return tokens


def _evidence_sources(evidence: Any) -> List[str]:
    if not isinstance(evidence, list):
        return []
    sources: List[str] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "").strip()
        if source:
            sources.append(source)
    return list(dict.fromkeys(sources))


def _evidence_items(value: Any, *, required_sources_only: bool) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    evidence: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "").strip()
        if required_sources_only and source not in MANUFACTURING_EVIDENCE_REQUIRED_SOURCES:
            continue
        evidence.append(dict(item))
    return evidence


def _manufacturing_evidence_items_from_results(
    results_payload: Dict[str, Any]
) -> List[Dict[str, Any]]:
    classification = results_payload.get("classification", {}) or {}
    if not isinstance(classification, dict):
        classification = {}
    decision_contract = classification.get("decision_contract")
    if not isinstance(decision_contract, dict):
        decision_contract = {}

    evidence: List[Dict[str, Any]] = []
    candidates = [
        (results_payload.get("manufacturing_evidence"), False),
        (classification.get("manufacturing_evidence"), False),
        (classification.get("evidence"), True),
        (decision_contract.get("evidence"), True),
    ]
    for candidate, required_sources_only in candidates:
        evidence = _evidence_items(
            candidate,
            required_sources_only=required_sources_only,
        )
        if evidence:
            break

    return evidence


def _collect_manufacturing_evidence_fields(
    results_payload: Dict[str, Any]
) -> Dict[str, Any]:
    evidence = _manufacturing_evidence_items_from_results(results_payload)
    sources = _evidence_sources(evidence)
    source_set = set(sources)
    payload: Dict[str, Any] = {
        "manufacturing_evidence": _json_list_cell(evidence),
        "manufacturing_evidence_count": len(evidence),
        "manufacturing_evidence_sources": ";".join(sources),
        "manufacturing_evidence_required_sources_present": all(
            source in source_set for source in MANUFACTURING_EVIDENCE_REQUIRED_SOURCES
        ),
    }
    for source, field_name in MANUFACTURING_EVIDENCE_SOURCE_FLAGS.items():
        payload[field_name] = source in source_set
    return payload


def _source_tokens_from_text(value: Any) -> Tuple[str, ...]:
    return _parse_manufacturing_source_tokens(value)[0]


def _manufacturing_correctness_fields(
    predicted_sources: Iterable[str],
    expected_sources: Iterable[str],
    *,
    reviewed: bool,
) -> Dict[str, Any]:
    predicted_set = set(predicted_sources)
    expected_set = set(expected_sources)
    true_positive = sorted(predicted_set & expected_set)
    false_positive = sorted(predicted_set - expected_set)
    false_negative = sorted(expected_set - predicted_set)
    precision_denominator = len(true_positive) + len(false_positive)
    recall_denominator = len(true_positive) + len(false_negative)
    precision = (
        len(true_positive) / precision_denominator
        if precision_denominator
        else 0.0
    )
    recall = len(true_positive) / recall_denominator if recall_denominator else 0.0
    f1 = (
        (2.0 * precision * recall / (precision + recall))
        if precision + recall
        else 0.0
    )
    return {
        "manufacturing_evidence_reviewed": bool(reviewed),
        "expected_manufacturing_evidence_sources": ";".join(sorted(expected_set)),
        "manufacturing_evidence_true_positive_sources": ";".join(true_positive),
        "manufacturing_evidence_false_positive_sources": ";".join(false_positive),
        "manufacturing_evidence_false_negative_sources": ";".join(false_negative),
        "manufacturing_evidence_source_exact_match": (
            bool(reviewed) and predicted_set == expected_set
        ),
        "manufacturing_evidence_source_precision": round(precision, 6),
        "manufacturing_evidence_source_recall": round(recall, 6),
        "manufacturing_evidence_source_f1": round(f1, 6),
    }


def _normalize_payload_compare_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().casefold()


def _payload_value_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def _get_payload_field_value(item: Dict[str, Any], field_name: str) -> str:
    if field_name in MANUFACTURING_PAYLOAD_FIELDS:
        return _payload_value_text(item.get(field_name))
    if not field_name.startswith(MANUFACTURING_PAYLOAD_DETAIL_PREFIX):
        return ""
    current: Any = item.get("details") if isinstance(item, dict) else {}
    for part in field_name[len(MANUFACTURING_PAYLOAD_DETAIL_PREFIX) :].split("."):
        if not isinstance(current, dict):
            return ""
        current = current.get(part)
    return _payload_value_text(current)


def _is_payload_detail_field(field_name: str) -> bool:
    return str(field_name or "").startswith(MANUFACTURING_PAYLOAD_DETAIL_PREFIX)


def _manufacturing_evidence_by_source(
    evidence: Iterable[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    by_source: Dict[str, Dict[str, Any]] = {}
    for item in evidence:
        if not isinstance(item, dict):
            continue
        source = _normalize_manufacturing_source_token(item.get("source"))
        if source and source not in by_source:
            by_source[source] = item
    return by_source


def _manufacturing_payload_quality_fields(
    evidence: Iterable[Dict[str, Any]],
    expected_payloads: Dict[str, Dict[str, str]],
    *,
    reviewed: bool,
) -> Dict[str, Any]:
    evidence_by_source = _manufacturing_evidence_by_source(evidence)
    expected_field_count = 0
    matched_field_count = 0
    mismatched_field_count = 0
    missing_field_count = 0
    detail_expected_field_count = 0
    detail_matched_field_count = 0
    detail_mismatched_field_count = 0
    detail_missing_field_count = 0
    per_source: Dict[str, Dict[str, Any]] = {}

    for source, expected_fields in sorted(expected_payloads.items()):
        actual_item = evidence_by_source.get(source) or {}
        source_expected_count = 0
        source_matched_count = 0
        source_detail_expected_count = 0
        source_detail_matched_count = 0
        source_mismatches: List[Dict[str, str]] = []
        source_missing_fields: List[str] = []
        actual_values: Dict[str, str] = {}

        for field_name in sorted(expected_fields.keys()):
            expected_value = _clean_expected_payload_value(expected_fields.get(field_name))
            if not expected_value:
                continue
            is_detail_field = _is_payload_detail_field(field_name)
            source_expected_count += 1
            expected_field_count += 1
            if is_detail_field:
                detail_expected_field_count += 1
                source_detail_expected_count += 1
            actual_value = _get_payload_field_value(actual_item, field_name)
            actual_values[field_name] = actual_value
            if not actual_value:
                missing_field_count += 1
                if is_detail_field:
                    detail_missing_field_count += 1
                source_missing_fields.append(field_name)
                continue
            if _normalize_payload_compare_value(actual_value) == (
                _normalize_payload_compare_value(expected_value)
            ):
                matched_field_count += 1
                source_matched_count += 1
                if is_detail_field:
                    detail_matched_field_count += 1
                    source_detail_matched_count += 1
            else:
                mismatched_field_count += 1
                if is_detail_field:
                    detail_mismatched_field_count += 1
                source_mismatches.append(
                    {
                        "field": field_name,
                        "expected": expected_value,
                        "actual": actual_value,
                    }
                )

        if source_expected_count:
            per_source[source] = {
                "expected": {
                    key: value
                    for key, value in expected_fields.items()
                    if (
                        key in MANUFACTURING_PAYLOAD_FIELDS
                        or _is_payload_detail_field(key)
                    )
                    and _clean_expected_payload_value(value)
                },
                "actual": actual_values,
                "expected_field_count": source_expected_count,
                "matched_field_count": source_matched_count,
                "detail_expected_field_count": source_detail_expected_count,
                "detail_matched_field_count": source_detail_matched_count,
                "mismatched_fields": source_mismatches,
                "missing_fields": source_missing_fields,
                "accuracy": round(source_matched_count / source_expected_count, 6),
                "detail_accuracy": (
                    round(source_detail_matched_count / source_detail_expected_count, 6)
                    if source_detail_expected_count
                    else 0.0
                ),
            }

    reviewed = bool(reviewed and expected_field_count > 0)
    detail_reviewed = bool(reviewed and detail_expected_field_count > 0)
    return {
        "manufacturing_evidence_payload_quality_reviewed": reviewed,
        "manufacturing_evidence_payload_detail_quality_reviewed": detail_reviewed,
        "expected_manufacturing_evidence_payloads": _json_cell(expected_payloads),
        "manufacturing_evidence_payload_quality": _json_cell(per_source),
        "manufacturing_evidence_payload_expected_fields": expected_field_count,
        "manufacturing_evidence_payload_matched_fields": matched_field_count,
        "manufacturing_evidence_payload_mismatched_fields": mismatched_field_count,
        "manufacturing_evidence_payload_missing_fields": missing_field_count,
        "manufacturing_evidence_payload_detail_expected_fields": detail_expected_field_count,
        "manufacturing_evidence_payload_detail_matched_fields": detail_matched_field_count,
        "manufacturing_evidence_payload_detail_mismatched_fields": (
            detail_mismatched_field_count
        ),
        "manufacturing_evidence_payload_detail_missing_fields": (
            detail_missing_field_count
        ),
        "manufacturing_evidence_payload_quality_accuracy": (
            round(matched_field_count / expected_field_count, 6)
            if expected_field_count
            else 0.0
        ),
        "manufacturing_evidence_payload_detail_quality_accuracy": (
            round(detail_matched_field_count / detail_expected_field_count, 6)
            if detail_expected_field_count
            else 0.0
        ),
    }


def _collect_decision_contract_fields(results_payload: Dict[str, Any]) -> Dict[str, Any]:
    classification = results_payload.get("classification", {}) or {}
    decision_contract = classification.get("decision_contract")
    if not isinstance(decision_contract, dict):
        decision_contract = {}

    evidence = classification.get("evidence")
    if not isinstance(evidence, list):
        evidence = decision_contract.get("evidence")
    if not isinstance(evidence, list):
        evidence = []

    fallback_flags = _string_tokens(classification.get("fallback_flags"))
    if not fallback_flags:
        fallback_flags = _string_tokens(decision_contract.get("fallback_flags"))

    review_reasons = _string_tokens(classification.get("review_reasons"))
    if not review_reasons:
        review_reasons = _string_tokens(decision_contract.get("review_reasons"))

    branch_conflicts = classification.get("branch_conflicts")
    if not isinstance(branch_conflicts, dict):
        branch_conflicts = decision_contract.get("branch_conflicts")
    if not isinstance(branch_conflicts, dict):
        branch_conflicts = {}

    contract_version = (
        classification.get("contract_version")
        or classification.get("decision_contract_version")
        or decision_contract.get("contract_version")
    )
    decision_source = (
        classification.get("decision_source")
        or decision_contract.get("decision_source")
        or classification.get("confidence_source")
    )
    source_tokens = _evidence_sources(evidence)

    return {
        "decision_contract_present": bool(decision_contract),
        "decision_contract_version": contract_version,
        "decision_source": decision_source,
        "decision_contract": _json_cell(decision_contract),
        "decision_evidence": _json_list_cell(evidence),
        "decision_evidence_count": len(evidence),
        "decision_evidence_sources": ";".join(source_tokens),
        "decision_fallback_flags": ";".join(fallback_flags),
        "decision_review_reasons": ";".join(review_reasons),
        "decision_branch_conflicts": _json_cell(branch_conflicts),
    }


def _collect_knowledge_fields(results_payload: Dict[str, Any]) -> Dict[str, Any]:
    classification = results_payload.get("classification", {}) or {}
    knowledge_checks = classification.get("knowledge_checks") or []
    violations = classification.get("violations") or []
    standards_candidates = classification.get("standards_candidates") or []
    knowledge_hints = classification.get("knowledge_hints") or []

    def _token_join(items: Any, key: str) -> str:
        if not isinstance(items, list):
            return ""
        tokens: List[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            token = str(item.get(key) or "").strip()
            if token:
                tokens.append(token)
        return ";".join(tokens)

    return {
        "knowledge_checks": _json_list_cell(knowledge_checks),
        "violations": _json_list_cell(violations),
        "standards_candidates": _json_list_cell(standards_candidates),
        "knowledge_hints": _json_list_cell(knowledge_hints),
        "knowledge_check_categories": _token_join(knowledge_checks, "category"),
        "knowledge_violation_categories": _token_join(violations, "category"),
        "knowledge_standard_types": _token_join(standards_candidates, "type"),
        "knowledge_hint_labels": _token_join(knowledge_hints, "label"),
    }


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


def _collect_review_fields(results_payload: Dict[str, Any]) -> Dict[str, Any]:
    from src.core.classification import build_review_governance

    classification = results_payload.get("classification", {}) or {}
    review_payload = build_review_governance(
        confidence=classification.get("confidence", 0.0),
        hybrid_rejection=classification.get("hybrid_rejection"),
        branch_conflicts=classification.get("branch_conflicts"),
        violations=classification.get("violations"),
    )
    return {
        "needs_review": classification.get(
            "needs_review", review_payload.get("needs_review")
        ),
        "confidence_band": classification.get(
            "confidence_band", review_payload.get("confidence_band")
        ),
        "review_priority": classification.get(
            "review_priority", review_payload.get("review_priority")
        ),
        "review_priority_score": classification.get(
            "review_priority_score", review_payload.get("review_priority_score")
        ),
        "review_reasons": ";".join(
            classification.get("review_reasons")
            or review_payload.get("review_reasons")
            or []
        ),
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
    row.update(_collect_knowledge_fields(results_payload))
    row.update(_collect_review_fields(results_payload))
    row.update(_collect_decision_contract_fields(results_payload))
    manufacturing_evidence = _manufacturing_evidence_items_from_results(results_payload)
    row.update(_collect_manufacturing_evidence_fields(results_payload))
    predicted_sources = _source_tokens_from_text(
        row.get("manufacturing_evidence_sources")
    )
    row.update(
        _manufacturing_correctness_fields(
            predicted_sources,
            case.expected_manufacturing_evidence_sources,
            reviewed=case.manufacturing_evidence_reviewed,
        )
    )
    row.update(
        _manufacturing_payload_quality_fields(
            manufacturing_evidence,
            case.expected_manufacturing_evidence_payloads,
            reviewed=case.manufacturing_payload_reviewed,
        )
    )
    row.update(_collect_prep_fields(results_payload))
    return row


def _summarize_knowledge_signals(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    check_category_counts: Counter[str] = Counter()
    violation_category_counts: Counter[str] = Counter()
    standard_type_counts: Counter[str] = Counter()
    hint_label_counts: Counter[str] = Counter()
    rows_with_checks = 0
    rows_with_violations = 0
    rows_with_standards_candidates = 0
    rows_with_hints = 0
    total_checks = 0
    total_violations = 0
    total_standards_candidates = 0
    total_hints = 0

    for row in rows:
        check_tokens = [
            token for token in str(row.get("knowledge_check_categories") or "").split(";") if token
        ]
        violation_tokens = [
            token
            for token in str(row.get("knowledge_violation_categories") or "").split(";")
            if token
        ]
        standard_tokens = [
            token for token in str(row.get("knowledge_standard_types") or "").split(";") if token
        ]
        hint_tokens = [
            token for token in str(row.get("knowledge_hint_labels") or "").split(";") if token
        ]
        if check_tokens:
            rows_with_checks += 1
            total_checks += len(check_tokens)
            check_category_counts.update(check_tokens)
        if violation_tokens:
            rows_with_violations += 1
            total_violations += len(violation_tokens)
            violation_category_counts.update(violation_tokens)
        if standard_tokens:
            rows_with_standards_candidates += 1
            total_standards_candidates += len(standard_tokens)
            standard_type_counts.update(standard_tokens)
        if hint_tokens:
            rows_with_hints += 1
            total_hints += len(hint_tokens)
            hint_label_counts.update(hint_tokens)

    def _top(counter: Counter[str]) -> Dict[str, int]:
        return {name: int(count) for name, count in counter.most_common(10)}

    return {
        "rows_with_checks": rows_with_checks,
        "rows_with_violations": rows_with_violations,
        "rows_with_standards_candidates": rows_with_standards_candidates,
        "rows_with_hints": rows_with_hints,
        "total_checks": total_checks,
        "total_violations": total_violations,
        "total_standards_candidates": total_standards_candidates,
        "total_hints": total_hints,
        "top_check_categories": _top(check_category_counts),
        "top_violation_categories": _top(violation_category_counts),
        "top_standard_types": _top(standard_type_counts),
        "top_hint_labels": _top(hint_label_counts),
    }


def _summarize_review_signals(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    confidence_band_counts: Counter[str] = Counter()
    review_priority_counts: Counter[str] = Counter()
    review_reason_counts: Counter[str] = Counter()
    needs_review_count = 0

    for row in rows:
        if row.get("needs_review") is True:
            needs_review_count += 1
        confidence_band = str(row.get("confidence_band") or "").strip()
        if confidence_band:
            confidence_band_counts[confidence_band] += 1
        review_priority = str(row.get("review_priority") or "").strip()
        if review_priority:
            review_priority_counts[review_priority] += 1
        reason_tokens = [
            token for token in str(row.get("review_reasons") or "").split(";") if token
        ]
        review_reason_counts.update(reason_tokens)

    def _top(counter: Counter[str]) -> Dict[str, int]:
        return {name: int(count) for name, count in counter.most_common(10)}

    return {
        "needs_review_count": needs_review_count,
        "confidence_band_counts": _top(confidence_band_counts),
        "review_priority_counts": _top(review_priority_counts),
        "top_review_reasons": _top(review_reason_counts),
    }


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


def _summarize_decision_contract_signals(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    version_counts: Counter[str] = Counter()
    evidence_source_counts: Counter[str] = Counter()
    fallback_flag_counts: Counter[str] = Counter()
    review_reason_counts: Counter[str] = Counter()
    row_count = 0
    decision_contract_count = 0
    evidence_row_count = 0
    evidence_total_count = 0
    branch_conflict_count = 0

    for row in rows:
        row_count += 1
        version = str(row.get("decision_contract_version") or "").strip()
        if version:
            decision_contract_count += 1
            version_counts[version] += 1

        try:
            evidence_count = int(row.get("decision_evidence_count") or 0)
        except (TypeError, ValueError):
            evidence_count = 0
        if evidence_count > 0:
            evidence_row_count += 1
            evidence_total_count += evidence_count

        evidence_sources = [
            token
            for token in str(row.get("decision_evidence_sources") or "").split(";")
            if token
        ]
        evidence_source_counts.update(evidence_sources)

        fallback_flags = [
            token
            for token in str(row.get("decision_fallback_flags") or "").split(";")
            if token
        ]
        fallback_flag_counts.update(fallback_flags)

        review_reasons = [
            token
            for token in str(row.get("decision_review_reasons") or "").split(";")
            if token
        ]
        review_reason_counts.update(review_reasons)

        raw_conflicts = row.get("decision_branch_conflicts") or ""
        try:
            branch_conflicts = json.loads(raw_conflicts) if raw_conflicts else {}
        except (TypeError, json.JSONDecodeError):
            branch_conflicts = {}
        if isinstance(branch_conflicts, dict) and any(branch_conflicts.values()):
            branch_conflict_count += 1

    def _top(counter: Counter[str]) -> Dict[str, int]:
        return {name: int(count) for name, count in counter.most_common(10)}

    return {
        "row_count": row_count,
        "decision_contract_count": decision_contract_count,
        "decision_contract_coverage_rate": round(
            decision_contract_count / row_count, 6
        )
        if row_count
        else 0.0,
        "decision_evidence_row_count": evidence_row_count,
        "decision_evidence_total_count": evidence_total_count,
        "decision_evidence_coverage_rate": round(evidence_row_count / row_count, 6)
        if row_count
        else 0.0,
        "branch_conflict_count": branch_conflict_count,
        "contract_version_counts": _top(version_counts),
        "evidence_source_counts": _top(evidence_source_counts),
        "fallback_flag_counts": _top(fallback_flag_counts),
        "review_reason_counts": _top(review_reason_counts),
    }


def _ordered_manufacturing_sources(counter: Counter[str]) -> List[str]:
    ordered = [
        source
        for source in MANUFACTURING_EVIDENCE_REQUIRED_SOURCES
        if counter.get(source, 0) > 0
    ]
    ordered.extend(
        sorted(source for source in counter.keys() if source not in set(ordered))
    )
    return ordered


def _summarize_manufacturing_evidence(
    rows: Iterable[Dict[str, Any]]
) -> Dict[str, Any]:
    row_count = 0
    records_with_evidence = 0
    evidence_total_count = 0
    reviewed_sample_count = 0
    exact_match_count = 0
    true_positive_total = 0
    false_positive_total = 0
    false_negative_total = 0
    payload_quality_reviewed_sample_count = 0
    payload_expected_field_total = 0
    payload_matched_field_total = 0
    payload_mismatched_field_total = 0
    payload_missing_field_total = 0
    payload_detail_quality_reviewed_sample_count = 0
    payload_detail_expected_field_total = 0
    payload_detail_matched_field_total = 0
    payload_detail_mismatched_field_total = 0
    payload_detail_missing_field_total = 0
    source_counts: Counter[str] = Counter()
    expected_source_counts: Counter[str] = Counter()
    true_positive_counts: Counter[str] = Counter()
    false_positive_counts: Counter[str] = Counter()
    false_negative_counts: Counter[str] = Counter()
    payload_expected_field_counts: Counter[str] = Counter()
    payload_matched_field_counts: Counter[str] = Counter()
    payload_mismatched_field_counts: Counter[str] = Counter()
    payload_missing_field_counts: Counter[str] = Counter()
    payload_detail_expected_field_counts: Counter[str] = Counter()
    payload_detail_matched_field_counts: Counter[str] = Counter()
    payload_detail_mismatched_field_counts: Counter[str] = Counter()
    payload_detail_missing_field_counts: Counter[str] = Counter()

    for row in rows:
        row_count += 1
        try:
            evidence_count = int(row.get("manufacturing_evidence_count") or 0)
        except (TypeError, ValueError):
            evidence_count = 0
        if evidence_count > 0:
            records_with_evidence += 1
            evidence_total_count += evidence_count

        row_sources = [
            token
            for token in str(row.get("manufacturing_evidence_sources") or "").split(";")
            if token
        ]
        source_counts.update(dict.fromkeys(row_sources, 1))

        if row.get("manufacturing_evidence_reviewed") is True:
            reviewed_sample_count += 1
            expected_sources = _source_tokens_from_text(
                row.get("expected_manufacturing_evidence_sources")
            )
            true_positive_sources = _source_tokens_from_text(
                row.get("manufacturing_evidence_true_positive_sources")
            )
            false_positive_sources = _source_tokens_from_text(
                row.get("manufacturing_evidence_false_positive_sources")
            )
            false_negative_sources = _source_tokens_from_text(
                row.get("manufacturing_evidence_false_negative_sources")
            )
            expected_source_counts.update(dict.fromkeys(expected_sources, 1))
            true_positive_counts.update(dict.fromkeys(true_positive_sources, 1))
            false_positive_counts.update(dict.fromkeys(false_positive_sources, 1))
            false_negative_counts.update(dict.fromkeys(false_negative_sources, 1))
            true_positive_total += len(true_positive_sources)
            false_positive_total += len(false_positive_sources)
            false_negative_total += len(false_negative_sources)
            if row.get("manufacturing_evidence_source_exact_match") is True:
                exact_match_count += 1

        if row.get("manufacturing_evidence_payload_quality_reviewed") is True:
            payload_quality_reviewed_sample_count += 1
            try:
                expected_fields = int(
                    row.get("manufacturing_evidence_payload_expected_fields") or 0
                )
            except (TypeError, ValueError):
                expected_fields = 0
            try:
                matched_fields = int(
                    row.get("manufacturing_evidence_payload_matched_fields") or 0
                )
            except (TypeError, ValueError):
                matched_fields = 0
            try:
                mismatched_fields = int(
                    row.get("manufacturing_evidence_payload_mismatched_fields") or 0
                )
            except (TypeError, ValueError):
                mismatched_fields = 0
            try:
                missing_fields = int(
                    row.get("manufacturing_evidence_payload_missing_fields") or 0
                )
            except (TypeError, ValueError):
                missing_fields = 0
            payload_expected_field_total += expected_fields
            payload_matched_field_total += matched_fields
            payload_mismatched_field_total += mismatched_fields
            payload_missing_field_total += missing_fields
            try:
                detail_expected_fields = int(
                    row.get("manufacturing_evidence_payload_detail_expected_fields")
                    or 0
                )
            except (TypeError, ValueError):
                detail_expected_fields = 0
            try:
                detail_matched_fields = int(
                    row.get("manufacturing_evidence_payload_detail_matched_fields")
                    or 0
                )
            except (TypeError, ValueError):
                detail_matched_fields = 0
            try:
                detail_mismatched_fields = int(
                    row.get("manufacturing_evidence_payload_detail_mismatched_fields")
                    or 0
                )
            except (TypeError, ValueError):
                detail_mismatched_fields = 0
            try:
                detail_missing_fields = int(
                    row.get("manufacturing_evidence_payload_detail_missing_fields")
                    or 0
                )
            except (TypeError, ValueError):
                detail_missing_fields = 0
            if row.get("manufacturing_evidence_payload_detail_quality_reviewed") is True:
                payload_detail_quality_reviewed_sample_count += 1
                payload_detail_expected_field_total += detail_expected_fields
                payload_detail_matched_field_total += detail_matched_fields
                payload_detail_mismatched_field_total += detail_mismatched_fields
                payload_detail_missing_field_total += detail_missing_fields

            try:
                quality = json.loads(
                    row.get("manufacturing_evidence_payload_quality") or "{}"
                )
            except (TypeError, json.JSONDecodeError):
                quality = {}
            if isinstance(quality, dict):
                for source, metric in quality.items():
                    normalized_source = _normalize_manufacturing_source_token(source)
                    if not normalized_source or not isinstance(metric, dict):
                        continue
                    payload_expected_field_counts[normalized_source] += int(
                        metric.get("expected_field_count") or 0
                    )
                    payload_matched_field_counts[normalized_source] += int(
                        metric.get("matched_field_count") or 0
                    )
                    payload_mismatched_field_counts[normalized_source] += len(
                        metric.get("mismatched_fields") or []
                    )
                    payload_missing_field_counts[normalized_source] += len(
                        metric.get("missing_fields") or []
                    )
                    payload_detail_expected_field_counts[normalized_source] += int(
                        metric.get("detail_expected_field_count") or 0
                    )
                    payload_detail_matched_field_counts[normalized_source] += int(
                        metric.get("detail_matched_field_count") or 0
                    )
                    payload_detail_mismatched_field_counts[normalized_source] += sum(
                        1
                        for item in metric.get("mismatched_fields") or []
                        if isinstance(item, dict)
                        and _is_payload_detail_field(str(item.get("field") or ""))
                    )
                    payload_detail_missing_field_counts[normalized_source] += sum(
                        1
                        for field_name in metric.get("missing_fields") or []
                        if _is_payload_detail_field(str(field_name or ""))
                    )

    ordered_source_counts: Dict[str, int] = {
        source: int(source_counts.get(source, 0))
        for source in MANUFACTURING_EVIDENCE_REQUIRED_SOURCES
    }
    for source in sorted(
        source for source in source_counts if source not in ordered_source_counts
    ):
        ordered_source_counts[source] = int(source_counts[source])

    def _rate(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 6) if denominator else 0.0

    source_precision = _rate(
        true_positive_total,
        true_positive_total + false_positive_total,
    )
    source_recall = _rate(
        true_positive_total,
        true_positive_total + false_negative_total,
    )
    source_f1 = (
        round(2.0 * source_precision * source_recall / (source_precision + source_recall), 6)
        if source_precision + source_recall
        else 0.0
    )

    def _source_metric(source: str) -> Dict[str, Any]:
        tp = int(true_positive_counts.get(source, 0))
        fp = int(false_positive_counts.get(source, 0))
        fn = int(false_negative_counts.get(source, 0))
        precision = _rate(tp, tp + fp)
        recall = _rate(tp, tp + fn)
        f1 = (
            round(2.0 * precision * recall / (precision + recall), 6)
            if precision + recall
            else 0.0
        )
        return {
            "expected_count": int(expected_source_counts.get(source, 0)),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    payload_quality_accuracy = _rate(
        payload_matched_field_total,
        payload_expected_field_total,
    )
    payload_detail_quality_accuracy = _rate(
        payload_detail_matched_field_total,
        payload_detail_expected_field_total,
    )

    def _payload_quality_metric(source: str) -> Dict[str, Any]:
        expected = int(payload_expected_field_counts.get(source, 0))
        matched = int(payload_matched_field_counts.get(source, 0))
        mismatched = int(payload_mismatched_field_counts.get(source, 0))
        missing = int(payload_missing_field_counts.get(source, 0))
        detail_expected = int(payload_detail_expected_field_counts.get(source, 0))
        detail_matched = int(payload_detail_matched_field_counts.get(source, 0))
        detail_mismatched = int(payload_detail_mismatched_field_counts.get(source, 0))
        detail_missing = int(payload_detail_missing_field_counts.get(source, 0))
        return {
            "expected_field_count": expected,
            "matched_field_count": matched,
            "mismatched_field_count": mismatched,
            "missing_field_count": missing,
            "accuracy": _rate(matched, expected),
            "detail_expected_field_count": detail_expected,
            "detail_matched_field_count": detail_matched,
            "detail_mismatched_field_count": detail_mismatched,
            "detail_missing_field_count": detail_missing,
            "detail_accuracy": _rate(detail_matched, detail_expected),
        }

    return {
        "sample_size": row_count,
        "records_with_manufacturing_evidence": records_with_evidence,
        "manufacturing_evidence_coverage_rate": round(
            records_with_evidence / row_count, 6
        )
        if row_count
        else 0.0,
        "manufacturing_evidence_total_count": evidence_total_count,
        "source_counts": ordered_source_counts,
        "source_coverage_rates": {
            source: round(count / row_count, 6) if row_count else 0.0
            for source, count in ordered_source_counts.items()
        },
        "sources": _ordered_manufacturing_sources(source_counts),
        "required_sources": list(MANUFACTURING_EVIDENCE_REQUIRED_SOURCES),
        "source_correctness_available": reviewed_sample_count > 0,
        "reviewed_sample_count": reviewed_sample_count,
        "source_exact_match_count": exact_match_count,
        "source_exact_match_rate": _rate(exact_match_count, reviewed_sample_count),
        "source_true_positive_total": true_positive_total,
        "source_false_positive_total": false_positive_total,
        "source_false_negative_total": false_negative_total,
        "source_precision": source_precision,
        "source_recall": source_recall,
        "source_f1": source_f1,
        "expected_source_counts": {
            source: int(expected_source_counts.get(source, 0))
            for source in MANUFACTURING_EVIDENCE_REQUIRED_SOURCES
        },
        "source_correctness": {
            source: _source_metric(source)
            for source in MANUFACTURING_EVIDENCE_REQUIRED_SOURCES
        },
        "payload_quality_available": payload_quality_reviewed_sample_count > 0,
        "payload_quality_reviewed_sample_count": payload_quality_reviewed_sample_count,
        "payload_quality_expected_field_total": payload_expected_field_total,
        "payload_quality_matched_field_total": payload_matched_field_total,
        "payload_quality_mismatched_field_total": payload_mismatched_field_total,
        "payload_quality_missing_field_total": payload_missing_field_total,
        "payload_quality_accuracy": payload_quality_accuracy,
        "payload_detail_quality_available": (
            payload_detail_quality_reviewed_sample_count > 0
        ),
        "payload_detail_quality_reviewed_sample_count": (
            payload_detail_quality_reviewed_sample_count
        ),
        "payload_detail_quality_expected_field_total": (
            payload_detail_expected_field_total
        ),
        "payload_detail_quality_matched_field_total": payload_detail_matched_field_total,
        "payload_detail_quality_mismatched_field_total": (
            payload_detail_mismatched_field_total
        ),
        "payload_detail_quality_missing_field_total": payload_detail_missing_field_total,
        "payload_detail_quality_accuracy": payload_detail_quality_accuracy,
        "payload_quality": {
            source: _payload_quality_metric(source)
            for source in MANUFACTURING_EVIDENCE_REQUIRED_SOURCES
        },
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
        "knowledge_signals": _summarize_knowledge_signals(ok_rows),
        "review_signals": _summarize_review_signals(ok_rows),
        "decision_signals": _summarize_decision_contract_signals(ok_rows),
        "manufacturing_evidence": _summarize_manufacturing_evidence(ok_rows),
        "prep_signals": _summarize_prep_signals(ok_rows),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
