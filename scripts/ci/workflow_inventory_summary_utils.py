from __future__ import annotations

from typing import Any


def _normalize_names(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    names: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if text:
            names.append(text)
    return names


def _extract_duplicate_names(payload: dict[str, Any]) -> list[str]:
    duplicate_names = _normalize_names(payload.get("duplicate_names"))
    if duplicate_names:
        return duplicate_names

    duplicates = payload.get("duplicates")
    if not isinstance(duplicates, list):
        return []

    names: list[str] = []
    for item in duplicates:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name:
            names.append(name)
    return names


def _extract_required_mapping_names(payload: dict[str, Any], target_status: str) -> list[str]:
    mapping = payload.get("required_workflow_mapping")
    if not isinstance(mapping, list):
        return []

    names: list[str] = []
    for item in mapping:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "").strip()
        if status != target_status:
            continue
        name = str(item.get("name") or "").strip()
        if name:
            names.append(name)
    return names


def summarize_workflow_inventory_payload(payload: dict[str, Any]) -> str:
    summary = str(payload.get("summary") or "").strip()
    if not summary:
        workflow_count = int(payload.get("workflow_count") or 0)
        duplicate_count = int(payload.get("duplicate_name_count") or 0)
        missing_required_count = int(payload.get("missing_required_count") or 0)
        non_unique_required_count = int(payload.get("non_unique_required_count") or 0)
        summary = (
            "workflows="
            f"{workflow_count}, duplicate={duplicate_count}, "
            f"missing_required={missing_required_count}, "
            f"non_unique_required={non_unique_required_count}"
        )

    duplicate_names = _extract_duplicate_names(payload)
    missing_required_names = _normalize_names(payload.get("missing_required_names"))
    if not missing_required_names:
        missing_required_names = _extract_required_mapping_names(payload, "missing")

    non_unique_required_names = _normalize_names(payload.get("non_unique_required_names"))
    if not non_unique_required_names:
        non_unique_required_names = _extract_required_mapping_names(payload, "non_unique")

    detail_parts: list[str] = []
    if duplicate_names:
        detail_parts.append(f"duplicate_names={'/'.join(duplicate_names)}")
    if missing_required_names:
        detail_parts.append(f"missing_names={'/'.join(missing_required_names)}")
    if non_unique_required_names:
        detail_parts.append(f"non_unique_names={'/'.join(non_unique_required_names)}")

    if detail_parts:
        return f"{summary}; {'; '.join(detail_parts)}"
    return summary
