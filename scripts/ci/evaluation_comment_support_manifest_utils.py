from __future__ import annotations

from typing import Any


def _safe_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _derive_counts_from_entries(entries: list[dict[str, Any]]) -> tuple[int, int, int]:
    present_count = 0
    missing_count = 0
    invalid_count = 0

    for entry in entries:
        if bool(entry.get("present")):
            present_count += 1
        else:
            missing_count += 1
        if str(entry.get("parse_status") or "").strip() == "error":
            invalid_count += 1

    return present_count, missing_count, invalid_count


def summarize_manifest_payload(payload: dict[str, Any]) -> str:
    summary = str(payload.get("summary") or "").strip()
    if summary:
        return summary

    entries_raw = payload.get("entries")
    if isinstance(entries_raw, list):
        entries = [item for item in entries_raw if isinstance(item, dict)]
    else:
        entries = []

    present_count = _safe_int(payload.get("present_count"))
    missing_count = _safe_int(payload.get("missing_count"))
    invalid_count = _safe_int(payload.get("invalid_count"))

    if entries and (present_count is None or missing_count is None or invalid_count is None):
        derived_present, derived_missing, derived_invalid = _derive_counts_from_entries(entries)
        if present_count is None:
            present_count = derived_present
        if missing_count is None:
            missing_count = derived_missing
        if invalid_count is None:
            invalid_count = derived_invalid

    total = 0
    if present_count is not None and missing_count is not None:
        total = present_count + missing_count
    elif entries:
        total = len(entries)

    if (
        present_count is not None
        and missing_count is not None
        and invalid_count is not None
        and total >= 0
    ):
        return f"present={present_count}/{total}, missing={missing_count}, invalid={invalid_count}"

    verdict = str(payload.get("verdict") or "").strip()
    if verdict:
        return f"verdict={verdict}"

    overall_status = str(payload.get("overall_status") or "").strip()
    if overall_status:
        return f"status={overall_status}"

    return "n/a"
