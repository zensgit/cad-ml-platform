from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_json_object(path: Path, kind_label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"failed to read {kind_label} json: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"failed to parse {kind_label} json: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{kind_label} json must be an object")
    return payload


def is_zeroish(value: Any) -> bool:
    return str(value if value is not None else "").strip() == "0"


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value if value is not None else "").strip().lower()
    return text in {"1", "true", "yes", "on"}


def top_nonempty(values: Iterable[Any], limit: int = 3) -> list[str]:
    rows: list[str] = []
    for value in values:
        text = str(value if value is not None else "").strip()
        if not text:
            continue
        rows.append(text)
        if len(rows) >= limit:
            break
    return rows


def render_inline_items(values: Iterable[Any], empty: str = "(none)") -> str:
    rows = [str(value).strip() for value in values if str(value if value is not None else "").strip()]
    return ", ".join(rows) if rows else empty


def append_markdown_section(
    lines: list[str],
    title: str,
    rows: Iterable[tuple[str, Any]],
    *,
    level: int = 2,
    blank_after_header: bool = True,
) -> None:
    header_prefix = "#" * max(level, 1)
    lines.extend(["", f"{header_prefix} {title}"])
    if blank_after_header:
        lines.append("")
    for key, value in rows:
        lines.append(f"- {key}: {value}")


def append_failure_diagnostics_section(
    lines: list[str],
    diagnostics: dict[str, Any],
    failed_jobs: Iterable[dict[str, Any]],
) -> None:
    if not diagnostics:
        return

    append_markdown_section(
        lines,
        "Failure Diagnostics",
        [
            ("available", diagnostics.get("available", "n/a")),
            ("failed_job_count", diagnostics.get("failed_job_count", "n/a")),
        ],
        level=3,
        blank_after_header=False,
    )
    detail_reason = str(diagnostics.get("reason") or "").strip()
    if detail_reason:
        lines.append(f"- reason: {detail_reason}")
    for item in failed_jobs:
        if not isinstance(item, dict):
            continue
        lines.append(
            "- failed_job: {} job_conclusion={} failed_step={} step_conclusion={}".format(
                item.get("job_name", ""),
                item.get("job_conclusion", ""),
                item.get("failed_step_name", ""),
                item.get("failed_step_conclusion", ""),
            )
        )


def append_dispatch_verdict_and_snapshot_sections(
    lines: list[str],
    *,
    verdict: str,
    expected_conclusion: Any,
    conclusion: Any,
    top_failed_jobs: str,
    top_failed_steps: str,
    failed_job_count: int,
    diagnostics_reason: str = "",
    fallback_reason: str = "",
) -> None:
    append_markdown_section(
        lines,
        "Dispatch Verdict",
        [
            ("verdict", verdict),
            ("conclusion_pair", f"expected={expected_conclusion} actual={conclusion}"),
            ("top_failed_jobs", top_failed_jobs),
            ("top_failed_steps", top_failed_steps),
        ],
    )
    if diagnostics_reason:
        lines.append(f"- diagnostics_reason: {diagnostics_reason}")
    elif fallback_reason:
        lines.append(f"- diagnostics_reason: {fallback_reason}")

    append_markdown_section(
        lines,
        "Dispatch Snapshot",
        [
            ("failed_job_count", failed_job_count),
            ("top_failed_jobs", top_failed_jobs),
            ("top_failed_steps", top_failed_steps),
            ("failure_reason", diagnostics_reason or fallback_reason or "n/a"),
        ],
    )
