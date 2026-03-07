"""Build lightweight knowledge checks from drawing text and geometry signals."""

from __future__ import annotations

import re
import json
from typing import Any, Dict, List, Optional, Sequence

from src.core.classification.coarse_labels import normalize_coarse_label
from src.core.knowledge.dynamic.manager import get_knowledge_manager
from src.core.knowledge.gdt.application import interpret_feature_control_frame
from src.core.knowledge.gdt.datums import validate_datum_sequence
from src.core.knowledge.standards.threads import (
    calculate_thread_engagement,
    get_thread_spec,
)
from src.core.knowledge.tolerance.it_grades import get_grade_info
from src.core.knowledge.design_standards.general_tolerances import (
    GeneralToleranceSpec,
)

THREAD_RE = re.compile(
    r"(?<![A-Za-z0-9])M\s*(\d+(?:\.\d+)?(?:\s*[xX×]\s*\d+(?:\.\d+)?)?)(?=$|[^A-Za-z0-9.])",
    re.IGNORECASE,
)
ISO_2768_RE = re.compile(
    r"(?<![A-Za-z0-9])ISO\s*2768\s*-\s*([FMCV][HKL]?)(?=$|[^A-Za-z0-9])",
    re.IGNORECASE,
)
IT_GRADE_RE = re.compile(
    r"(?<![A-Za-z0-9])IT(?:01|0|[1-9]|1[0-8])(?=$|[^A-Za-z0-9])",
    re.IGNORECASE,
)
GDT_HINT_RE = re.compile(
    r"(直线度|平面度|圆度|圆柱度|垂直度|平行度|倾斜度|位置度|同心度|对称度|圆跳动|全跳动|线轮廓度|面轮廓度)[^\n;|]*",
    re.IGNORECASE,
)


def _dedupe(items: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        cleaned = str(item or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def _split_text_items(text_signals: str, text_items: Optional[Sequence[str]]) -> List[str]:
    items: List[str] = []
    if text_signals:
        items.extend(
            token.strip()
            for token in re.split(r"[\n\r;|]+", text_signals)
            if str(token).strip()
        )
        items.append(text_signals)
    if text_items:
        items.extend(str(item).strip() for item in text_items if str(item).strip())
    return _dedupe(items)


def _format_iso_2768_designation(raw_designation: str) -> str:
    cleaned = str(raw_designation or "").strip().upper().replace(" ", "")
    if cleaned.startswith("ISO2768-"):
        return f"ISO 2768-{cleaned.split('-', 1)[1]}"
    return cleaned


def _dedupe_dict_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for row in rows:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(row))
    return out


def build_knowledge_summary(
    *,
    text_signals: str,
    text_items: Optional[Sequence[str]] = None,
    geometric_features: Optional[Dict[str, Any]] = None,
    entity_counts: Optional[Dict[str, int]] = None,
    fine_part_type: Optional[str] = None,
    coarse_part_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Build lightweight knowledge checks and warnings for API/reporting use."""
    checks: List[Dict[str, Any]] = []
    violations: List[Dict[str, Any]] = []
    standards_candidates: List[Dict[str, Any]] = []

    combined_text = str(text_signals or "").strip()
    search_text = combined_text.replace("_", " ")
    snippets = _split_text_items(search_text, text_items)

    thread_matches = _dedupe(
        match.group(0).replace(" ", "").replace("×", "x")
        for match in THREAD_RE.finditer(search_text)
    )
    for designation in thread_matches:
        spec = get_thread_spec(designation)
        if spec is None:
            continue
        engagement = calculate_thread_engagement(spec.designation) or {}
        checks.append(
            {
                "category": "thread_standard",
                "item": spec.designation,
                "value": {
                    "pitch": spec.pitch,
                    "tap_drill_size": spec.tap_drill_size,
                    "thread_type": spec.thread_type.value,
                    "recommended_engagement_mm": engagement.get(
                        "recommended_engagement_mm"
                    ),
                },
                "confidence": 0.95,
                "source": "text",
                "status": "ok",
            }
        )
        standards_candidates.append(
            {
                "type": "metric_thread",
                "designation": spec.designation,
                "thread_type": spec.thread_type.value,
                "tap_drill_size": spec.tap_drill_size,
            }
        )

    iso_matches = _dedupe(
        _format_iso_2768_designation(f"ISO2768-{match.group(1)}")
        for match in ISO_2768_RE.finditer(search_text)
    )
    for designation in iso_matches:
        spec = GeneralToleranceSpec.from_designation(designation)
        if spec is None:
            continue
        checks.append(
            {
                "category": "general_tolerance",
                "item": spec.designation,
                "value": {
                    "linear_class": spec.linear_class.value,
                    "angular_class": spec.angular_class.value,
                },
                "confidence": 0.9,
                "source": "text",
                "status": "ok",
            }
        )
        standards_candidates.append(
            {
                "type": "general_tolerance",
                "designation": _format_iso_2768_designation(spec.designation),
                "linear_class": spec.linear_class.value,
            }
        )

    grade_matches = _dedupe(
        match.group(0).upper() for match in IT_GRADE_RE.finditer(search_text)
    )
    for grade in grade_matches:
        info = get_grade_info(grade)
        if not info:
            continue
        checks.append(
            {
                "category": "it_grade",
                "item": grade,
                "value": info,
                "confidence": 0.85,
                "source": "text",
                "status": "ok",
            }
        )

    gdt_snippets = _dedupe(
        [match.group(0).strip() for match in GDT_HINT_RE.finditer(search_text)] + snippets
    )
    for snippet in gdt_snippets:
        frame = interpret_feature_control_frame(snippet)
        if frame is None:
            continue
        datums = [
            datum
            for datum in [
                frame.primary_datum,
                frame.secondary_datum,
                frame.tertiary_datum,
            ]
            if datum
        ]
        datum_validation = validate_datum_sequence(datums) if datums else None
        checks.append(
            {
                "category": "gdt",
                "item": frame.characteristic.value,
                "value": {
                    "tolerance_value": frame.tolerance_value,
                    "datums": datums,
                    "modifier": (
                        frame.tolerance_modifier.value
                        if frame.tolerance_modifier is not None
                        else None
                    ),
                },
                "confidence": 0.8,
                "source": "text",
                "status": "ok",
            }
        )
        if datum_validation and (datum_validation.get("issues") or []):
            violations.append(
                {
                    "category": "datum_sequence",
                    "severity": (
                        "warn"
                        if datum_validation.get("is_valid", True)
                        else "error"
                    ),
                    "message": ";".join(datum_validation.get("issues") or []),
                    "source": "gdt",
                }
            )

    knowledge_hints: Dict[str, float] = {}
    try:
        knowledge_hints = get_knowledge_manager().get_part_hints(
            search_text,
            geometric_features=geometric_features or {},
            entity_counts=entity_counts or {},
        )
    except Exception:
        knowledge_hints = {}

    ranked_hints = sorted(
        (
            {
                "label": str(label).strip(),
                "coarse_label": normalize_coarse_label(label),
                "score": round(float(score), 6),
            }
            for label, score in knowledge_hints.items()
            if str(label).strip()
        ),
        key=lambda item: (-item["score"], item["label"]),
    )

    coarse_label = normalize_coarse_label(coarse_part_type or fine_part_type)
    if coarse_label and ranked_hints:
        best_hint = ranked_hints[0]
        best_hint_coarse = normalize_coarse_label(best_hint.get("label"))
        if best_hint_coarse and best_hint_coarse != coarse_label:
            violations.append(
                {
                    "category": "knowledge_conflict",
                    "severity": "warn",
                    "message": (
                        f"knowledge_hint={best_hint.get('label')} conflicts with "
                        f"classification={coarse_label}"
                    ),
                    "source": "knowledge_manager",
                }
            )

    return {
        "knowledge_checks": _dedupe_dict_rows(checks),
        "violations": _dedupe_dict_rows(violations),
        "standards_candidates": standards_candidates,
        "knowledge_hints": ranked_hints[:5],
    }
