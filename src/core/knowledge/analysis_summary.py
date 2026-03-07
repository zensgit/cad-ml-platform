"""Build lightweight knowledge checks from drawing text and geometry signals."""

from __future__ import annotations

import re
import json
from typing import Any, Dict, List, Optional, Sequence

from src.core.classification.coarse_labels import normalize_coarse_label
from src.core.materials import classify_material_detailed
from src.core.knowledge.dynamic.manager import get_knowledge_manager
from src.core.knowledge.design_standards.surface_finish import (
    SurfaceFinishGrade,
    get_ra_value,
    suggest_surface_finish,
)
from src.core.knowledge.gdt.application import interpret_feature_control_frame
from src.core.knowledge.gdt.datums import validate_datum_sequence
from src.core.knowledge.standards.threads import (
    calculate_thread_engagement,
    get_thread_spec,
)
from src.core.knowledge.tolerance.it_grades import get_grade_info
from src.core.knowledge.design_standards.general_tolerances import (
    GeneralToleranceClass,
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
GBT_1804_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:GB\s*/\s*T|GBT)\s*1804\s*[-－]?\s*([FMCV])(?=$|[^A-Za-z0-9])",
    re.IGNORECASE,
)
IT_GRADE_RE = re.compile(
    r"(?<![A-Za-z0-9])IT(?:01|0|[1-9]|1[0-8])(?=$|[^A-Za-z0-9])",
    re.IGNORECASE,
)
SURFACE_RA_RE = re.compile(
    r"(?<![A-Za-z0-9])RA\s*([0-9]+(?:\.[0-9]+)?)(?=$|[^A-Za-z0-9.])",
    re.IGNORECASE,
)
SURFACE_GRADE_RE = re.compile(
    r"(?<![A-Za-z0-9])(N(?:[1-9]|1[0-2]))(?=$|[^A-Za-z0-9])",
    re.IGNORECASE,
)
GDT_HINT_RE = re.compile(
    r"(直线度|平面度|圆度|圆柱度|垂直度|平行度|倾斜度|位置度|同心度|对称度|圆跳动|全跳动|线轮廓度|面轮廓度)[^\n;|]*",
    re.IGNORECASE,
)
MATERIAL_LABEL_RE = re.compile(r"(?:材料|材质)\s*[:：]?\s*(.+)", re.IGNORECASE)


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


def _format_gbt_1804_designation(raw_class: str) -> str:
    cleaned = str(raw_class or "").strip().upper()
    return f"GB/T 1804-{cleaned}"


def _extract_material_candidates(snippets: Sequence[str]) -> List[str]:
    candidates: List[str] = []
    for snippet in snippets:
        match = MATERIAL_LABEL_RE.search(str(snippet or ""))
        if not match:
            continue
        candidate = str(match.group(1) or "").strip()
        if not candidate:
            continue
        candidate = re.split(r"[\n\r;|,，]", candidate, maxsplit=1)[0].strip()
        if candidate:
            candidates.append(candidate)
    return _dedupe(candidates)


def _resolve_material_candidate(raw_candidate: str) -> Optional[Dict[str, Any]]:
    candidate = str(raw_candidate or "").strip()
    if not candidate:
        return None

    candidate = re.split(
        r"R[Aa]\s*[0-9]|表面粗糙度|粗糙度|位置度|直线度|平面度|"
        r"IT(?:01|0|[1-9]|1[0-8])|ISO|GBT|GB/T|N(?:[1-9]|1[0-2])|"
        r"图号|名称|零件|人孔",
        candidate,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    if not candidate:
        return None

    info = classify_material_detailed(candidate)
    if info is not None:
        return {"raw": candidate, "info": info}

    tokens = [
        token
        for token in re.split(r"\s+", candidate)
        if token and token not in {"表面粗糙度", "粗糙度"}
    ]
    for end in range(1, min(len(tokens), 4) + 1):
        probe = " ".join(tokens[:end]).strip()
        info = classify_material_detailed(probe)
        if info is not None:
            return {"raw": probe, "info": info}

    return None


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

    gbt_matches = _dedupe(
        _format_gbt_1804_designation(match.group(1))
        for match in GBT_1804_RE.finditer(search_text)
    )
    for designation in gbt_matches:
        linear_code = designation.rsplit("-", 1)[-1].lower()
        try:
            spec_class = GeneralToleranceClass(linear_code)
        except ValueError:
            continue
        checks.append(
            {
                "category": "general_tolerance",
                "item": designation,
                "value": {
                    "linear_class": spec_class.value,
                    "angular_class": spec_class.value,
                    "equivalent_iso_designation": f"ISO 2768-{spec_class.value.upper()}",
                },
                "confidence": 0.9,
                "source": "text",
                "status": "ok",
            }
        )
        standards_candidates.append(
            {
                "type": "general_tolerance",
                "designation": designation,
                "linear_class": spec_class.value,
                "equivalent_designation": f"ISO 2768-{spec_class.value.upper()}",
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

    ra_matches = _dedupe(match.group(1) for match in SURFACE_RA_RE.finditer(search_text))
    for ra_text in ra_matches:
        try:
            ra_um = float(ra_text)
        except (TypeError, ValueError):
            continue
        grade = suggest_surface_finish(ra_um)
        checks.append(
            {
                "category": "surface_finish",
                "item": f"Ra {ra_text}",
                "value": {
                    "ra_um": ra_um,
                    "grade": grade.value,
                    "standard_ra_um": get_ra_value(grade),
                },
                "confidence": 0.88,
                "source": "text",
                "status": "ok",
            }
        )
        standards_candidates.append(
            {
                "type": "surface_finish",
                "designation": f"Ra {ra_text}",
                "grade": grade.value,
                "ra_um": ra_um,
            }
        )

    surface_grade_matches = _dedupe(
        match.group(1).upper() for match in SURFACE_GRADE_RE.finditer(search_text)
    )
    for grade_text in surface_grade_matches:
        try:
            grade = SurfaceFinishGrade(grade_text)
        except ValueError:
            continue
        checks.append(
            {
                "category": "surface_finish",
                "item": grade.value,
                "value": {
                    "grade": grade.value,
                    "ra_um": get_ra_value(grade),
                },
                "confidence": 0.88,
                "source": "text",
                "status": "ok",
            }
        )
        standards_candidates.append(
            {
                "type": "surface_finish",
                "designation": grade.value,
                "grade": grade.value,
                "ra_um": get_ra_value(grade),
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

    material_candidates = _extract_material_candidates(snippets)
    for candidate in material_candidates:
        resolved = _resolve_material_candidate(candidate)
        if resolved is None:
            continue
        material_item = str(resolved["raw"])
        material_info = resolved["info"]
        checks.append(
            {
                "category": "material",
                "item": material_item,
                "value": {
                    "grade": material_info.grade,
                    "name": material_info.name,
                    "group": material_info.group.value,
                    "category": material_info.category.value,
                    "standards": material_info.standards[:5],
                },
                "confidence": 0.92,
                "source": "text",
                "status": "ok",
            }
        )
        standards_candidates.append(
            {
                "type": "material",
                "designation": material_item,
                "grade": material_info.grade,
                "group": material_info.group.value,
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
