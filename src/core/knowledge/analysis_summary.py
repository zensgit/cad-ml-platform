"""Build lightweight knowledge checks from drawing text and geometry signals."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from src.core.knowledge.design_standards.general_tolerances import (
    GeneralToleranceClass,
    GeneralToleranceSpec,
)
from src.core.knowledge.design_standards.surface_finish import (
    SurfaceFinishGrade,
    get_ra_value,
    suggest_surface_finish,
)
from src.core.knowledge.dynamic.manager import get_knowledge_manager
from src.core.knowledge.gdt.application import interpret_feature_control_frame
from src.core.knowledge.gdt.datums import validate_datum_sequence
from src.core.knowledge.standards.threads import (
    calculate_thread_engagement,
    get_thread_spec,
)
from src.core.knowledge.tolerance.fits import get_fit_info
from src.core.knowledge.tolerance.it_grades import get_grade_info
from src.core.materials import classify_material_detailed
from src.core.materials.equivalence import get_material_equivalence

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
FIT_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Z]{1,2}\s*\d{1,2})\s*/\s*"
    r"([a-z]{1,2}\s*\d{1,2})(?=$|[^A-Za-z0-9])"
)
GDT_HINT_RE = re.compile(
    r"(直线度|平面度|圆度|圆柱度|垂直度|平行度|倾斜度|位置度|同心度|对称度|圆跳动|全跳动|线轮廓度|面轮廓度)[^\n;|]*",
    re.IGNORECASE,
)
MATERIAL_LABEL_RE = re.compile(r"(?:材料|材质)\s*[:：]?\s*(.+)", re.IGNORECASE)
MATERIAL_SUBSTITUTION_RE = re.compile(
    r"([A-Za-z0-9#._+\-/]+)\s*(?:->|=>|→|替代为|代替为|替换为|改为|to)\s*"
    r"([A-Za-z0-9#._+\-/]+)",
    re.IGNORECASE,
)
SURFACE_RECOMMENDATION_RE = re.compile(
    r"(?:建议|推荐|要求|指定|recommend|recommended|require).*"
    r"(?:Ra\s*[0-9]+(?:\.[0-9]+)?|N(?:[1-9]|1[0-2]))|"
    r"(?:Ra\s*[0-9]+(?:\.[0-9]+)?|N(?:[1-9]|1[0-2])).*"
    r"(?:建议|推荐|要求|指定|recommend|recommended|require)",
    re.IGNORECASE,
)

PROCESS_ROUTE_TERMS = (
    ("turning", "车削", ("车削", "车", "turning", "lathe")),
    ("milling", "铣削", ("铣削", "铣", "milling", "mill")),
    ("drilling", "钻孔", ("钻孔", "钻", "drilling", "drill")),
    ("boring", "镗孔", ("镗孔", "镗", "boring", "bore")),
    ("grinding", "磨削", ("磨削", "磨", "grinding", "grind")),
    ("heat_treatment", "热处理", ("热处理", "heat treatment")),
    ("surface_treatment", "表面处理", ("表面处理", "surface treatment")),
)
PROCESS_ROUTE_CONTEXT_RE = re.compile(
    r"工艺路线|加工路线|加工工序|工序|process\s*route|machining\s*route|route",
    re.IGNORECASE,
)
MANUFACTURABILITY_TEXT_PATTERNS = (
    (
        "THIN_WALL",
        "thin_wall",
        "薄壁风险",
        re.compile(r"薄壁|thin\s*wall", re.IGNORECASE),
    ),
    (
        "DEEP_HOLE",
        "deep_hole",
        "深孔加工风险",
        re.compile(r"深孔|deep\s*hole", re.IGNORECASE),
    ),
    (
        "SHARP_INTERNAL_CORNER",
        "sharp_internal_corner",
        "内尖角加工风险",
        re.compile(r"内尖角|内直角|尖角|sharp\s*internal\s*corner", re.IGNORECASE),
    ),
    (
        "HIGH_STOCK_REMOVAL",
        "high_stock_removal",
        "高材料去除率风险",
        re.compile(r"高去除率|材料去除率高|high\s*stock\s*removal", re.IGNORECASE),
    ),
)

KNOWLEDGE_RULE_VERSION = "knowledge_grounding.v1"
_CHECK_RULE_SOURCES = {
    "thread_standard": "iso_metric_thread_catalog",
    "general_tolerance": "iso2768_gbt1804_tolerance_catalog",
    "it_grade": "iso286_it_grade_catalog",
    "fit_validation": "iso286_fit_catalog",
    "surface_finish": "iso1302_surface_finish_catalog",
    "surface_finish_recommendation": "iso1302_surface_finish_catalog",
    "gdt": "iso1101_gdt_catalog",
    "material": "materials_catalog",
    "material_substitution": "materials_catalog",
    "machining_process_route": "machining_process_knowledge_base",
    "manufacturability_risk": "dfm_manufacturability_rules",
}
_CANDIDATE_RULE_SOURCES = {
    "metric_thread": "iso_metric_thread_catalog",
    "general_tolerance": "iso2768_gbt1804_tolerance_catalog",
    "iso_fit": "iso286_fit_catalog",
    "surface_finish": "iso1302_surface_finish_catalog",
    "material": "materials_catalog",
}
_VIOLATION_RULE_SOURCES = {
    "datum_sequence": "iso1101_gdt_catalog",
    "knowledge_conflict": "knowledge_manager",
    "manufacturability_risk": "dfm_manufacturability_rules",
}


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


def _split_text_items(
    text_signals: str, text_items: Optional[Sequence[str]]
) -> List[str]:
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


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_fit_code(match: re.Match[str]) -> Optional[str]:
    hole_raw = re.sub(r"\s+", "", str(match.group(1) or ""))
    shaft_raw = re.sub(r"\s+", "", str(match.group(2) or ""))
    if not hole_raw or not shaft_raw:
        return None

    hole_match = re.match(r"([A-Za-z]+)(\d+)$", hole_raw)
    shaft_match = re.match(r"([A-Za-z]+)(\d+)$", shaft_raw)
    if not hole_match or not shaft_match:
        return None
    hole_symbol, hole_grade = hole_match.groups()
    shaft_symbol, shaft_grade = shaft_match.groups()
    return f"{hole_symbol.upper()}{hole_grade}/{shaft_symbol.lower()}{shaft_grade}"


def _extract_fit_codes(search_text: str) -> List[str]:
    return _dedupe(
        fit_code
        for fit_code in (
            _normalize_fit_code(match) for match in FIT_RE.finditer(search_text)
        )
        if fit_code
    )


def _material_summary_payload(raw_candidate: str) -> Optional[Dict[str, Any]]:
    resolved = _resolve_material_candidate(raw_candidate)
    if resolved is None:
        return None
    material_info = resolved["info"]
    raw = str(resolved["raw"])
    return {
        "raw": raw,
        "grade": material_info.grade,
        "name": material_info.name,
        "group": material_info.group.value,
        "category": material_info.category.value,
        "standards": material_info.standards[:5],
        "equivalents": get_material_equivalence(raw) or {},
    }


def _extract_material_substitutions(snippets: Sequence[str]) -> List[Dict[str, Any]]:
    substitutions: List[Dict[str, Any]] = []
    for snippet in snippets:
        if not re.search(
            r"替代|代替|替换|substitut|replace|->|=>|→",
            snippet,
            re.IGNORECASE,
        ):
            continue
        for match in MATERIAL_SUBSTITUTION_RE.finditer(snippet):
            source_material = _material_summary_payload(match.group(1))
            target_material = _material_summary_payload(match.group(2))
            if source_material is None or target_material is None:
                continue
            substitutions.append(
                {
                    "source_material": source_material,
                    "target_material": target_material,
                }
            )
    return _dedupe_dict_rows(substitutions)


def _extract_process_route(search_text: str) -> List[Dict[str, str]]:
    matches: List[tuple[int, str, str]] = []
    for process_key, label, terms in PROCESS_ROUTE_TERMS:
        for term in terms:
            for match in re.finditer(re.escape(term), search_text, re.IGNORECASE):
                matches.append((match.start(), process_key, label))

    route: List[Dict[str, str]] = []
    seen: set[str] = set()
    for _, process_key, label in sorted(matches, key=lambda item: item[0]):
        if process_key in seen:
            continue
        seen.add(process_key)
        route.append({"process": process_key, "label": label})

    if not route:
        return []
    if len(route) < 2 and not PROCESS_ROUTE_CONTEXT_RE.search(search_text):
        return []
    return route


def _append_unique_risk(
    risks: List[Dict[str, Any]],
    *,
    code: str,
    risk_type: str,
    description: str,
    trigger: str,
    severity: str = "warn",
    details: Optional[Dict[str, Any]] = None,
) -> None:
    if any(item.get("code") == code for item in risks):
        return
    row: Dict[str, Any] = {
        "code": code,
        "risk_type": risk_type,
        "description": description,
        "trigger": trigger,
        "severity": severity,
    }
    if details:
        row["details"] = details
    risks.append(row)


def _extract_manufacturability_risks(
    search_text: str,
    geometric_features: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    risks: List[Dict[str, Any]] = []
    for code, risk_type, description, pattern in MANUFACTURABILITY_TEXT_PATTERNS:
        if pattern.search(search_text):
            _append_unique_risk(
                risks,
                code=code,
                risk_type=risk_type,
                description=description,
                trigger="text",
            )

    features = geometric_features or {}
    if features.get("thin_walls_detected"):
        min_thickness = _safe_float(features.get("min_thickness_estimate"))
        if min_thickness is None or min_thickness < 0.8:
            _append_unique_risk(
                risks,
                code="THIN_WALL",
                risk_type="thin_wall",
                description="薄壁低于默认制造性阈值",
                trigger="geometry",
                severity=(
                    "high"
                    if min_thickness is not None and min_thickness < 0.5
                    else "warn"
                ),
                details={"min_thickness_mm": min_thickness, "threshold_mm": 0.8},
            )

    stock_removal = _safe_float(features.get("stock_removal_ratio"))
    if stock_removal is not None and stock_removal > 0.85:
        _append_unique_risk(
            risks,
            code="HIGH_STOCK_REMOVAL",
            risk_type="high_stock_removal",
            description="材料去除率超过默认制造性阈值",
            trigger="geometry",
            details={"stock_removal_ratio": stock_removal, "threshold": 0.85},
        )

    aspect_ratio = _safe_float(features.get("aspect_ratio_max_min"))
    if aspect_ratio is not None and aspect_ratio > 10.0:
        _append_unique_risk(
            risks,
            code="SLENDER_PART",
            risk_type="slender_part",
            description="长细比超过默认车削稳定性阈值",
            trigger="geometry",
            details={"aspect_ratio_max_min": aspect_ratio, "threshold": 10.0},
        )

    return risks


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


def _rule_source_for_row(row: Dict[str, Any], *, default: str) -> str:
    category = str(row.get("category") or "").strip()
    if category in _CHECK_RULE_SOURCES:
        return _CHECK_RULE_SOURCES[category]
    if category in _VIOLATION_RULE_SOURCES:
        return _VIOLATION_RULE_SOURCES[category]
    candidate_type = str(row.get("type") or "").strip()
    if candidate_type in _CANDIDATE_RULE_SOURCES:
        return _CANDIDATE_RULE_SOURCES[candidate_type]
    return default


def _with_rule_metadata(
    row: Dict[str, Any],
    *,
    default_source: str,
) -> Dict[str, Any]:
    enriched = dict(row)
    enriched.setdefault(
        "rule_source",
        _rule_source_for_row(enriched, default=default_source),
    )
    enriched.setdefault("rule_version", KNOWLEDGE_RULE_VERSION)
    return enriched


def _with_rule_metadata_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    default_source: str,
) -> List[Dict[str, Any]]:
    return [_with_rule_metadata(row, default_source=default_source) for row in rows]


def _normalize_coarse_label(label: Any) -> str:
    from src.core.classification.coarse_labels import normalize_coarse_label

    return str(normalize_coarse_label(label) or "")


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

    fit_matches = _extract_fit_codes(search_text)
    for fit_code in fit_matches:
        fit_info = get_fit_info(fit_code)
        if not fit_info:
            continue
        checks.append(
            {
                "category": "fit_validation",
                "item": fit_code,
                "value": {
                    "fit_type": _enum_value(fit_info.get("type")),
                    "fit_class": _enum_value(fit_info.get("class")),
                    "name_zh": fit_info.get("name_zh"),
                    "name_en": fit_info.get("name_en"),
                    "application_zh": fit_info.get("application_zh"),
                },
                "confidence": 0.9,
                "source": "text",
                "status": "ok",
            }
        )
        standards_candidates.append(
            {
                "type": "iso_fit",
                "designation": fit_code,
                "fit_type": _enum_value(fit_info.get("type")),
                "fit_class": _enum_value(fit_info.get("class")),
            }
        )

    surface_designations: List[Dict[str, Any]] = []
    ra_matches = _dedupe(
        match.group(1) for match in SURFACE_RA_RE.finditer(search_text)
    )
    for ra_text in ra_matches:
        try:
            ra_um = float(ra_text)
        except (TypeError, ValueError):
            continue
        grade = suggest_surface_finish(ra_um)
        surface_designations.append(
            {
                "designation": f"Ra {ra_text}",
                "grade": grade.value,
                "ra_um": ra_um,
            }
        )
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
        surface_designations.append(
            {
                "designation": grade.value,
                "grade": grade.value,
                "ra_um": get_ra_value(grade),
            }
        )
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

    if surface_designations and SURFACE_RECOMMENDATION_RE.search(search_text):
        checks.append(
            {
                "category": "surface_finish_recommendation",
                "item": str(surface_designations[0]["designation"]),
                "value": {
                    "recommended_finish": surface_designations[0],
                    "candidate_finishes": surface_designations,
                },
                "confidence": 0.84,
                "source": "text",
                "status": "ok",
            }
        )

    gdt_snippets = _dedupe(
        [match.group(0).strip() for match in GDT_HINT_RE.finditer(search_text)]
        + snippets
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
                        "warn" if datum_validation.get("is_valid", True) else "error"
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

    for substitution in _extract_material_substitutions(snippets):
        source_material = substitution["source_material"]
        target_material = substitution["target_material"]
        checks.append(
            {
                "category": "material_substitution",
                "item": (f"{source_material['raw']} -> " f"{target_material['raw']}"),
                "value": substitution,
                "confidence": 0.84,
                "source": "text",
                "status": "ok",
            }
        )

    process_route = _extract_process_route(search_text)
    if process_route:
        checks.append(
            {
                "category": "machining_process_route",
                "item": " -> ".join(item["label"] for item in process_route),
                "value": {
                    "route": process_route,
                    "processes": [item["process"] for item in process_route],
                },
                "confidence": 0.78,
                "source": "text",
                "status": "ok",
            }
        )

    manufacturability_risks = _extract_manufacturability_risks(
        search_text,
        geometric_features,
    )
    if manufacturability_risks:
        checks.append(
            {
                "category": "manufacturability_risk",
                "item": ",".join(item["code"] for item in manufacturability_risks),
                "value": {
                    "risks": manufacturability_risks,
                    "risk_count": len(manufacturability_risks),
                },
                "confidence": 0.78,
                "source": "text_geometry",
                "status": "warn",
            }
        )
        for risk in manufacturability_risks:
            violations.append(
                {
                    "category": "manufacturability_risk",
                    "severity": risk.get("severity", "warn"),
                    "message": f"{risk.get('code')}: {risk.get('description')}",
                    "source": "dfm",
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
                "coarse_label": _normalize_coarse_label(label),
                "score": round(float(score), 6),
            }
            for label, score in knowledge_hints.items()
            if str(label).strip()
        ),
        key=lambda item: (-item["score"], item["label"]),
    )

    coarse_label = _normalize_coarse_label(coarse_part_type or fine_part_type)
    if coarse_label and ranked_hints:
        best_hint = ranked_hints[0]
        best_hint_coarse = _normalize_coarse_label(best_hint.get("label"))
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
        "knowledge_checks": _with_rule_metadata_rows(
            _dedupe_dict_rows(checks),
            default_source="knowledge_analysis_summary",
        ),
        "violations": _with_rule_metadata_rows(
            _dedupe_dict_rows(violations),
            default_source="knowledge_analysis_summary",
        ),
        "standards_candidates": _with_rule_metadata_rows(
            standards_candidates,
            default_source="knowledge_analysis_summary",
        ),
        "knowledge_hints": _with_rule_metadata_rows(
            ranked_hints[:5],
            default_source="knowledge_manager",
        ),
    }
