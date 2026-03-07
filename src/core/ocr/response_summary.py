"""Shared summary builders for OCR and drawing response surfaces."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from src.core.ocr.base import ProcessRequirements, SymbolInfo, SymbolType, TitleBlock

STANDARD_CANDIDATE_PATTERN = re.compile(
    r"(?:GB/T|GB|ISO|DIN|ANSI|ASME|ASTM|JIS|EN)\s*[-/]?\s*[A-Z0-9.\-]+",
    re.IGNORECASE,
)
CRITICAL_TITLE_BLOCK_FIELDS = (
    "drawing_number",
    "part_name",
    "revision",
    "material",
)


def build_field_coverage(
    title_block: TitleBlock,
    tracked_fields: Iterable[str],
) -> Dict[str, Any]:
    tracked_fields = list(tracked_fields)
    recognized_keys = [key for key in tracked_fields if getattr(title_block, key, None)]
    missing_keys = [key for key in tracked_fields if key not in recognized_keys]
    total_fields = len(tracked_fields)
    recognized_count = len(recognized_keys)
    coverage_ratio = (recognized_count / total_fields) if total_fields else 0.0
    return {
        "recognized_count": recognized_count,
        "total_fields": total_fields,
        "coverage_ratio": round(coverage_ratio, 4),
        "recognized_keys": recognized_keys,
        "missing_keys": missing_keys,
    }


def extract_standard_candidates(process_requirements: ProcessRequirements) -> List[str]:
    candidates: List[str] = []
    seen = set()

    def _add(value: str | None) -> None:
        if not value:
            return
        normalized = re.sub(r"\s+", "", value.strip())
        if not normalized:
            return
        dedupe = normalized.upper()
        if dedupe in seen:
            return
        seen.add(dedupe)
        candidates.append(normalized)

    for surface_treatment in process_requirements.surface_treatments:
        _add(surface_treatment.standard)

    notes_blob = "\n".join(process_requirements.general_notes)
    if process_requirements.raw_text:
        notes_blob = f"{notes_blob}\n{process_requirements.raw_text}".strip()
    for match in STANDARD_CANDIDATE_PATTERN.finditer(notes_blob):
        _add(match.group(0))
    return candidates


def build_engineering_signals(
    *,
    title_block: TitleBlock,
    dimensions: List[Dict[str, Any]],
    symbols: List[SymbolInfo],
    process_requirements: ProcessRequirements,
) -> Dict[str, Any]:
    symbol_types = sorted({symbol.type.value for symbol in symbols})
    gdt_symbol_types = sorted(
        {symbol.type.value for symbol in symbols if symbol.type != SymbolType.surface_roughness}
    )
    materials_detected = []
    if title_block.material:
        materials_detected.append(title_block.material)

    return {
        "dimension_count": len(dimensions),
        "symbol_count": len(symbols),
        "symbol_types": symbol_types,
        "gdt_symbol_types": gdt_symbol_types,
        "has_surface_finish": SymbolType.surface_roughness.value in symbol_types,
        "has_gdt": bool(gdt_symbol_types),
        "process_requirement_counts": {
            "heat_treatments": len(process_requirements.heat_treatments),
            "surface_treatments": len(process_requirements.surface_treatments),
            "welding": len(process_requirements.welding),
            "general_notes": len(process_requirements.general_notes),
        },
        "materials_detected": materials_detected,
        "standards_candidates": extract_standard_candidates(process_requirements),
    }


def build_review_hints(
    *,
    title_block: TitleBlock,
    identifiers: List[Dict[str, Any]] | List[Any],
    field_coverage: Dict[str, Any],
    engineering_signals: Dict[str, Any],
) -> Dict[str, Any]:
    recognized_keys = set(field_coverage.get("recognized_keys", []))
    present_critical_fields = [
        field for field in CRITICAL_TITLE_BLOCK_FIELDS if getattr(title_block, field, None)
    ]
    missing_critical_fields = [
        field for field in CRITICAL_TITLE_BLOCK_FIELDS if field not in recognized_keys
    ]

    has_identifiers = bool(identifiers)
    has_dimensions = bool(engineering_signals.get("dimension_count", 0))
    has_symbols = bool(engineering_signals.get("symbol_count", 0))
    process_counts = engineering_signals.get("process_requirement_counts", {})
    has_process_requirements = any(process_counts.values()) if process_counts else False
    has_standards_candidates = bool(engineering_signals.get("standards_candidates", []))
    has_materials = bool(engineering_signals.get("materials_detected", []))

    readiness_score = (
        float(field_coverage.get("coverage_ratio", 0.0)) * 0.45
        + (0.2 if has_identifiers else 0.0)
        + (0.15 if has_dimensions else 0.0)
        + (0.1 if has_symbols else 0.0)
        + (0.1 if (has_process_requirements or has_standards_candidates or has_materials) else 0.0)
    )
    readiness_score = round(min(readiness_score, 1.0), 4)
    if readiness_score >= 0.75:
        readiness_band = "high"
    elif readiness_score >= 0.45:
        readiness_band = "medium"
    else:
        readiness_band = "low"

    review_reasons: List[str] = []
    if missing_critical_fields:
        review_reasons.append("missing_critical_fields")
    if not has_identifiers:
        review_reasons.append("no_identifiers")
    if not has_dimensions:
        review_reasons.append("no_dimensions")
    if not (has_process_requirements or has_standards_candidates or has_materials):
        review_reasons.append("limited_engineering_context")
    if readiness_band == "low":
        review_reasons.append("low_readiness_score")

    return {
        "critical_fields": list(CRITICAL_TITLE_BLOCK_FIELDS),
        "present_critical_fields": present_critical_fields,
        "missing_critical_fields": missing_critical_fields,
        "has_identifiers": has_identifiers,
        "has_dimensions": has_dimensions,
        "has_symbols": has_symbols,
        "has_process_requirements": has_process_requirements,
        "has_standards_candidates": has_standards_candidates,
        "review_recommended": bool(review_reasons),
        "review_reasons": review_reasons,
        "readiness_score": readiness_score,
        "readiness_band": readiness_band,
    }
