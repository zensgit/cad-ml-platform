"""Shared summary builders for OCR and drawing response surfaces."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from src.core.ocr.base import ProcessRequirements, SymbolInfo, SymbolType, TitleBlock

STANDARD_CANDIDATE_PATTERN = re.compile(
    r"(?:GB/T|GB|ISO|DIN|ANSI|ASME|ASTM|JIS|EN)\s*[-/]?\s*[A-Z0-9.\-]+",
    re.IGNORECASE,
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
