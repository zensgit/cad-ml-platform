"""Process requirements parser for CAD drawings.

Extracts manufacturing process information from OCR text:
- Heat treatments (淬火, 回火, 渗碳, etc.)
- Surface treatments (电镀, 镀锌, 阳极氧化, etc.)
- Welding requirements (焊接符号, 焊材, etc.)
- General technical notes

Uses regex patterns optimized for Chinese mechanical drawing conventions.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from src.core.ocr.base import (
    HeatTreatmentInfo,
    HeatTreatmentType,
    ProcessRequirements,
    SurfaceTreatmentInfo,
    SurfaceTreatmentType,
    WeldingInfo,
    WeldingType,
)

# ============================================================================
# Heat Treatment Patterns (热处理)
# ============================================================================

# Hardness patterns: HRC58-62, HB200-250, HV500
HARDNESS_PATTERN = re.compile(
    r"(HRC|HB|HV|HRA)\s*(\d+(?:\.\d+)?)\s*[-~至到]\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
HARDNESS_SINGLE_PATTERN = re.compile(
    r"(HRC|HB|HV|HRA)\s*[≥>=]?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Depth patterns: 渗碳深度0.8-1.2mm, 淬硬层深度≥0.5
DEPTH_PATTERN = re.compile(
    r"(?:渗碳|渗氮|淬硬|硬化|有效硬化)?层?深[度]?\s*[≥>=]?\s*(\d+(?:\.\d+)?)\s*[-~至到]?\s*(\d+(?:\.\d+)?)?\s*(?:mm|MM)?",
    re.UNICODE,
)

# Heat treatment type mapping
HEAT_TREATMENT_PATTERNS = {
    HeatTreatmentType.quenching: re.compile(r"淬火|淬硬|quench", re.IGNORECASE | re.UNICODE),
    HeatTreatmentType.tempering: re.compile(r"回火|temper", re.IGNORECASE | re.UNICODE),
    HeatTreatmentType.annealing: re.compile(r"退火|anneal", re.IGNORECASE | re.UNICODE),
    HeatTreatmentType.normalizing: re.compile(r"正火|normaliz", re.IGNORECASE | re.UNICODE),
    HeatTreatmentType.carburizing: re.compile(r"渗碳|carburiz", re.IGNORECASE | re.UNICODE),
    HeatTreatmentType.nitriding: re.compile(r"渗氮|氮化|nitrid", re.IGNORECASE | re.UNICODE),
    HeatTreatmentType.induction_hardening: re.compile(r"感应淬火|高频淬火|中频淬火|induction", re.IGNORECASE | re.UNICODE),
    HeatTreatmentType.flame_hardening: re.compile(r"火焰淬火|flame", re.IGNORECASE | re.UNICODE),
    HeatTreatmentType.stress_relief: re.compile(r"去应力|消除应力|stress.?relief", re.IGNORECASE | re.UNICODE),
    HeatTreatmentType.aging: re.compile(r"时效|aging", re.IGNORECASE | re.UNICODE),
    HeatTreatmentType.quench_temper: re.compile(r"调质", re.UNICODE),
    HeatTreatmentType.solution_treatment: re.compile(r"固溶|solution.?treat", re.IGNORECASE | re.UNICODE),
    HeatTreatmentType.general_heat_treatment: re.compile(r"热处理(?!类型)", re.UNICODE),  # 通用热处理，排除"热处理类型"
}

# ============================================================================
# Surface Treatment Patterns (表面处理)
# ============================================================================

# Thickness patterns: 镀层厚度8-12μm, 涂层≥30μm
COATING_THICKNESS_PATTERN = re.compile(
    r"(?:镀层|涂层|膜层)?厚[度]?\s*[≥>=]?\s*(\d+(?:\.\d+)?)\s*[-~至到]?\s*(\d+(?:\.\d+)?)?\s*(?:μm|um|微米|mm)?",
    re.UNICODE,
)

SURFACE_TREATMENT_PATTERNS = {
    SurfaceTreatmentType.electroplating: re.compile(r"电镀|electroplat", re.IGNORECASE | re.UNICODE),
    SurfaceTreatmentType.galvanizing: re.compile(r"镀锌|热镀锌|冷镀锌|galvaniz", re.IGNORECASE | re.UNICODE),
    SurfaceTreatmentType.chromating: re.compile(r"镀铬|镀硬铬|装饰铬|chrom(?:e|ium)", re.IGNORECASE | re.UNICODE),
    SurfaceTreatmentType.nickel_plating: re.compile(r"镀镍|化学镍|nickel", re.IGNORECASE | re.UNICODE),
    SurfaceTreatmentType.anodizing: re.compile(r"阳极氧化|硬质阳极|anodiz", re.IGNORECASE | re.UNICODE),
    SurfaceTreatmentType.phosphating: re.compile(r"磷化|phosphat", re.IGNORECASE | re.UNICODE),
    SurfaceTreatmentType.blackening: re.compile(r"发黑|氧化发黑|黑色氧化|blacken", re.IGNORECASE | re.UNICODE),
    SurfaceTreatmentType.painting: re.compile(r"喷漆|油漆|涂漆|paint", re.IGNORECASE | re.UNICODE),
    SurfaceTreatmentType.powder_coating: re.compile(r"粉末喷涂|静电喷涂|powder.?coat", re.IGNORECASE | re.UNICODE),
    SurfaceTreatmentType.polishing: re.compile(r"抛光|研磨|polish", re.IGNORECASE | re.UNICODE),
    SurfaceTreatmentType.sandblasting: re.compile(r"喷砂|喷丸|sand.?blast|shot.?blast", re.IGNORECASE | re.UNICODE),
    SurfaceTreatmentType.passivation: re.compile(r"钝化|passivat", re.IGNORECASE | re.UNICODE),
}

# ============================================================================
# Welding Patterns (焊接)
# ============================================================================

# Weld leg size: 焊脚6, 焊缝高度8mm
WELD_LEG_PATTERN = re.compile(
    r"(?:焊脚|焊缝高度|leg)\s*[≥>=]?\s*(\d+(?:\.\d+)?)\s*(?:mm|MM)?",
    re.IGNORECASE | re.UNICODE,
)

WELDING_PATTERNS = {
    WeldingType.arc_welding: re.compile(r"电弧焊|手工电弧|arc.?weld", re.IGNORECASE | re.UNICODE),
    WeldingType.mig_welding: re.compile(r"MIG焊|熔化极|GMAW", re.IGNORECASE | re.UNICODE),
    WeldingType.tig_welding: re.compile(r"TIG焊|氩弧焊|GTAW", re.IGNORECASE | re.UNICODE),
    WeldingType.spot_welding: re.compile(r"点焊|spot.?weld", re.IGNORECASE | re.UNICODE),
    WeldingType.seam_welding: re.compile(r"缝焊|seam.?weld", re.IGNORECASE | re.UNICODE),
    WeldingType.laser_welding: re.compile(r"激光焊|laser.?weld", re.IGNORECASE | re.UNICODE),
    WeldingType.electron_beam: re.compile(r"电子束焊|electron.?beam", re.IGNORECASE | re.UNICODE),
    WeldingType.brazing: re.compile(r"钎焊|硬钎焊|braz", re.IGNORECASE | re.UNICODE),
    WeldingType.soldering: re.compile(r"软钎焊|锡焊|solder", re.IGNORECASE | re.UNICODE),
}

# Filler material: 焊丝ER50-6, 焊条E4303
FILLER_PATTERN = re.compile(
    r"(?:焊丝|焊条|焊材)[:\s]*([A-Z]{1,3}\d+[-]?\d*)",
    re.IGNORECASE | re.UNICODE,
)

# ============================================================================
# General Technical Notes
# ============================================================================

GENERAL_NOTE_PATTERNS = [
    re.compile(r"未注公差.{0,10}[GB/T\d]+", re.UNICODE),
    re.compile(r"未注圆角[R]?\s*\d+", re.UNICODE),
    re.compile(r"未注倒角[C]?\s*\d+", re.UNICODE),
    re.compile(r"去毛刺|去锐边|倒钝", re.UNICODE),
    re.compile(r"不允许[有]?裂纹|无裂纹", re.UNICODE),
    re.compile(r"表面不[得允许]?有[划伤碰伤锈蚀]", re.UNICODE),
    re.compile(r"调质处理", re.UNICODE),
]


def _extract_hardness(text: str) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[str]]:
    """Extract hardness requirements from text."""
    # Try range pattern first: HRC58-62
    match = HARDNESS_PATTERN.search(text)
    if match:
        unit = match.group(1).upper()
        min_val = float(match.group(2))
        max_val = float(match.group(3))
        raw = match.group(0)
        return raw, min_val, max_val, unit

    # Try single value pattern: HRC≥58
    match = HARDNESS_SINGLE_PATTERN.search(text)
    if match:
        unit = match.group(1).upper()
        val = float(match.group(2))
        raw = match.group(0)
        return raw, val, None, unit

    return None, None, None, None


def _extract_depth(text: str) -> Optional[float]:
    """Extract treatment depth from text."""
    match = DEPTH_PATTERN.search(text)
    if match:
        val1 = float(match.group(1))
        val2 = match.group(2)
        if val2:
            # Return average of range
            return (val1 + float(val2)) / 2
        return val1
    return None


def _extract_coating_thickness(text: str) -> Optional[float]:
    """Extract coating thickness from text."""
    match = COATING_THICKNESS_PATTERN.search(text)
    if match:
        val1 = float(match.group(1))
        val2 = match.group(2)
        if val2:
            return (val1 + float(val2)) / 2
        return val1
    return None


def _extract_weld_leg(text: str) -> Optional[float]:
    """Extract weld leg size from text."""
    match = WELD_LEG_PATTERN.search(text)
    if match:
        return float(match.group(1))
    return None


def _extract_filler(text: str) -> Optional[str]:
    """Extract filler material from text."""
    match = FILLER_PATTERN.search(text)
    if match:
        return match.group(1)
    return None


def parse_process_requirements(text: str) -> ProcessRequirements:
    """
    Parse process requirements from OCR text.

    Args:
        text: Raw OCR text from drawing

    Returns:
        ProcessRequirements with extracted heat treatments, surface treatments,
        welding info, and general notes.
    """
    if not text:
        return ProcessRequirements()

    heat_treatments: List[HeatTreatmentInfo] = []
    surface_treatments: List[SurfaceTreatmentInfo] = []
    welding: List[WeldingInfo] = []
    general_notes: List[str] = []

    # Split text into sentences/lines for context
    lines = re.split(r"[。\n;；]", text)

    # Extract heat treatments
    for ht_type, pattern in HEAT_TREATMENT_PATTERNS.items():
        for line in lines:
            if pattern.search(line):
                hardness_raw, h_min, h_max, h_unit = _extract_hardness(line)
                depth = _extract_depth(line)

                ht_info = HeatTreatmentInfo(
                    type=ht_type,
                    hardness=hardness_raw,
                    hardness_min=h_min,
                    hardness_max=h_max,
                    hardness_unit=h_unit,
                    depth=depth,
                    raw=line.strip(),
                    confidence=0.8 if hardness_raw else 0.6,
                )
                # Avoid duplicates
                if not any(h.type == ht_type and h.raw == ht_info.raw for h in heat_treatments):
                    heat_treatments.append(ht_info)

    # Extract surface treatments
    for st_type, pattern in SURFACE_TREATMENT_PATTERNS.items():
        for line in lines:
            if pattern.search(line):
                thickness = _extract_coating_thickness(line)

                st_info = SurfaceTreatmentInfo(
                    type=st_type,
                    thickness=thickness,
                    raw=line.strip(),
                    confidence=0.8 if thickness else 0.6,
                )
                if not any(s.type == st_type and s.raw == st_info.raw for s in surface_treatments):
                    surface_treatments.append(st_info)

    # Extract welding info
    for weld_type, pattern in WELDING_PATTERNS.items():
        for line in lines:
            if pattern.search(line):
                leg_size = _extract_weld_leg(line)
                filler = _extract_filler(line)

                weld_info = WeldingInfo(
                    type=weld_type,
                    leg_size=leg_size,
                    filler_material=filler,
                    raw=line.strip(),
                    confidence=0.7,
                )
                if not any(w.type == weld_type and w.raw == weld_info.raw for w in welding):
                    welding.append(weld_info)

    # Extract general notes
    for note_pattern in GENERAL_NOTE_PATTERNS:
        for line in lines:
            match = note_pattern.search(line)
            if match:
                note = match.group(0).strip()
                if note and note not in general_notes:
                    general_notes.append(note)

    return ProcessRequirements(
        heat_treatments=heat_treatments,
        surface_treatments=surface_treatments,
        welding=welding,
        general_notes=general_notes,
        raw_text=text if heat_treatments or surface_treatments or welding else None,
    )
