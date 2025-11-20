"""Dimension & symbol parser using regex patterns.

Parses CAD drawing textual OCR output to extract:
 - Diameter (Φ / ⌀ / ∅)
 - Radius (R)
 - Thread (M<number>x<pitch>)
 - Surface roughness (Ra)
 - Dual tolerance (e.g. +0.02 -0.01) attached to preceding dimension

Normalization rules:
 - Units default to mm; cm -> mm; μm -> mm (Ra stays numeric value, unit tracked separately if needed)
 - Thread pitch optional.

The parser is heuristic: it scans text left-to-right, assigns tolerances to the last captured dimension token.
"""

from __future__ import annotations

import re
from typing import List

from src.core.ocr.base import DimensionInfo, DimensionType, SymbolInfo, SymbolType

# Unit token without internal optional to avoid consuming long whitespace when unit absent
_UNIT_RE = r"(mm|MM|cm|CM|μm|um)"
_TOL_TOKEN = r"([±+＋\-]\s*\d+(?:\.\d+)?)"
DIAMETER_PATTERN = re.compile(
    rf"[Φ⌀∅]\s*(\d+(?:\.\d+)?)(?:\s*{_UNIT_RE})?(?:\s{{0,12}}{_TOL_TOKEN})?",
    re.UNICODE,
)
RADIUS_PATTERN = re.compile(
    rf"R\s*(\d+(?:\.\d+)?)(?:\s*{_UNIT_RE})?(?:\s{{0,12}}{_TOL_TOKEN})?",
    re.UNICODE,
)
# Thread pattern supports separators × x X * and optional pitch.
THREAD_PATTERN = re.compile(r"M(\d+)(?:[×xX\*](\d+(?:\.\d+)?))?", re.UNICODE)
ROUGHNESS_PATTERN = re.compile(r"Ra\s*(\d+(?:\.\d+)?)(?:\s*(μm|um|MM|mm|CM|cm))?", re.UNICODE)
DUAL_TOL_PATTERN = re.compile(r"([+＋]\s*\d+(?:\.\d+)?)[\s,]+([-－]\s*\d+(?:\.\d+)?)", re.UNICODE)

# Geometric feature symbols (subset)
PERP_PATTERN = re.compile(r"⊥")
PARA_PATTERN = re.compile(r"∥")
# Additional GD&T tokens (ASCII proxies, heuristic)
FLATNESS_TOK = re.compile(r"\bflatness\b|⏤")
STRAIGHTNESS_TOK = re.compile(r"\bstraightness\b")
CIRCULARITY_TOK = re.compile(r"\bcircularity\b")
CYLINDRICITY_TOK = re.compile(r"\bcylindricity\b")
POSITION_TOK = re.compile(r"\bposition\b")
CONCENTRICITY_TOK = re.compile(r"\bconcentricity\b")
SYMMETRY_TOK = re.compile(r"\bsymmetry\b")
RUNOUT_TOK = re.compile(r"\brunout\b")
TOTAL_RUNOUT_TOK = re.compile(r"\btotal\s*runout\b")
PROFILE_LINE_TOK = re.compile(r"\bprofile\s*of\s*a\s*line\b")
PROFILE_SURF_TOK = re.compile(r"\bprofile\s*of\s*a\s*surface\b")


def _norm_number(token: str) -> float:
    return float(token.replace("＋", "+").replace("－", "-").replace(" ", ""))


def _to_mm(value: float, unit: str | None) -> float:
    if not unit:
        return value
    u = unit.lower()
    if u == "mm":
        return value
    if u == "cm":
        return value * 10.0
    if u in ("μm", "um"):
        return value / 1000.0
    return value


def _extract_dual_tolerance(text: str) -> List[tuple[float, float]]:
    pairs = []
    for m in DUAL_TOL_PATTERN.finditer(text):
        pos = m.group(1)
        neg = m.group(2)
        try:
            pos_v = abs(_norm_number(pos))
            neg_v = abs(_norm_number(neg))
            pairs.append((pos_v, neg_v))
        except Exception:
            continue
    return pairs


def parse_dimensions_and_symbols(text: str) -> tuple[List[DimensionInfo], List[SymbolInfo]]:
    dimensions: List[DimensionInfo] = []
    symbols: List[SymbolInfo] = []
    spans: List[tuple[int, int, DimensionInfo]] = []  # track token spans for tolerance binding

    # Diameter
    for m in DIAMETER_PATTERN.finditer(text):
        val_raw = float(m.group(1))
        unit = m.group(2)
        tol_token = m.group(3)
        val = _to_mm(val_raw, unit)
        tol = None
        tol_pos = None
        tol_neg = None
        # Ignore inline tolerance if it is too far from the value (likely belongs to later token)
        if tol_token:
            try:
                tol_start = m.start(3)
                anchor = m.end(2) if m.group(2) else m.end(1)
                if tol_start - anchor > 24:
                    tol_token = None
            except Exception:
                pass
        if tol_token:
            # normalize '±a' or '+a' or '-a' -> use absolute numeric component
            cleaned = tol_token.replace("±", "").replace("＋", "+").replace("－", "-")
            cleaned = cleaned.replace(" ", "")
            cleaned = cleaned.lstrip("+")
            try:
                if "±" in tol_token:
                    tol = abs(float(cleaned))
                elif tol_token.strip().startswith("+"):
                    tol_pos = abs(float(cleaned))
                    tol = tol_pos
                elif tol_token.strip().startswith("-"):
                    tol_neg = abs(float(cleaned))
                    tol = tol_neg
                else:
                    tol = abs(float(cleaned))
            except Exception:
                pass
        dim_obj = DimensionInfo(
            type=DimensionType.diameter, value=val, tolerance=tol, raw=m.group(0)
        )
        if tol_pos and not tol_neg:
            dim_obj.tol_pos = tol_pos
        if tol_neg and not tol_pos:
            dim_obj.tol_neg = tol_neg
        dimensions.append(dim_obj)
        spans.append((m.start(), m.end(), dim_obj))

    # Radius
    for m in RADIUS_PATTERN.finditer(text):
        val_raw = float(m.group(1))
        unit = m.group(2)
        tol_token = m.group(3)
        val = _to_mm(val_raw, unit)
        tol = None
        tol_pos = None
        tol_neg = None
        # Ignore inline tolerance if it is too far from the value
        if tol_token:
            try:
                tol_start = m.start(3)
                anchor = m.end(2) if m.group(2) else m.end(1)
                if tol_start - anchor > 24:
                    tol_token = None
            except Exception:
                pass
        if tol_token:
            cleaned = (
                tol_token.replace("±", "")
                .replace("＋", "+")
                .replace("－", "-")
                .replace(" ", "")
                .lstrip("+")
            )
            try:
                if "±" in tol_token:
                    tol = abs(float(cleaned))
                elif tol_token.strip().startswith("+"):
                    tol_pos = abs(float(cleaned))
                    tol = tol_pos
                elif tol_token.strip().startswith("-"):
                    tol_neg = abs(float(cleaned))
                    tol = tol_neg
                else:
                    tol = abs(float(cleaned))
            except Exception:
                pass
        dim_obj = DimensionInfo(type=DimensionType.radius, value=val, tolerance=tol, raw=m.group(0))
        if tol_pos:
            dim_obj.tol_pos = tol_pos
        if tol_neg:
            dim_obj.tol_neg = tol_neg
        dimensions.append(dim_obj)
        spans.append((m.start(), m.end(), dim_obj))

    # Thread
    for m in THREAD_PATTERN.finditer(text):
        major = float(m.group(1))
        pitch = m.group(2)
        pitch_val = float(pitch) if pitch else None
        dim_obj = DimensionInfo(
            type=DimensionType.thread, value=major, pitch=pitch_val, raw=m.group(0)
        )
        dimensions.append(dim_obj)
        spans.append((m.start(), m.end(), dim_obj))

    # Surface roughness
    for m in ROUGHNESS_PATTERN.finditer(text):
        val = m.group(1)
        symbols.append(
            SymbolInfo(type=SymbolType.surface_roughness, value=str(val), raw=m.group(0))
        )

    # Geometric symbols
    for m in PERP_PATTERN.finditer(text):
        symbols.append(
            SymbolInfo(
                type=SymbolType.perpendicular,
                value="⊥",
                normalized_form="perpendicular",
                raw=m.group(0),
            )
        )
    for m in PARA_PATTERN.finditer(text):
        symbols.append(
            SymbolInfo(
                type=SymbolType.parallel, value="∥", normalized_form="parallel", raw=m.group(0)
            )
        )
    # ASCII tokens (GD&T proxies)
    if FLATNESS_TOK.search(text):
        symbols.append(
            SymbolInfo(
                type=SymbolType.flatness,
                value="flatness",
                normalized_form="flatness",
                raw="flatness",
            )
        )
    if STRAIGHTNESS_TOK.search(text):
        symbols.append(
            SymbolInfo(
                type=SymbolType.straightness,
                value="straightness",
                normalized_form="straightness",
                raw="straightness",
            )
        )
    if CIRCULARITY_TOK.search(text):
        symbols.append(
            SymbolInfo(
                type=SymbolType.circularity,
                value="circularity",
                normalized_form="circularity",
                raw="circularity",
            )
        )
    if CYLINDRICITY_TOK.search(text):
        symbols.append(
            SymbolInfo(
                type=SymbolType.cylindricity,
                value="cylindricity",
                normalized_form="cylindricity",
                raw="cylindricity",
            )
        )
    if POSITION_TOK.search(text):
        symbols.append(
            SymbolInfo(
                type=SymbolType.position,
                value="position",
                normalized_form="position",
                raw="position",
            )
        )
    if CONCENTRICITY_TOK.search(text):
        symbols.append(
            SymbolInfo(
                type=SymbolType.concentricity,
                value="concentricity",
                normalized_form="concentricity",
                raw="concentricity",
            )
        )
    if SYMMETRY_TOK.search(text):
        symbols.append(
            SymbolInfo(
                type=SymbolType.symmetry,
                value="symmetry",
                normalized_form="symmetry",
                raw="symmetry",
            )
        )
    if TOTAL_RUNOUT_TOK.search(text):
        symbols.append(
            SymbolInfo(
                type=SymbolType.total_runout,
                value="total_runout",
                normalized_form="total_runout",
                raw="total runout",
            )
        )
    elif RUNOUT_TOK.search(text):
        symbols.append(
            SymbolInfo(
                type=SymbolType.runout, value="runout", normalized_form="runout", raw="runout"
            )
        )
    if PROFILE_LINE_TOK.search(text):
        symbols.append(
            SymbolInfo(
                type=SymbolType.profile_line,
                value="profile_of_a_line",
                normalized_form="profile_line",
                raw="profile of a line",
            )
        )
    if PROFILE_SURF_TOK.search(text):
        symbols.append(
            SymbolInfo(
                type=SymbolType.profile_surface,
                value="profile_of_a_surface",
                normalized_form="profile_surface",
                raw="profile of a surface",
            )
        )

    # Dual tolerance assignment: attach first pair to last dimension without tolerance
    # Attach dual tolerance pairs to nearest preceding dimension within a small gap threshold
    gap_threshold = 24  # allow wider separation (spaces / OCR noise)
    for m in DUAL_TOL_PATTERN.finditer(text):
        pos_tok = m.group(1)
        neg_tok = m.group(2)
        try:
            pos_v = abs(_norm_number(pos_tok))
            neg_v = abs(_norm_number(neg_tok))
        except Exception:
            continue
        # choose closest preceding dimension without existing dual assignment
        candidates = []
        for start, end, dim in spans:
            # allow overlap: dual tolerance starts inside raw token span (e.g. diameter consumed '+')
            if start <= m.start() <= end:
                candidates.append((0, dim))
            elif end <= m.start() and (m.start() - end) <= gap_threshold:
                candidates.append((m.start() - end, dim))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            _, candidate = candidates[0]
            # merge if only one-sided tolerance existed
            if candidate.tol_pos and not candidate.tol_neg:
                candidate.tol_neg = neg_v
                candidate.tolerance = max(candidate.tol_pos, candidate.tol_neg)
            elif candidate.tol_neg and not candidate.tol_pos:
                candidate.tol_pos = pos_v
                candidate.tolerance = max(candidate.tol_pos, candidate.tol_neg)
            elif candidate.tol_pos is None and candidate.tol_neg is None:
                candidate.tol_pos = pos_v
                candidate.tol_neg = neg_v
                candidate.tolerance = max(pos_v, neg_v)

    return dimensions, symbols
