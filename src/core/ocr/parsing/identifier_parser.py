"""Identifier extraction with lightweight provenance for drawing OCR."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from src.core.ocr.base import IdentifierInfo

from .title_block_parser import parse_title_block, parse_title_block_with_confidence

IDENTIFIER_LABELS: Dict[str, str] = {
    "drawing_number": "Drawing Number",
    "revision": "Revision",
    "part_name": "Part Name",
    "material": "Material",
    "scale": "Scale",
    "sheet": "Sheet",
    "date": "Date",
    "weight": "Weight",
    "company": "Company",
    "projection": "Projection",
}

CAPTION_PATTERNS: Dict[str, re.Pattern[str]] = {
    "drawing_number": re.compile(
        r"^(?:图号|图纸编号|图纸号|图纸代号|图纸代码|零件号|零件编号|"
        r"part\s*(?:no|number)|drawing\s*(?:no|number|id|#)|drawing#|"
        r"dwg\s*(?:no|number|id|#)|dwg#)\s*[:：]?\s*$",
        re.IGNORECASE,
    ),
    "revision": re.compile(
        r"^(?:版本|版本号|修订|rev\.?|rev(?:ision)?|rev\s*(?:no|#)|"
        r"ver(?:sion)?)\s*[:：]?\s*$",
        re.IGNORECASE,
    ),
    "part_name": re.compile(
        r"^(?:名称|零件名称|零件名|part\s*name|part\s*title|title|description|desc)\s*[:：]?\s*$",
        re.IGNORECASE,
    ),
    "material": re.compile(r"^(?:材料|材质|material|mat(?:'l|l)?)\s*[:：]?\s*$", re.IGNORECASE),
    "scale": re.compile(r"^(?:比例|比例尺|scale)\s*[:：]?\s*$", re.IGNORECASE),
    "sheet": re.compile(r"^(?:页码|页|sheet\s*(?:no|#)?|sht\.?)\s*[:：]?\s*$", re.IGNORECASE),
    "date": re.compile(r"^(?:日期|date|drawn\s*date)\s*[:：]?\s*$", re.IGNORECASE),
    "weight": re.compile(r"^(?:重量|weight|wt\.?|mass)\s*[:：]?\s*$", re.IGNORECASE),
    "company": re.compile(r"^(?:公司|单位|company|customer|client)\s*[:：]?\s*$", re.IGNORECASE),
    "projection": re.compile(
        r"^(?:投影|projection|first\s*angle|third\s*angle|1st\s*angle|"
        r"3rd\s*angle|第一角法|第三角法)\s*[:：]?\s*$",
        re.IGNORECASE,
    ),
}


def _normalize_identifier_value(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _normalize_identifier_key(identifier_type: str, value: str) -> str:
    normalized = _normalize_identifier_value(value)
    if identifier_type in {"drawing_number", "revision", "projection"}:
        return normalized.upper()
    return normalized


def _is_value_like(line_text: str) -> bool:
    stripped = line_text.strip()
    return bool(stripped) and not any(
        pattern.match(stripped) for pattern in CAPTION_PATTERNS.values()
    )


def _line_text(line: Dict[str, Any]) -> str:
    return str(line.get("text", "") or "").strip()


def _find_best_evidence_line(
    identifier_type: str,
    value: str,
    ocr_lines: Iterable[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    normalized_value = _normalize_identifier_key(identifier_type, value)
    best_line: Optional[Dict[str, Any]] = None
    best_score = -1
    caption_pattern = CAPTION_PATTERNS.get(identifier_type)

    for line in ocr_lines:
        text = _line_text(line)
        if not text:
            continue
        normalized_text = _normalize_identifier_key(identifier_type, text)
        score = 0
        if normalized_value in normalized_text:
            score += 2
        if normalized_text == normalized_value:
            score += 2
        if caption_pattern and caption_pattern.search(text):
            score += 1
        if score > best_score:
            best_score = score
            best_line = line

    if best_score <= 0:
        return None
    return best_line


def _append_identifier(
    identifiers: List[IdentifierInfo],
    seen: set[tuple[str, str]],
    identifier_type: str,
    value: str,
    *,
    source: str,
    confidence: Optional[float] = None,
    source_text: Optional[str] = None,
    bbox: Optional[List[int]] = None,
) -> None:
    normalized_value = _normalize_identifier_key(identifier_type, value)
    dedupe_key = (identifier_type, normalized_value.upper())
    if dedupe_key in seen:
        return
    seen.add(dedupe_key)
    identifiers.append(
        IdentifierInfo(
            identifier_type=identifier_type,
            label=IDENTIFIER_LABELS.get(identifier_type),
            value=_normalize_identifier_value(value),
            normalized_value=normalized_value,
            source_text=source_text or value,
            bbox=bbox,
            confidence=confidence,
            source=source,
        )
    )


def _extract_split_line_values(ocr_lines: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    values: Dict[str, Dict[str, Any]] = {}
    for index, line in enumerate(ocr_lines[:-1]):
        text = _line_text(line)
        if not text:
            continue
        next_line = ocr_lines[index + 1]
        next_text = _line_text(next_line)
        if not _is_value_like(next_text):
            continue
        for identifier_type, pattern in CAPTION_PATTERNS.items():
            if identifier_type in values:
                continue
            if pattern.match(text):
                values[identifier_type] = {
                    "value": next_text,
                    "confidence": next_line.get("score"),
                    "source_text": f"{text} {next_text}".strip(),
                    "bbox": next_line.get("bbox"),
                }
    return values


def extract_identifiers(
    *,
    text: Optional[str] = None,
    ocr_lines: Optional[List[Dict[str, Any]]] = None,
    title_block_values: Optional[Dict[str, str]] = None,
    field_confidence: Optional[Dict[str, float]] = None,
    default_source: str = "regex_text",
) -> List[IdentifierInfo]:
    identifiers: List[IdentifierInfo] = []
    seen: set[tuple[str, str]] = set()
    field_confidence = field_confidence or {}
    ocr_lines = ocr_lines or []

    values = dict(title_block_values or {})
    if ocr_lines and not values:
        line_values, line_confidence = parse_title_block_with_confidence(ocr_lines)
        values.update(line_values)
        for key, score in line_confidence.items():
            field_confidence.setdefault(key, score)

    split_line_values = _extract_split_line_values(ocr_lines)
    for key, payload in split_line_values.items():
        values.setdefault(key, payload["value"])
        if payload.get("confidence") is not None:
            field_confidence.setdefault(key, float(payload["confidence"]))

    for identifier_type, value in values.items():
        split_line = split_line_values.get(identifier_type)
        evidence_line = _find_best_evidence_line(identifier_type, value, ocr_lines)
        if split_line:
            _append_identifier(
                identifiers,
                seen,
                identifier_type,
                value,
                source="ocr_line" if default_source == "regex_text" else default_source,
                confidence=field_confidence.get(identifier_type) or split_line.get("confidence"),
                source_text=split_line.get("source_text"),
                bbox=split_line.get("bbox"),
            )
        elif evidence_line:
            _append_identifier(
                identifiers,
                seen,
                identifier_type,
                value,
                source="ocr_line" if default_source == "regex_text" else default_source,
                confidence=field_confidence.get(identifier_type) or evidence_line.get("score"),
                source_text=_line_text(evidence_line),
                bbox=evidence_line.get("bbox"),
            )
        else:
            _append_identifier(
                identifiers,
                seen,
                identifier_type,
                value,
                source=default_source,
                confidence=field_confidence.get(identifier_type),
            )

    if text:
        for identifier_type, value in parse_title_block(text).items():
            _append_identifier(
                identifiers,
                seen,
                identifier_type,
                value,
                source="regex_text",
                confidence=field_confidence.get(identifier_type),
            )

    return identifiers
