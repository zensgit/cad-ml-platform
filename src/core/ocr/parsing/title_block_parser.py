"""Title block parser for common CAD drawing metadata."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Tuple

LABEL_TOKENS: Tuple[str, ...] = (
    "图号",
    "图纸编号",
    "图纸号",
    "drawing\\s*(?:no|number)",
    "dwg\\s*(?:no|number)",
    "版本",
    "修订",
    "rev(?:ision)?",
    "ver(?:sion)?",
    "名称",
    "零件名称",
    "part\\s*name",
    "title",
    "材料",
    "material",
    "比例",
    "scale",
    "页码",
    "页",
    "sheet",
    "日期",
    "date",
    "重量",
    "weight",
    "公司",
    "单位",
    "company",
    "投影",
    "projection",
)

LABEL_PATTERN = "|".join(LABEL_TOKENS)
LABEL_BOUNDARY_REGEXES = (
    re.compile(rf"(?:{LABEL_PATTERN})\s*[:：]", re.IGNORECASE),
    re.compile(r"(?:scale|比例)\s*[0-9]", re.IGNORECASE),
    re.compile(r"(?:sheet|页码|页)\s*[0-9]", re.IGNORECASE),
    re.compile(r"(?:date|日期)\s*[0-9]", re.IGNORECASE),
    re.compile(r"(?:weight|重量)\s*[0-9]", re.IGNORECASE),
    re.compile(r"(?:projection|投影)\s*(?:first|1st|third|3rd|第一角|第三角)", re.IGNORECASE),
    re.compile(r"(?:company|公司|单位)\s*[A-Za-z0-9]", re.IGNORECASE),
)

TITLE_BLOCK_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (
        "drawing_number",
        r"(?:图号|图纸编号|图纸号|drawing\s*(?:no|number)|dwg\s*(?:no|number))[:：\s]*([A-Z0-9][A-Z0-9\-_/]+)",
    ),
    (
        "revision",
        r"(?:版本|修订|rev(?:ision)?|ver(?:sion)?)[:：\s]*([A-Z0-9]+)",
    ),
    (
        "part_name",
        r"(?:名称|零件名称|part\s*name|title)[:：\s]*([^\n,]+)",
    ),
    (
        "material",
        r"(?:材料|material)[:：\s]*([^\n,]+)",
    ),
    (
        "scale",
        r"(?:比例|scale)[:：\s]*([0-9]+\s*[:/\-]\s*[0-9]+|[0-9]+(?:\.[0-9]+)?)",
    ),
    (
        "sheet",
        r"(?:页码|页|sheet)[:：\s]*([0-9]+\s*/\s*[0-9]+|[0-9]+\s*of\s*[0-9]+|[0-9]+)",
    ),
    (
        "date",
        r"(?:日期|date)[:：\s]*([0-9]{4}[-/.][0-9]{1,2}[-/.][0-9]{1,2}|[0-9]{1,2}[-/.][0-9]{1,2}[-/.][0-9]{2,4})",
    ),
    (
        "weight",
        r"(?:重量|weight)[:：\s]*([0-9]+(?:\.[0-9]+)?\s*(?:kg|g|lb|lbs))",
    ),
    (
        "company",
        r"(?:公司|单位|company)[:：\s]*([^\n,]+)",
    ),
    (
        "projection",
        r"(?:投影|projection)[:：\s]*(first|1st|third|3rd|第一角|第三角)",
    ),
)


def _trim_to_next_label(value: str) -> str:
    if not value:
        return value
    earliest = None
    for pattern in LABEL_BOUNDARY_REGEXES:
        match = pattern.search(value)
        if match and match.start() > 0:
            if earliest is None or match.start() < earliest:
                earliest = match.start()
    if earliest is None:
        return value.strip()
    return value[:earliest].strip()


def _normalize_sheet(value: str) -> str:
    if "of" in value.lower():
        parts = re.split(r"\s*of\s*", value, flags=re.IGNORECASE)
        if len(parts) == 2:
            return f"{parts[0].strip()}/{parts[1].strip()}"
    return value


def parse_title_block(text: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not text:
        return values
    for field, pattern in TITLE_BLOCK_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        value = _trim_to_next_label(match.group(1))
        if not value:
            continue
        if field == "sheet":
            value = _normalize_sheet(value)
        values[field] = value
    return values


def parse_title_block_with_confidence(
    ocr_lines: Iterable[Dict[str, Any]],
) -> Tuple[Dict[str, str], Dict[str, float]]:
    values: Dict[str, str] = {}
    confidences: Dict[str, float] = {}
    for line in ocr_lines:
        text = str(line.get("text", ""))
        score = line.get("score")
        if not text:
            continue
        for field, pattern in TITLE_BLOCK_PATTERNS:
            if field in values:
                continue
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue
            value = _trim_to_next_label(match.group(1))
            if not value:
                continue
            if field == "sheet":
                value = _normalize_sheet(value)
            values[field] = value
            if isinstance(score, (int, float)):
                confidences[field] = float(score)
    return values, confidences
