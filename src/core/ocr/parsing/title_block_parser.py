"""Title block parser for common CAD drawing metadata."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Tuple

LABEL_TOKENS: Tuple[str, ...] = (
    "图号",
    "图纸编号",
    "图纸号",
    "图纸代号",
    "图纸代码",
    "零件号",
    "零件编号",
    "drawing\\s*(?:no|number)",
    "drawing\\s*(?:id|#)",
    "dwg\\s*(?:no|number)",
    "dwg\\s*(?:id|#)",
    "dwg#",
    "drawing#",
    "part\\s*(?:no|number)",
    "版本",
    "版本号",
    "修订",
    "rev\\.?",
    "rev(?:ision)?",
    "rev\\s*(?:no|#)",
    "ver(?:sion)?",
    "名称",
    "零件名称",
    "零件名",
    "part\\s*name",
    "part\\s*title",
    "title",
    "description",
    "desc",
    "材料",
    "材质",
    "material",
    "mat(?:'l|l)?",
    "比例",
    "比例尺",
    "scale",
    "页码",
    "页",
    "sheet",
    "sheet\\s*(?:no|#)",
    "sht\\.?",
    "日期",
    "date",
    "drawn\\s*date",
    "重量",
    "weight",
    "wt\\.?",
    "mass",
    "公司",
    "单位",
    "company",
    "customer",
    "client",
    "投影",
    "projection",
    "first\\s*angle",
    "third\\s*angle",
    "1st\\s*angle",
    "3rd\\s*angle",
    "第一角法",
    "第三角法",
)

LABEL_PATTERN = "|".join(LABEL_TOKENS)
LABEL_BOUNDARY_REGEXES = (
    re.compile(rf"(?:{LABEL_PATTERN})\s*[:：]", re.IGNORECASE),
    re.compile(
        r"(?:scale|比例|比例尺)\s*[0-9]",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:sheet|页码|页|sht\.?)\s*[0-9]",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:date|日期|drawn\s*date)\s*[0-9]",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:weight|重量|wt\.?|mass)\s*[0-9]",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:projection|投影|first\s*angle|third\s*angle|1st\s*angle|3rd\s*angle)\s*"
        r"(?:first|1st|third|3rd|第一角|第三角)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:company|公司|单位|customer|client)\s*[A-Za-z0-9]",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:rev\.?|revision|ver\.?|version|版本|修订)\s*[A-Za-z0-9]",
        re.IGNORECASE,
    ),
)

TITLE_BLOCK_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (
        "drawing_number",
        r"(?:图号|图纸编号|图纸号|图纸代号|图纸代码|零件号|零件编号|"
        r"part\s*(?:no|number)|drawing\s*(?:no|number|id|#)|drawing#|"
        r"dwg\s*(?:no|number|id|#)|dwg#)[:：\s]*"
        r"([A-Z0-9][A-Z0-9\-_/\.]+)",
    ),
    (
        "revision",
        r"(?:版本|版本号|修订|rev\.?|rev(?:ision)?|rev\s*(?:no|#)|"
        r"ver(?:sion)?)[:：\s]*([A-Z0-9]+)",
    ),
    (
        "part_name",
        r"(?:名称|零件名称|零件名|part\s*name|part\s*title|title|description|desc)[:：\s]*([^\n,]+)",
    ),
    (
        "material",
        r"(?:材料|材质|material|mat(?:'l|l)?)[:：\s]*([^\n,]+)",
    ),
    (
        "scale",
        r"(?:比例|比例尺|scale)[:：\s]*"
        r"([0-9]+\s*[:/\-]\s*[0-9]+|[0-9]+(?:\.[0-9]+)?|"
        r"N\.?\s*T\.?\s*S\.?|not\s*to\s*scale)",
    ),
    (
        "sheet",
        r"(?:页码|页|sheet\s*(?:no|#)?|sht\.?)[:：\s]*"
        r"([0-9]+\s*/\s*[0-9]+|[0-9]+\s*of\s*[0-9]+|[0-9]+)",
    ),
    (
        "date",
        r"(?:日期|date|drawn\s*date)[:：\s]*"
        r"([0-9]{4}[-/.][0-9]{1,2}[-/.][0-9]{1,2}|"
        r"[0-9]{1,2}[-/.][0-9]{1,2}[-/.][0-9]{2,4})",
    ),
    (
        "weight",
        r"(?:重量|weight|wt\.?|mass)[:：\s]*([0-9]+(?:\.[0-9]+)?\s*(?:kg|g|lb|lbs))",
    ),
    (
        "company",
        r"(?:公司|单位|company|customer|client)[:：\s]*([^\n,]+)",
    ),
    (
        "projection",
        r"(?:投影|projection|first\s*angle|third\s*angle|1st\s*angle|3rd\s*angle|"
        r"第一角法|第三角法)[:：\s]*(first|1st|third|3rd|第一角|第三角)",
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


def _normalize_scale(value: str) -> str:
    normalized = value.strip()
    if re.fullmatch(r"n\.?\s*t\.?\s*s\.?", normalized, re.IGNORECASE):
        return "NTS"
    if re.fullmatch(r"not\s*to\s*scale", normalized, re.IGNORECASE):
        return "NTS"
    normalized = re.sub(r"\s*([:/-])\s*", r"\1", normalized)
    return normalized


def _normalize_projection(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"first", "1st", "第一角"}:
        return "first"
    if normalized in {"third", "3rd", "第三角"}:
        return "third"
    return value.strip()


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
        elif field == "scale":
            value = _normalize_scale(value)
        elif field == "projection":
            value = _normalize_projection(value)
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
            elif field == "scale":
                value = _normalize_scale(value)
            elif field == "projection":
                value = _normalize_projection(value)
            values[field] = value
            if isinstance(score, (int, float)):
                confidences[field] = float(score)
    return values, confidences
