"""Three-level fallback parser extracted from tests (production module).

Levels:
1. Strict JSON
2. Markdown fenced code blocks (```json)
3. Regex text patterns

Enhancements: thread pitch extraction, bidirectional tolerance parsing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from .title_block_parser import parse_title_block


class FallbackLevel(str, Enum):
    JSON_STRICT = "json_strict"
    MARKDOWN_FENCE = "markdown_fence"
    TEXT_REGEX = "text_regex"


@dataclass
class ParseResult:
    success: bool
    data: Optional[Dict[str, Any]]
    fallback_level: FallbackLevel
    error: Optional[str] = None


class FallbackParser:
    def parse(self, raw_output: str) -> ParseResult:
        # Level 1
        result = self._parse_json_strict(raw_output)
        if result.success:
            return result
        # Level 2
        result = self._parse_markdown_fence(raw_output)
        if result.success:
            return result
        # Level 3
        return self._parse_text_regex(raw_output)

    def _parse_json_strict(self, raw_output: str) -> ParseResult:
        try:
            data = json.loads(raw_output)
            if self._validate_json_schema(data):
                return ParseResult(True, data, FallbackLevel.JSON_STRICT)
            return ParseResult(False, None, FallbackLevel.JSON_STRICT, "Schema validation failed")
        except json.JSONDecodeError as e:
            return ParseResult(False, None, FallbackLevel.JSON_STRICT, str(e))

    def _parse_markdown_fence(self, raw_output: str) -> ParseResult:
        # Support ```json / ```JSON / ``` json  (case insensitive with optional spaces)
        pattern = r"```\s*json\s*(.*?)\s*```"
        matches = re.findall(pattern, raw_output, re.IGNORECASE | re.DOTALL)
        for match in matches:
            try:
                data = json.loads(match)
                if self._validate_json_schema(data):
                    return ParseResult(True, data, FallbackLevel.MARKDOWN_FENCE)
            except json.JSONDecodeError:
                continue
        return ParseResult(False, None, FallbackLevel.MARKDOWN_FENCE, "No valid JSON fenced block")

    def _parse_text_regex(self, raw_output: str) -> ParseResult:
        data = {"dimensions": [], "symbols": [], "title_block": {}}

        # Dimension patterns with bidirectional tolerance and thread pitch
        # Φ20±0.02 | Φ20 +0.02 -0.01 | R5 | M10×1.5
        diameter_pattern = r"[Φ⌀∅](\d+\.?\d*)(?:\s*([±+\-]\d+\.?\d*))?(?:\s*([+\-]\d+\.?\d*))?"
        radius_pattern = r"R(\d+\.?\d*)(?:\s*([±+\-]\d+\.?\d*))?(?:\s*([+\-]\d+\.?\d*))?"
        thread_pattern = r"M(\d+)(?:[×x\*](\d+\.?\d*))?"

        for match in re.finditer(diameter_pattern, raw_output):
            value = float(match.group(1))
            tol_primary = match.group(2)
            tol_secondary = match.group(3)
            tolerance = None
            if tol_primary and not tol_primary.startswith("+") and tol_primary.startswith("±"):
                tolerance = float(tol_primary.replace("±", ""))
            elif (
                tol_primary
                and tol_primary.startswith("+")
                and tol_secondary
                and tol_secondary.startswith("-")
            ):
                # dual tolerance +a -b -> take max(abs(a), abs(b)) conservative
                try:
                    tolerance = max(abs(float(tol_primary)), abs(float(tol_secondary)))
                except ValueError:
                    pass
            elif tol_primary and tol_primary.startswith("±") is False and tol_secondary is None:
                # single sided tolerance treat absolute
                try:
                    tolerance = abs(float(tol_primary))
                except ValueError:
                    pass
            dim = {"type": "diameter", "value": value, "unit": "mm"}
            if tolerance is not None:
                dim["tolerance"] = tolerance
            data["dimensions"].append(dim)

        for match in re.finditer(radius_pattern, raw_output):
            value = float(match.group(1))
            tol_primary = match.group(2)
            tol_secondary = match.group(3)
            tolerance = None
            if tol_primary and tol_primary.startswith("±"):
                tolerance = float(tol_primary.replace("±", ""))
            elif tol_primary and tol_secondary:
                try:
                    tolerance = max(abs(float(tol_primary)), abs(float(tol_secondary)))
                except ValueError:
                    pass
            dim = {"type": "radius", "value": value, "unit": "mm"}
            if tolerance is not None:
                dim["tolerance"] = tolerance
            data["dimensions"].append(dim)

        for match in re.finditer(thread_pattern, raw_output):
            major = float(match.group(1))
            pitch = match.group(2)
            dim = {"type": "thread", "value": major, "unit": "mm"}
            if pitch:
                try:
                    dim["pitch"] = float(pitch)
                except ValueError:
                    pass
            data["dimensions"].append(dim)

        # Symbols
        symbol_patterns = [
            (r"Ra(\d+\.?\d*)", "surface_roughness"),
            (r"[⟂⊥]", "perpendicular"),
            (r"[∥‖]", "parallel"),
        ]
        for pattern, sym_type in symbol_patterns:
            for m in re.finditer(pattern, raw_output):
                val = m.group(1) if m.groups() else m.group(0)
                data["symbols"].append({"type": sym_type, "value": val})

        # Title block (Chinese / English)
        parsed_title = parse_title_block(raw_output)
        for field, value in parsed_title.items():
            data["title_block"].setdefault(field, value)

        success = bool(data["dimensions"] or data["symbols"] or data["title_block"])
        return ParseResult(
            success,
            data if success else None,
            FallbackLevel.TEXT_REGEX,
            None if success else "No structured data",
        )

    def _validate_json_schema(self, data: Dict[str, Any]) -> bool:
        # Required keys
        if not isinstance(data, dict):
            return False
        if "dimensions" not in data or "symbols" not in data:
            return False
        if not isinstance(data.get("dimensions"), list) or not isinstance(
            data.get("symbols"), list
        ):
            return False
        # Element structure shallow validation
        for dim in data.get("dimensions", [])[:10]:  # sample first few
            if not isinstance(dim, dict) or "type" not in dim or "value" not in dim:
                return False
        return True
