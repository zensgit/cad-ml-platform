"""Deterministic evidence extraction for assistant responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .knowledge_retriever import RetrievalResult, RetrievalSource


@dataclass(frozen=True)
class AssistantEvidence:
    """Structured evidence item derived from a retrieval result."""

    reference_id: str
    source: str
    summary: str
    relevance: float
    match_type: str
    key_facts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the evidence item for API responses."""
        return {
            "reference_id": self.reference_id,
            "source": self.source,
            "summary": self.summary,
            "relevance": self.relevance,
            "match_type": self.match_type,
            "key_facts": list(self.key_facts),
        }


def build_assistant_evidence(
    results: Sequence[RetrievalResult],
    max_facts: int = 4,
) -> List[AssistantEvidence]:
    """Convert retrieval results into a stable evidence list."""
    ordered_results = sorted(results, key=_evidence_sort_key)
    evidence: List[AssistantEvidence] = []

    for index, result in enumerate(ordered_results, start=1):
        evidence.append(
            AssistantEvidence(
                reference_id=f"E{index}",
                source=result.source.value,
                summary=result.summary,
                relevance=_normalize_relevance(result.relevance),
                match_type=_normalize_match_type(result),
                key_facts=_extract_key_facts(result, max_facts=max_facts),
            )
        )

    return evidence


def _evidence_sort_key(result: RetrievalResult) -> Tuple[float, str, str]:
    return (
        -_normalize_relevance(result.relevance),
        result.source.value,
        result.summary,
    )


def _normalize_relevance(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    numeric = max(0.0, min(1.0, numeric))
    return round(numeric, 4)


def _normalize_match_type(result: RetrievalResult) -> str:
    match_type = result.metadata.get("match_type")
    if isinstance(match_type, str) and match_type.strip():
        return match_type.strip()
    return "keyword"


def _extract_key_facts(result: RetrievalResult, max_facts: int) -> List[str]:
    source_handlers = {
        RetrievalSource.MATERIALS: _facts_for_materials,
        RetrievalSource.TOLERANCE: _facts_for_tolerance,
        RetrievalSource.THREADS: _facts_for_threads,
        RetrievalSource.BEARINGS: _facts_for_bearings,
        RetrievalSource.SEALS: _facts_for_seals,
        RetrievalSource.MACHINING: _facts_for_machining,
    }
    handler = source_handlers.get(result.source, _facts_generic)
    return handler(result.data, max_facts=max_facts)


def _facts_for_materials(data: Dict[str, Any], max_facts: int) -> List[str]:
    grade = data.get("grade")
    props = data.get("properties", {})
    cost = data.get("cost", {})
    candidates = [
        _format_fact("牌号", grade),
        _format_fact("名称", props.get("name")),
        _format_fact("抗拉强度", props.get("tensile_strength"), "MPa"),
        _format_fact("屈服强度", props.get("yield_strength"), "MPa"),
        _format_fact("硬度", props.get("hardness")),
        _format_fact("成本等级", cost.get("cost_tier")),
    ]
    return _compact_facts(candidates, max_facts)


def _facts_for_tolerance(data: Dict[str, Any], max_facts: int) -> List[str]:
    deviations = data.get("deviations", {})

    if "fit_code" in data and deviations:
        candidates = [
            _format_fact("配合", data.get("fit_code")),
            _format_fact("尺寸", data.get("diameter"), "mm"),
            _format_fact(
                "间隙范围",
                f"{_format_scalar(deviations.get('min_clearance'))}"
                f" ~ {_format_scalar(deviations.get('max_clearance'))} μm",
            ),
            _format_fact(
                "孔偏差",
                f"{_format_scalar(deviations.get('hole_lower'))}"
                f" ~ {_format_scalar(deviations.get('hole_upper'))} μm",
            ),
            _format_fact(
                "轴偏差",
                f"{_format_scalar(deviations.get('shaft_lower'))}"
                f" ~ {_format_scalar(deviations.get('shaft_upper'))} μm",
            ),
        ]
        return _compact_facts(candidates, max_facts)

    symbol = data.get("symbol")
    grade = data.get("grade")
    if symbol and grade and "lower_deviation_um" in data and "upper_deviation_um" in data:
        is_hole = data.get("type") == "hole"
        low_label = "EI" if is_hole else "ei"
        high_label = "ES" if is_hole else "es"
        candidates = [
            _format_fact("公差带", f"{symbol}{grade}"),
            _format_fact("尺寸", data.get("diameter"), "mm"),
            _format_fact(low_label, data.get("lower_deviation_um"), "μm"),
            _format_fact(high_label, data.get("upper_deviation_um"), "μm"),
            _format_fact("公差值", data.get("tolerance_um"), "μm"),
            _format_fact("数据来源", data.get("source")),
        ]
        return _compact_facts(candidates, max_facts)

    candidates = [
        _format_fact("公差等级", data.get("grade")),
        _format_fact("尺寸", data.get("diameter"), "mm"),
        _format_fact("公差值", data.get("tolerance_um"), "μm"),
        _format_fact("类型", data.get("fit_code")),
    ]
    return _compact_facts(candidates, max_facts)


def _facts_for_threads(data: Dict[str, Any], max_facts: int) -> List[str]:
    candidates = [
        _format_fact("螺纹规格", data.get("designation")),
        _format_fact("公称直径", data.get("nominal_diameter"), "mm"),
        _format_fact("螺距", data.get("pitch"), "mm"),
        _format_fact("攻丝底孔", data.get("tap_drill"), "mm"),
    ]
    return _compact_facts(candidates, max_facts)


def _facts_for_bearings(data: Dict[str, Any], max_facts: int) -> List[str]:
    candidates = [
        _format_fact("轴承型号", data.get("designation")),
        _format_fact("内径", data.get("bore"), "mm"),
        _format_fact("外径", data.get("outer_d"), "mm"),
        _format_fact("宽度", data.get("width"), "mm"),
        _format_fact("动载荷", data.get("dynamic_load"), "kN"),
    ]
    return _compact_facts(candidates, max_facts)


def _facts_for_seals(data: Dict[str, Any], max_facts: int) -> List[str]:
    candidates = [
        _format_fact("密封规格", data.get("designation")),
        _format_fact("内径", data.get("inner_diameter"), "mm"),
        _format_fact("截面", data.get("cross_section"), "mm"),
        _format_fact("沟槽宽度", data.get("groove_width"), "mm"),
        _format_fact("沟槽深度", data.get("groove_depth"), "mm"),
    ]
    return _compact_facts(candidates, max_facts)


def _facts_for_machining(data: Dict[str, Any], max_facts: int) -> List[str]:
    candidates = [
        _format_fact("材料", data.get("material")),
        _format_fact("ISO材料组", data.get("iso_group")),
        _format_fact("推荐切削速度", data.get("cutting_speed"), "m/min"),
        _format_fact("推荐进给量", data.get("feed"), "mm/rev"),
        _format_fact("推荐刀具材料", data.get("tool_material")),
    ]
    return _compact_facts(candidates, max_facts)


def _facts_generic(data: Dict[str, Any], max_facts: int) -> List[str]:
    candidates: List[str] = []

    for key, value in data.items():
        if len(candidates) >= max_facts:
            break
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            candidates.append(_format_fact(_humanize_key(key), value))
            continue
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if len(candidates) >= max_facts:
                    break
                if isinstance(nested_value, (str, int, float, bool)):
                    candidates.append(
                        _format_fact(
                            f"{_humanize_key(key)}.{_humanize_key(nested_key)}",
                            nested_value,
                        )
                    )

    return _compact_facts(candidates, max_facts)


def _compact_facts(facts: Iterable[str], max_facts: int) -> List[str]:
    compacted: List[str] = []
    seen = set()
    for fact in facts:
        if not fact or fact in seen:
            continue
        compacted.append(fact)
        seen.add(fact)
        if len(compacted) >= max_facts:
            break
    return compacted


def _format_fact(label: str, value: Any, unit: str | None = None) -> str:
    if value is None or value == "":
        return ""
    formatted_value = _format_scalar(value)
    if unit and formatted_value:
        return f"{label}: {formatted_value} {unit}"
    return f"{label}: {formatted_value}"


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _humanize_key(key: str) -> str:
    return key.replace("_", " ").strip()
