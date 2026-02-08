"""
Context Assembler for CAD-ML Assistant.

Assembles retrieved knowledge into a structured context for LLM prompting.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .knowledge_retriever import RetrievalResult, RetrievalSource
from .query_analyzer import AnalyzedQuery, QueryIntent


@dataclass
class AssembledContext:
    """Assembled context for LLM prompting."""

    query: AnalyzedQuery
    knowledge_context: str  # Formatted knowledge for prompt
    system_prompt: str  # System prompt with domain expertise
    user_prompt: str  # User prompt with query and context

    # Metadata
    sources_used: List[RetrievalSource] = field(default_factory=list)
    token_estimate: int = 0


class ContextAssembler:
    """
    Assembles retrieved knowledge into prompts for LLM.

    Responsibilities:
    - Format retrieved data into readable context
    - Build system prompts with domain expertise
    - Construct user prompts with query and context
    - Manage context window limits

    Example:
        >>> assembler = ContextAssembler()
        >>> context = assembler.assemble(query, retrieval_results)
        >>> print(context.system_prompt)
        >>> print(context.user_prompt)
    """

    # Domain expertise descriptions
    DOMAIN_EXPERTISE = {
        "materials": """你是材料工程专家，熟悉:
- 金属材料(钢、铝、铜、钛等)的物理和机械性能
- 材料选型原则和应用场景
- 热处理对材料性能的影响
- 材料成本和可加工性""",
        "tolerance": """你是公差配合专家，熟悉:
- ISO 286标准公差等级(IT01-IT18)
- 基孔制/基轴制配合系统
- 配合选择原则和应用场景
- 公差计算和偏差分析""",
        "standards": """你是标准件专家，熟悉:
- ISO公制螺纹规格和选型
- 滚动轴承类型、尺寸和选型计算
- 密封件(O形圈等)规格和材料选择
- 标准件的设计应用""",
        "machining": """你是机械加工专家，熟悉:
- 切削参数计算(切削速度、进给、切深)
- 刀具选择和几何参数
- 材料可加工性评估
- 工艺路线设计""",
        "gdt": """你是几何尺寸和公差(GD&T)专家，熟悉:
- ASME Y14.5 / ISO 1101标准
- 形位公差符号和解读
- 基准系统和公差框架
- GD&T在设计中的应用""",
    }

    def __init__(self, max_context_tokens: int = 4000):
        self.max_context_tokens = max_context_tokens

    def assemble(
        self,
        query: AnalyzedQuery,
        results: List[RetrievalResult],
    ) -> AssembledContext:
        """
        Assemble context from query and retrieval results.

        Args:
            query: Analyzed user query
            results: Retrieved knowledge results

        Returns:
            AssembledContext ready for LLM
        """
        # Build knowledge context
        knowledge_context = self._format_knowledge(results)

        # Build system prompt
        system_prompt = self._build_system_prompt(query, results)

        # Build user prompt
        user_prompt = self._build_user_prompt(query, knowledge_context)

        # Extract sources used
        sources_used = list(set(r.source for r in results))

        # Estimate tokens (rough: 1 token ≈ 2 Chinese chars or 4 English chars)
        token_estimate = self._estimate_tokens(system_prompt + user_prompt)

        return AssembledContext(
            query=query,
            knowledge_context=knowledge_context,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            sources_used=sources_used,
            token_estimate=token_estimate,
        )

    def _format_knowledge(self, results: List[RetrievalResult]) -> str:
        """Format retrieval results into readable context."""
        if not results:
            return "未找到相关知识。"

        sections = []

        for result in results:
            section = self._format_result(result)
            if section:
                sections.append(section)

        return "\n\n".join(sections)

    def _format_result(self, result: RetrievalResult) -> str:
        """Format a single retrieval result."""
        formatters = {
            RetrievalSource.MATERIALS: self._format_material,
            RetrievalSource.TOLERANCE: self._format_tolerance,
            RetrievalSource.THREADS: self._format_thread,
            RetrievalSource.BEARINGS: self._format_bearing,
            RetrievalSource.SEALS: self._format_seal,
            RetrievalSource.MACHINING: self._format_machining,
        }

        formatter = formatters.get(result.source)
        if formatter:
            return formatter(result.data)

        return f"[{result.source.value}] {result.summary}"

    def _format_material(self, data: Dict) -> str:
        """Format material data."""
        grade = data.get("grade", "Unknown")
        props = data.get("properties", {})

        lines = [f"【材料: {grade}】"]

        if "name" in props:
            lines.append(f"名称: {props['name']}")

        # Physical properties
        if "density" in props:
            lines.append(f"密度: {props['density']} g/cm³")
        if "melting_point" in props:
            lines.append(f"熔点: {props['melting_point']} °C")

        # Mechanical properties
        if "tensile_strength" in props:
            lines.append(f"抗拉强度: {props['tensile_strength']} MPa")
        if "yield_strength" in props:
            lines.append(f"屈服强度: {props['yield_strength']} MPa")
        if "hardness" in props:
            lines.append(f"硬度: {props['hardness']}")

        # Cost data
        cost = data.get("cost", {})
        if "cost_tier" in cost:
            lines.append(f"成本等级: {cost['cost_tier']}")

        return "\n".join(lines)

    def _format_tolerance(self, data: Dict) -> str:
        """Format tolerance data."""
        lines = []

        if "fit_code" in data and "deviations" in data:
            lines.append(f"【配合: {data['fit_code']}】")
            lines.append(f"尺寸: {data.get('diameter', 'N/A')} mm")
            dev = data["deviations"]
            lines.append(f"孔偏差: +{dev['hole_upper']}/+{dev['hole_lower']} μm")
            lines.append(f"轴偏差: {dev['shaft_upper']}/{dev['shaft_lower']} μm")
            lines.append(
                f"间隙范围: {dev['min_clearance']} ~ {dev['max_clearance']} μm"
            )

        elif (
            "symbol" in data
            and "grade" in data
            and "lower_deviation_um" in data
            and "upper_deviation_um" in data
        ):
            symbol = str(data.get("symbol") or "").strip()
            grade = str(data.get("grade") or "").strip()
            label = f"{symbol}{grade}" if symbol and grade else symbol or grade
            lines.append(f"【极限偏差: {label}】")
            lines.append(f"尺寸: {data.get('diameter', 'N/A')} mm")
            lower = data.get("lower_deviation_um")
            upper = data.get("upper_deviation_um")
            kind = str(data.get("type") or "")
            if kind == "hole":
                lines.append(f"EI={lower} μm, ES={upper} μm")
            elif kind == "shaft":
                lines.append(f"ei={lower} μm, es={upper} μm")
            else:
                lines.append(f"下偏差={lower} μm, 上偏差={upper} μm")
            try:
                lines.append(f"公差带: {float(upper) - float(lower)} μm")
            except Exception:
                pass

        elif "grade" in data and "tolerance_um" in data:
            lines.append(f"【公差: {data['grade']}】")
            lines.append(f"尺寸: {data.get('diameter', 'N/A')} mm")
            lines.append(f"公差值: {data['tolerance_um']} μm")

        elif "fit_code" in data:
            lines.append(f"【配合: {data['fit_code']}】")
            info = data.get("info", {})
            if "name_zh" in info:
                lines.append(f"类型: {info['name_zh']}")
            if "application_zh" in info:
                lines.append(f"应用: {info['application_zh']}")

        return "\n".join(lines)

    def _format_thread(self, data: Dict) -> str:
        """Format thread data."""
        lines = [f"【螺纹: {data.get('designation', 'Unknown')}】"]
        lines.append(f"公称直径: {data.get('nominal_diameter', 'N/A')} mm")
        lines.append(f"螺距: {data.get('pitch', 'N/A')} mm")
        lines.append(f"中径: {data.get('pitch_diameter', 'N/A')} mm")
        lines.append(f"小径: {data.get('minor_diameter', 'N/A')} mm")
        lines.append(f"攻丝底孔: {data.get('tap_drill', 'N/A')} mm")
        return "\n".join(lines)

    def _format_bearing(self, data: Dict) -> str:
        """Format bearing data."""
        lines = [f"【轴承: {data.get('designation', 'Unknown')}】"]
        lines.append(f"内径 d: {data.get('bore', 'N/A')} mm")
        lines.append(f"外径 D: {data.get('outer_d', 'N/A')} mm")
        lines.append(f"宽度 B: {data.get('width', 'N/A')} mm")
        if "dynamic_load" in data:
            lines.append(f"动载荷 C: {data['dynamic_load']} kN")
        if "static_load" in data:
            lines.append(f"静载荷 C0: {data['static_load']} kN")
        return "\n".join(lines)

    def _format_seal(self, data: Dict) -> str:
        """Format seal data."""
        lines = [f"【O形圈: {data.get('designation', 'Unknown')}】"]
        lines.append(f"内径 ID: {data.get('inner_diameter', 'N/A')} mm")
        lines.append(f"截面 CS: {data.get('cross_section', 'N/A')} mm")
        if "groove_width" in data:
            lines.append(f"沟槽宽度: {data['groove_width']} mm")
        if "groove_depth" in data:
            lines.append(f"沟槽深度: {data['groove_depth']} mm")
        return "\n".join(lines)

    def _format_machining(self, data: Dict) -> str:
        """Format machining data."""
        lines = [f"【加工参数: {data.get('material', 'Unknown')}】"]
        lines.append(f"ISO材料组: {data.get('iso_group', 'N/A')}")
        lines.append(f"可加工性: {data.get('machinability_rating', 'N/A')}%")
        if data.get("cutting_speed"):
            lines.append(f"推荐切削速度: {data['cutting_speed']} m/min")
        if data.get("feed"):
            lines.append(f"推荐进给量: {data['feed']} mm/rev")
        if data.get("tool_material"):
            lines.append(f"推荐刀具材料: {data['tool_material']}")
        return "\n".join(lines)

    def _build_system_prompt(
        self,
        query: AnalyzedQuery,
        results: List[RetrievalResult],
    ) -> str:
        """Build system prompt with domain expertise."""
        # Base prompt
        prompt = """你是CAD-ML平台的智能助手，专注于机械设计和制造工程领域。

你的职责:
1. 基于提供的知识库数据，准确回答用户问题
2. 提供专业、实用的工程建议
3. 如果知识库中没有相关数据，诚实说明并给出一般性指导
4. 回答要简洁明了，重点突出

"""

        # Add domain expertise based on sources
        domains_used = set()
        for result in results:
            if result.source == RetrievalSource.MATERIALS:
                domains_used.add("materials")
            elif result.source == RetrievalSource.TOLERANCE:
                domains_used.add("tolerance")
            elif result.source in [
                RetrievalSource.THREADS,
                RetrievalSource.BEARINGS,
                RetrievalSource.SEALS,
            ]:
                domains_used.add("standards")
            elif result.source == RetrievalSource.MACHINING:
                domains_used.add("machining")

        for domain in domains_used:
            expertise = self.DOMAIN_EXPERTISE.get(domain)
            if expertise:
                prompt += f"\n{expertise}\n"

        return prompt

    def _build_user_prompt(
        self,
        query: AnalyzedQuery,
        knowledge_context: str,
    ) -> str:
        """Build user prompt with query and context."""
        prompt = f"""用户问题: {query.original_query}

参考知识:
{knowledge_context}

请基于以上知识回答用户问题。如果知识库中没有完全匹配的信息，请给出合理的推断或建议。"""

        return prompt

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough estimation: Chinese ~2 chars/token, English ~4 chars/token
        chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        other_chars = len(text) - chinese_chars

        return int(chinese_chars / 1.5 + other_chars / 4)
