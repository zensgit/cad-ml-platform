"""
Response Quality Evaluation Module.

Provides multi-dimensional evaluation of assistant responses,
including relevance, completeness, clarity, technical depth, and actionability.
"""

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class QualityDimension(Enum):
    """Quality evaluation dimensions."""

    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    TECHNICAL_DEPTH = "technical_depth"
    ACTIONABILITY = "actionability"


@dataclass
class DimensionScore:
    """Score for a single quality dimension."""

    dimension: QualityDimension
    score: float  # 0.0 to 1.0
    weight: float  # Contribution to overall score
    details: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dimension": self.dimension.value,
            "score": self.score,
            "weight": self.weight,
            "weighted_score": self.score * self.weight,
            "details": self.details,
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result."""

    overall_score: float
    grade: str  # A, B, C, D, F
    dimension_scores: Dict[QualityDimension, DimensionScore]
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": self.overall_score,
            "grade": self.grade,
            "dimension_scores": {
                k.value: v.to_dict() for k, v in self.dimension_scores.items()
            },
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "suggestions": self.suggestions,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """Create from dictionary."""
        dimension_scores = {}
        for k, v in data.get("dimension_scores", {}).items():
            dim = QualityDimension(k)
            dimension_scores[dim] = DimensionScore(
                dimension=dim,
                score=v["score"],
                weight=v["weight"],
                details=v.get("details", []),
            )

        return cls(
            overall_score=data["overall_score"],
            grade=data["grade"],
            dimension_scores=dimension_scores,
            strengths=data.get("strengths", []),
            weaknesses=data.get("weaknesses", []),
            suggestions=data.get("suggestions", []),
            timestamp=data.get("timestamp", time.time()),
        )

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"评估结果: {self.grade} ({self.overall_score:.2f})",
            "",
            "维度分数:",
        ]

        for dim, score in self.dimension_scores.items():
            lines.append(f"  - {dim.value}: {score.score:.2f}")

        if self.strengths:
            lines.append("")
            lines.append("优点:")
            for s in self.strengths:
                lines.append(f"  + {s}")

        if self.weaknesses:
            lines.append("")
            lines.append("不足:")
            for w in self.weaknesses:
                lines.append(f"  - {w}")

        if self.suggestions:
            lines.append("")
            lines.append("建议:")
            for s in self.suggestions:
                lines.append(f"  * {s}")

        return "\n".join(lines)


class RelevanceEvaluator:
    """Evaluates response relevance to the query."""

    def __init__(self):
        self.domain_keywords = {
            "materials": ["钢", "铝", "铜", "合金", "材料", "强度", "硬度", "密度"],
            "welding": ["焊", "焊接", "TIG", "MIG", "电流", "电压", "气体"],
            "gdt": ["公差", "尺寸", "平面度", "位置度", "基准", "GD&T"],
            "machining": ["加工", "切削", "车削", "铣削", "钻孔", "数控"],
        }

    def evaluate(self, query: str, response: str) -> DimensionScore:
        """Evaluate relevance."""
        score = 0.0
        details = []

        # Keyword overlap
        query_words = set(query)
        response_words = set(response)
        overlap = len(query_words & response_words) / max(len(query_words), 1)
        score += overlap * 0.4
        details.append(f"关键词覆盖率: {overlap:.0%}")

        # Domain matching
        query_lower = query.lower()
        response_lower = response.lower()

        matched_domains = []
        for domain, keywords in self.domain_keywords.items():
            if any(kw in query_lower for kw in keywords):
                if any(kw in response_lower for kw in keywords):
                    matched_domains.append(domain)

        if matched_domains:
            score += 0.4
            details.append(f"领域匹配: {', '.join(matched_domains)}")
        else:
            details.append("未检测到特定领域匹配")

        # Response length relative to query
        if len(response) >= len(query) * 2:
            score += 0.2
            details.append("响应长度充分")
        else:
            details.append("响应可能过短")

        return DimensionScore(
            dimension=QualityDimension.RELEVANCE,
            score=min(score, 1.0),
            weight=0.25,
            details=details,
        )


class CompletenessEvaluator:
    """Evaluates response completeness."""

    def __init__(self):
        self.data_patterns = [
            r"\d+\.?\d*\s*(MPa|GPa|mm|cm|m|kg|g|°C|℃|A|V|L/min)",  # Units
            r"\d+\.?\d*\s*[-~～]\s*\d+\.?\d*",  # Ranges
            r"约\d+|大约\d+|approximately",  # Approximations
        ]

    def evaluate(self, query: str, response: str) -> DimensionScore:
        """Evaluate completeness."""
        score = 0.0
        details = []

        # Check for numerical data
        has_numbers = bool(re.search(r"\d+", response))
        if has_numbers:
            score += 0.2
            details.append("包含数值数据")

        # Check for units
        has_units = any(re.search(p, response) for p in self.data_patterns)
        if has_units:
            score += 0.3
            details.append("包含单位或范围")

        # Check for structured content
        has_structure = any([
            "：" in response or ":" in response,
            "。" in response,
            "\n" in response,
            "1." in response or "一、" in response,
        ])
        if has_structure:
            score += 0.2
            details.append("内容结构化")

        # Response length
        if len(response) >= 50:
            score += 0.15
            details.append("响应长度足够")
        if len(response) >= 100:
            score += 0.15
            details.append("响应详细")

        return DimensionScore(
            dimension=QualityDimension.COMPLETENESS,
            score=min(score, 1.0),
            weight=0.25,
            details=details,
        )


class ClarityEvaluator:
    """Evaluates response clarity."""

    def __init__(self):
        self.vague_patterns = [
            r"可能|也许|大概|或许|maybe|perhaps|probably",
            r"有时|sometimes|occasionally",
            r"某些|某种|一些|some",
            r"等等|之类|etc",
        ]

    def evaluate(self, query: str, response: str) -> DimensionScore:
        """Evaluate clarity."""
        score = 1.0  # Start high, deduct for issues
        details = []

        # Check for vague language
        vague_count = sum(
            len(re.findall(p, response, re.IGNORECASE))
            for p in self.vague_patterns
        )

        if vague_count > 3:
            score -= 0.3
            details.append(f"模糊语言较多 ({vague_count}处)")
        elif vague_count > 0:
            score -= 0.1
            details.append(f"存在少量模糊表达 ({vague_count}处)")
        else:
            details.append("表达清晰明确")

        # Check sentence structure
        sentences = re.split(r"[。！？.!?]", response)
        avg_length = sum(len(s) for s in sentences) / max(len(sentences), 1)

        if avg_length > 100:
            score -= 0.2
            details.append("句子过长，可能影响理解")
        elif avg_length < 10 and len(sentences) > 1:
            score -= 0.1
            details.append("句子过短，可能不够详细")
        else:
            details.append("句子长度适中")

        # Check for structure markers
        has_markers = any([
            re.search(r"首先|其次|最后|第[一二三四五]", response),
            re.search(r"1\.|2\.|3\.|①|②|③", response),
        ])
        if has_markers:
            score += 0.1
            details.append("使用了结构化标记")

        return DimensionScore(
            dimension=QualityDimension.CLARITY,
            score=max(0, min(score, 1.0)),
            weight=0.20,
            details=details,
        )


class TechnicalDepthEvaluator:
    """Evaluates technical depth of response."""

    def __init__(self):
        self.technical_patterns = [
            r"MPa|GPa|Pa",  # Pressure/stress units
            r"mm|cm|μm|nm",  # Length units
            r"kg/m³|g/cm³",  # Density
            r"°C|℃|K",  # Temperature
            r"HRC|HB|HV",  # Hardness
            r"ASTM|ISO|GB|JIS",  # Standards
        ]

    def evaluate(self, query: str, response: str) -> DimensionScore:
        """Evaluate technical depth."""
        score = 0.0
        details = []

        # Check for technical terms
        tech_matches = sum(
            len(re.findall(p, response))
            for p in self.technical_patterns
        )

        if tech_matches >= 3:
            score += 0.4
            details.append(f"技术术语丰富 ({tech_matches}个)")
        elif tech_matches >= 1:
            score += 0.2
            details.append(f"包含技术术语 ({tech_matches}个)")
        else:
            details.append("技术术语较少")

        # Check for numerical precision
        precise_numbers = re.findall(r"\d+\.\d+", response)
        if len(precise_numbers) >= 2:
            score += 0.3
            details.append("数据精确度高")
        elif len(precise_numbers) >= 1:
            score += 0.15
            details.append("包含精确数值")

        # Check for formulas or equations
        has_formula = bool(re.search(r"[=×÷+-].*\d|σ|ε|τ", response))
        if has_formula:
            score += 0.2
            details.append("包含公式或计算")

        # Check for references to standards
        has_standards = bool(re.search(r"GB|ISO|ASTM|JIS|DIN", response))
        if has_standards:
            score += 0.1
            details.append("引用了技术标准")

        return DimensionScore(
            dimension=QualityDimension.TECHNICAL_DEPTH,
            score=min(score, 1.0),
            weight=0.15,
            details=details,
        )


class ActionabilityEvaluator:
    """Evaluates actionability of response."""

    def __init__(self):
        self.action_patterns = [
            r"建议|推荐|应该|需要|可以|suggest|recommend|should",
            r"步骤|方法|流程|process|step|method",
            r"注意|警告|小心|avoid|warning|caution",
        ]

    def evaluate(self, query: str, response: str) -> DimensionScore:
        """Evaluate actionability."""
        score = 0.0
        details = []

        # Check for action-oriented language
        action_count = sum(
            len(re.findall(p, response, re.IGNORECASE))
            for p in self.action_patterns
        )

        if action_count >= 3:
            score += 0.4
            details.append(f"行动导向语言丰富 ({action_count}处)")
        elif action_count >= 1:
            score += 0.2
            details.append(f"包含行动建议 ({action_count}处)")
        else:
            details.append("缺少具体行动建议")

        # Check for step-by-step guidance
        has_steps = bool(re.search(
            r"第[一二三四五]步|步骤\s*\d|1\.\s*\S|首先.*然后.*最后",
            response
        ))
        if has_steps:
            score += 0.3
            details.append("包含分步骤指导")

        # Check for practical examples
        has_examples = bool(re.search(r"例如|比如|如：|例：|for example", response))
        if has_examples:
            score += 0.2
            details.append("包含实际示例")

        # Check for warnings/cautions
        has_warnings = bool(re.search(r"注意|警告|避免|小心|不要", response))
        if has_warnings:
            score += 0.1
            details.append("包含注意事项")

        return DimensionScore(
            dimension=QualityDimension.ACTIONABILITY,
            score=min(score, 1.0),
            weight=0.15,
            details=details,
        )


class ResponseQualityEvaluator:
    """
    Main evaluator combining all dimensions.

    Example:
        >>> evaluator = ResponseQualityEvaluator()
        >>> result = evaluator.evaluate(
        ...     "304不锈钢的强度是多少？",
        ...     "304不锈钢的抗拉强度约为520MPa，屈服强度约为205MPa。"
        ... )
        >>> print(result.grade)
        'B'
    """

    def __init__(self):
        self.evaluators = {
            QualityDimension.RELEVANCE: RelevanceEvaluator(),
            QualityDimension.COMPLETENESS: CompletenessEvaluator(),
            QualityDimension.CLARITY: ClarityEvaluator(),
            QualityDimension.TECHNICAL_DEPTH: TechnicalDepthEvaluator(),
            QualityDimension.ACTIONABILITY: ActionabilityEvaluator(),
        }

    def evaluate(self, query: str, response: str) -> EvaluationResult:
        """
        Evaluate response quality across all dimensions.

        Args:
            query: User's original query
            response: Assistant's response

        Returns:
            Complete evaluation result
        """
        # Evaluate each dimension
        dimension_scores = {}
        for dim, evaluator in self.evaluators.items():
            dimension_scores[dim] = evaluator.evaluate(query, response)

        # Calculate overall score
        overall_score = sum(
            score.score * score.weight
            for score in dimension_scores.values()
        )

        # Determine grade
        grade = self._score_to_grade(overall_score)

        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []

        for dim, score in dimension_scores.items():
            if score.score >= 0.7:
                strengths.append(f"{dim.value}: {score.details[0] if score.details else '良好'}")
            elif score.score <= 0.4:
                weaknesses.append(f"{dim.value}: {score.details[-1] if score.details else '需改进'}")

        # Generate suggestions
        suggestions = self._generate_suggestions(dimension_scores)

        return EvaluationResult(
            overall_score=overall_score,
            grade=grade,
            dimension_scores=dimension_scores,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
        )

    def _score_to_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"

    def _generate_suggestions(
        self,
        scores: Dict[QualityDimension, DimensionScore]
    ) -> List[str]:
        """Generate improvement suggestions based on scores."""
        suggestions = []

        if scores[QualityDimension.RELEVANCE].score < 0.6:
            suggestions.append("提高响应与问题的相关性，确保回答直接针对用户问题")

        if scores[QualityDimension.COMPLETENESS].score < 0.6:
            suggestions.append("补充更多具体数据和详细信息")

        if scores[QualityDimension.CLARITY].score < 0.6:
            suggestions.append("简化表达，减少模糊用语，使用更清晰的结构")

        if scores[QualityDimension.TECHNICAL_DEPTH].score < 0.6:
            suggestions.append("增加技术细节和专业术语，引用相关标准")

        if scores[QualityDimension.ACTIONABILITY].score < 0.6:
            suggestions.append("添加具体操作步骤和实际应用建议")

        return suggestions


class EvaluationHistory:
    """
    Track evaluation history over time.

    Supports persistence and trend analysis.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize evaluation history.

        Args:
            storage_path: Path for persistence (optional)
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.results: List[EvaluationResult] = []

        if self.storage_path and self.storage_path.exists():
            self._load()

    def add_result(self, result: EvaluationResult) -> None:
        """Add evaluation result to history."""
        self.results.append(result)

    def get_average_score(self) -> float:
        """Get average overall score."""
        if not self.results:
            return 0.0
        return sum(r.overall_score for r in self.results) / len(self.results)

    def get_trend(self, window: int = 10) -> List[float]:
        """Get score trend over recent evaluations."""
        recent = self.results[-window:]
        return [r.overall_score for r in recent]

    def get_dimension_averages(self) -> Dict[str, float]:
        """Get average scores by dimension."""
        if not self.results:
            return {}

        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}

        for result in self.results:
            for dim, score in result.dimension_scores.items():
                key = dim.value
                totals[key] = totals.get(key, 0) + score.score
                counts[key] = counts.get(key, 0) + 1

        return {k: totals[k] / counts[k] for k in totals}

    def save(self) -> bool:
        """Save history to disk."""
        if not self.storage_path:
            return False

        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = [r.to_dict() for r in self.results]
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except IOError:
            return False

    def _load(self) -> bool:
        """Load history from disk."""
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.results = [EvaluationResult.from_dict(d) for d in data]
            return True
        except (IOError, json.JSONDecodeError, KeyError):
            return False

    def clear(self) -> None:
        """Clear all history."""
        self.results = []
