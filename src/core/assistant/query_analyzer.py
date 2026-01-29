"""
Query Analyzer for CAD-ML Assistant.

Analyzes user queries to determine intent, extract keywords, and identify
which knowledge domains are relevant for retrieval.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set
import re


class QueryIntent(str, Enum):
    """User query intent classification."""

    # Material related
    MATERIAL_PROPERTY = "material_property"  # 查询材料属性
    MATERIAL_SELECTION = "material_selection"  # 材料选择建议
    MATERIAL_COMPARISON = "material_comparison"  # 材料对比

    # Tolerance related
    TOLERANCE_LOOKUP = "tolerance_lookup"  # 查询公差值
    FIT_SELECTION = "fit_selection"  # 配合选择
    FIT_CALCULATION = "fit_calculation"  # 配合计算

    # Standard parts
    THREAD_SPEC = "thread_spec"  # 螺纹规格
    BEARING_SPEC = "bearing_spec"  # 轴承规格
    SEAL_SPEC = "seal_spec"  # 密封件规格

    # Machining
    CUTTING_PARAMETERS = "cutting_parameters"  # 切削参数
    TOOL_SELECTION = "tool_selection"  # 刀具选择
    PROCESS_ROUTE = "process_route"  # 工艺路线

    # Welding
    WELDING_PARAMETERS = "welding_parameters"  # 焊接参数
    WELDING_JOINT = "welding_joint"  # 焊接接头设计
    WELDABILITY = "weldability"  # 焊接性评估

    # Heat Treatment
    HEAT_TREATMENT = "heat_treatment"  # 热处理参数
    HARDENING = "hardening"  # 淬火/硬化
    ANNEALING = "annealing"  # 退火

    # Surface Treatment
    ELECTROPLATING = "electroplating"  # 电镀
    ANODIZING = "anodizing"  # 阳极氧化
    COATING = "coating"  # 涂装/涂层

    # GD&T
    GDT_INTERPRETATION = "gdt_interpretation"  # GD&T解读
    GDT_APPLICATION = "gdt_application"  # GD&T应用

    # General
    GENERAL_QUESTION = "general_question"  # 一般问题
    UNKNOWN = "unknown"  # 无法识别


@dataclass
class AnalyzedQuery:
    """Result of query analysis."""

    original_query: str
    intent: QueryIntent
    confidence: float  # 0.0 to 1.0

    # Extracted information
    keywords: List[str] = field(default_factory=list)
    entities: Dict[str, str] = field(default_factory=dict)  # e.g., {"material": "304", "diameter": "25"}
    domains: List[str] = field(default_factory=list)  # Knowledge domains to search

    # Query refinement
    normalized_query: str = ""
    sub_queries: List[str] = field(default_factory=list)


class QueryAnalyzer:
    """
    Analyzes user queries to extract intent, keywords, and relevant domains.

    Example:
        >>> analyzer = QueryAnalyzer()
        >>> result = analyzer.analyze("304不锈钢的抗拉强度是多少?")
        >>> print(result.intent)  # QueryIntent.MATERIAL_PROPERTY
        >>> print(result.entities)  # {"material": "304"}
    """

    def __init__(self):
        # Intent patterns (regex patterns for each intent)
        self._intent_patterns = self._build_intent_patterns()

        # Domain mapping
        self._intent_to_domains = {
            QueryIntent.MATERIAL_PROPERTY: ["materials"],
            QueryIntent.MATERIAL_SELECTION: ["materials", "machining"],
            QueryIntent.MATERIAL_COMPARISON: ["materials"],
            QueryIntent.TOLERANCE_LOOKUP: ["tolerance"],
            QueryIntent.FIT_SELECTION: ["tolerance"],
            QueryIntent.FIT_CALCULATION: ["tolerance"],
            QueryIntent.THREAD_SPEC: ["standards.threads"],
            QueryIntent.BEARING_SPEC: ["standards.bearings"],
            QueryIntent.SEAL_SPEC: ["standards.seals"],
            QueryIntent.CUTTING_PARAMETERS: ["machining.cutting", "machining.materials"],
            QueryIntent.TOOL_SELECTION: ["machining.tooling"],
            QueryIntent.PROCESS_ROUTE: ["machining", "materials"],
            QueryIntent.WELDING_PARAMETERS: ["welding.parameters"],
            QueryIntent.WELDING_JOINT: ["welding.joints"],
            QueryIntent.WELDABILITY: ["welding.materials"],
            QueryIntent.HEAT_TREATMENT: ["heat_treatment.processes"],
            QueryIntent.HARDENING: ["heat_treatment.hardening"],
            QueryIntent.ANNEALING: ["heat_treatment.annealing"],
            QueryIntent.ELECTROPLATING: ["surface_treatment.plating"],
            QueryIntent.ANODIZING: ["surface_treatment.anodizing"],
            QueryIntent.COATING: ["surface_treatment.coating"],
            QueryIntent.GDT_INTERPRETATION: ["gdt"],
            QueryIntent.GDT_APPLICATION: ["gdt", "tolerance"],
        }

        # Entity extraction patterns
        self._entity_patterns = self._build_entity_patterns()

    def _build_intent_patterns(self) -> Dict[QueryIntent, List[re.Pattern]]:
        """Build regex patterns for intent detection."""
        patterns = {
            QueryIntent.MATERIAL_PROPERTY: [
                re.compile(r"(材料|钢|铝|铜|钛).*(属性|强度|硬度|密度|熔点|导热)", re.IGNORECASE),
                re.compile(r"(抗拉|屈服|延伸率|热处理)", re.IGNORECASE),
                re.compile(r"(什么|多少).*(强度|硬度|密度)", re.IGNORECASE),
            ],
            QueryIntent.MATERIAL_SELECTION: [
                re.compile(r"(选择|推荐|用什么|哪种).*材料", re.IGNORECASE),
                re.compile(r"材料.*(选择|推荐|建议)", re.IGNORECASE),
                re.compile(r"(适合|应该用).*什么.*(材料|钢|合金)", re.IGNORECASE),
            ],
            QueryIntent.TOLERANCE_LOOKUP: [
                re.compile(r"(IT|公差).*(等级|值|多少)", re.IGNORECASE),
                re.compile(r"(IT\d+|it\d+)", re.IGNORECASE),
                re.compile(r"公差.*(\d+mm|\d+毫米)", re.IGNORECASE),
            ],
            QueryIntent.FIT_SELECTION: [
                re.compile(r"(配合|H\d+|h\d+|g\d+|k\d+)", re.IGNORECASE),
                re.compile(r"(间隙|过盈|过渡).*配合", re.IGNORECASE),
                re.compile(r"(轴|孔).*配合", re.IGNORECASE),
            ],
            QueryIntent.THREAD_SPEC: [
                re.compile(r"(螺纹|螺栓|螺钉|螺母|M\d+)", re.IGNORECASE),
                re.compile(r"(攻丝|底孔|螺距)", re.IGNORECASE),
            ],
            QueryIntent.BEARING_SPEC: [
                re.compile(r"(轴承|bearing|\d{4,5})", re.IGNORECASE),
                re.compile(r"(内径|外径|宽度).*轴承", re.IGNORECASE),
                re.compile(r"(6\d{3}|6\d{4})", re.IGNORECASE),  # 6205, 62010 etc.
            ],
            QueryIntent.SEAL_SPEC: [
                re.compile(r"(O.*圈|密封|o-?ring)", re.IGNORECASE),
                re.compile(r"(\d+x\d+|\d+×\d+).*密封?", re.IGNORECASE),
            ],
            QueryIntent.CUTTING_PARAMETERS: [
                re.compile(r"(切削|车削|铣削|钻孔).*(参数|速度|进给)", re.IGNORECASE),
                re.compile(r"(转速|主轴|切深|吃刀)", re.IGNORECASE),
                re.compile(r"(Vc|vc|rpm|进给)", re.IGNORECASE),
            ],
            QueryIntent.TOOL_SELECTION: [
                re.compile(r"(刀具|刀片|铣刀|钻头).*(选择|推荐|用什么)", re.IGNORECASE),
                re.compile(r"(加工|切削).*用.*刀", re.IGNORECASE),
            ],
            QueryIntent.WELDING_PARAMETERS: [
                re.compile(r"(焊接|焊|SMAW|GMAW|GTAW|TIG|MIG|MAG).*(参数|电流|电压|速度)", re.IGNORECASE),
                re.compile(r"(电弧焊|气保焊|氩弧焊|埋弧焊)", re.IGNORECASE),
                re.compile(r"焊.*(多少|怎么)", re.IGNORECASE),
            ],
            QueryIntent.WELDING_JOINT: [
                re.compile(r"(坡口|接头|焊缝).*(设计|尺寸|角度)", re.IGNORECASE),
                re.compile(r"(V型|X型|U型|对接|搭接|T型).*焊", re.IGNORECASE),
            ],
            QueryIntent.WELDABILITY: [
                re.compile(r"(焊接性|可焊性|预热|焊后)", re.IGNORECASE),
                re.compile(r"(碳当量|CE|热输入)", re.IGNORECASE),
                re.compile(r"能.*焊", re.IGNORECASE),
            ],
            QueryIntent.HEAT_TREATMENT: [
                re.compile(r"(热处理|淬火|回火|退火|正火).*(参数|温度|时间)", re.IGNORECASE),
                re.compile(r"(热处理|淬火|回火).*怎么", re.IGNORECASE),
            ],
            QueryIntent.HARDENING: [
                re.compile(r"(淬火|淬透性|硬化|HRC).*(温度|介质|硬度)", re.IGNORECASE),
                re.compile(r"(油淬|水淬|空冷|回火).*温度", re.IGNORECASE),
            ],
            QueryIntent.ANNEALING: [
                re.compile(r"(退火|球化|去应力|正火).*(温度|时间|工艺)", re.IGNORECASE),
                re.compile(r"消除应力", re.IGNORECASE),
            ],
            QueryIntent.ELECTROPLATING: [
                re.compile(r"(电镀|镀锌|镀镍|镀铬|镀层).*(厚度|参数|工艺|推荐)", re.IGNORECASE),
                re.compile(r"(锌|镍|铬|金|银|硬铬).*镀", re.IGNORECASE),
                re.compile(r"镀.*(锌|镍|铬|金|银|硬)", re.IGNORECASE),
                re.compile(r"(盐雾|防腐|耐蚀).*镀", re.IGNORECASE),
            ],
            QueryIntent.ANODIZING: [
                re.compile(r"(阳极氧化|阳极|硬质阳极|氧化).*(类型|厚度|颜色|参数)", re.IGNORECASE),
                re.compile(r"(Type\s*I{1,3}|铬酸|硫酸).*阳极", re.IGNORECASE),
                re.compile(r"铝.*氧化", re.IGNORECASE),
            ],
            QueryIntent.COATING: [
                re.compile(r"(涂装|涂层|涂料|喷涂|喷漆).*(推荐|选择|厚度|参数)", re.IGNORECASE),
                re.compile(r"(粉末|环氧|聚氨酯|锌铬).*(涂|喷)", re.IGNORECASE),
                re.compile(r"(C[1-5]|CX|ISO\s*12944).*涂", re.IGNORECASE),
            ],
            QueryIntent.GDT_INTERPRETATION: [
                re.compile(r"(GD&?T|形位公差|几何公差)", re.IGNORECASE),
                re.compile(r"(平面度|圆度|垂直度|平行度|位置度|同轴度)", re.IGNORECASE),
            ],
        }
        return patterns

    def _build_entity_patterns(self) -> Dict[str, re.Pattern]:
        """Build regex patterns for entity extraction."""
        return {
            "material_grade": re.compile(
                r"(304|316|316L|Q235|45钢|40Cr|GCr15|6061|7075|Ti-?6Al-?4V|Inconel\s?\d+)",
                re.IGNORECASE
            ),
            "diameter": re.compile(r"[Ddφ直径]?\s*(\d+(?:\.\d+)?)\s*(?:mm|毫米)?"),
            "thread": re.compile(r"M(\d+(?:\.\d+)?)(x(\d+(?:\.\d+)?))?", re.IGNORECASE),
            "bearing": re.compile(r"(6[0-3]\d{2,3})", re.IGNORECASE),
            "it_grade": re.compile(r"IT\s?(\d{1,2})", re.IGNORECASE),
            "fit": re.compile(r"([HhGgKkMmNnPpRrSs]\d+)/([a-z]\d+)", re.IGNORECASE),
            "oring": re.compile(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)", re.IGNORECASE),
        }

    def analyze(self, query: str) -> AnalyzedQuery:
        """
        Analyze a user query.

        Args:
            query: User's natural language query

        Returns:
            AnalyzedQuery with intent, keywords, entities, and domains
        """
        # Normalize query
        normalized = self._normalize_query(query)

        # Detect intent
        intent, confidence = self._detect_intent(normalized)

        # Extract keywords
        keywords = self._extract_keywords(normalized)

        # Extract entities
        entities = self._extract_entities(query)  # Use original for entity extraction

        # Determine relevant domains
        domains = self._intent_to_domains.get(intent, ["general"])

        return AnalyzedQuery(
            original_query=query,
            intent=intent,
            confidence=confidence,
            keywords=keywords,
            entities=entities,
            domains=domains,
            normalized_query=normalized,
        )

    def _normalize_query(self, query: str) -> str:
        """Normalize query for processing."""
        # Remove extra whitespace
        normalized = " ".join(query.split())
        # Convert to lowercase for matching (keep original for display)
        return normalized

    def _detect_intent(self, query: str) -> tuple:
        """Detect query intent using pattern matching."""
        best_intent = QueryIntent.UNKNOWN
        best_confidence = 0.0

        for intent, patterns in self._intent_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    # Simple scoring: more specific patterns = higher confidence
                    confidence = 0.7 + (0.1 * len(pattern.pattern) / 50)
                    confidence = min(confidence, 0.95)

                    if confidence > best_confidence:
                        best_intent = intent
                        best_confidence = confidence

        # Default confidence for unknown
        if best_intent == QueryIntent.UNKNOWN:
            best_confidence = 0.3

        return best_intent, best_confidence

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query."""
        # Simple keyword extraction - can be enhanced with NLP
        # Remove common words
        stopwords = {"的", "是", "什么", "多少", "怎么", "如何", "请问", "能", "可以", "吗", "呢"}

        words = query.split()
        keywords = []

        for word in words:
            # Skip stopwords and very short words
            if word not in stopwords and len(word) > 1:
                keywords.append(word)

        return keywords

    def _extract_entities(self, query: str) -> Dict[str, str]:
        """Extract named entities from query."""
        entities = {}

        for entity_type, pattern in self._entity_patterns.items():
            match = pattern.search(query)
            if match:
                if entity_type == "thread":
                    entities["thread_diameter"] = match.group(1)
                    if match.group(3):
                        entities["thread_pitch"] = match.group(3)
                elif entity_type == "fit":
                    entities["hole_tolerance"] = match.group(1)
                    entities["shaft_tolerance"] = match.group(2)
                elif entity_type == "oring":
                    entities["oring_id"] = match.group(1)
                    entities["oring_cs"] = match.group(2)
                else:
                    entities[entity_type] = match.group(1) if match.groups() else match.group(0)

        return entities

    def get_suggested_queries(self, partial_query: str) -> List[str]:
        """Get query suggestions for autocomplete."""
        suggestions = [
            "304不锈钢的机械性能是什么?",
            "M10螺纹的底孔尺寸是多少?",
            "H7/g6配合的间隙范围?",
            "加工45钢用什么刀具?",
            "6205轴承的尺寸规格?",
            "IT7公差等级在25mm时的值?",
        ]

        # Filter by partial match
        if partial_query:
            lower_query = partial_query.lower()
            return [s for s in suggestions if lower_query in s.lower()]

        return suggestions[:5]
