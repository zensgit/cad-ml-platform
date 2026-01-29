"""
Knowledge Retriever for CAD-ML Assistant.

Retrieves relevant knowledge from domain-specific databases based on
analyzed query intent and extracted entities.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from .query_analyzer import AnalyzedQuery, QueryIntent


class RetrievalSource(str, Enum):
    """Knowledge source identifiers."""

    MATERIALS = "materials"
    TOLERANCE = "tolerance"
    THREADS = "threads"
    BEARINGS = "bearings"
    SEALS = "seals"
    MACHINING = "machining"
    GDT = "gdt"


@dataclass
class RetrievalResult:
    """Result from knowledge retrieval."""

    source: RetrievalSource
    relevance: float  # 0.0 to 1.0
    data: Dict[str, Any]
    summary: str  # Human-readable summary
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeRetriever:
    """
    Retrieves relevant knowledge from CAD-ML domain databases.

    Supports retrieval from:
    - Material database (classifier.py)
    - Tolerance knowledge (tolerance module)
    - Standard parts (standards module)
    - Machining parameters (machining module)

    Example:
        >>> retriever = KnowledgeRetriever()
        >>> results = retriever.retrieve(analyzed_query)
        >>> for r in results:
        ...     print(f"{r.source}: {r.summary}")
    """

    def __init__(self):
        self._retrievers = {
            RetrievalSource.MATERIALS: self._retrieve_materials,
            RetrievalSource.TOLERANCE: self._retrieve_tolerance,
            RetrievalSource.THREADS: self._retrieve_threads,
            RetrievalSource.BEARINGS: self._retrieve_bearings,
            RetrievalSource.SEALS: self._retrieve_seals,
            RetrievalSource.MACHINING: self._retrieve_machining,
        }

    def retrieve(
        self,
        query: AnalyzedQuery,
        max_results: int = 5,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant knowledge based on analyzed query.

        Args:
            query: Analyzed query with intent and entities
            max_results: Maximum number of results to return

        Returns:
            List of RetrievalResult sorted by relevance
        """
        results = []

        # Determine which sources to query based on intent
        sources = self._get_sources_for_intent(query.intent)

        for source in sources:
            retriever_fn = self._retrievers.get(source)
            if retriever_fn:
                try:
                    source_results = retriever_fn(query)
                    results.extend(source_results)
                except Exception as e:
                    # Log error but continue with other sources
                    print(f"Retrieval error from {source}: {e}")

        # Sort by relevance and limit
        results.sort(key=lambda x: x.relevance, reverse=True)
        return results[:max_results]

    def _get_sources_for_intent(self, intent: QueryIntent) -> List[RetrievalSource]:
        """Map intent to knowledge sources."""
        mapping = {
            QueryIntent.MATERIAL_PROPERTY: [RetrievalSource.MATERIALS],
            QueryIntent.MATERIAL_SELECTION: [RetrievalSource.MATERIALS, RetrievalSource.MACHINING],
            QueryIntent.MATERIAL_COMPARISON: [RetrievalSource.MATERIALS],
            QueryIntent.TOLERANCE_LOOKUP: [RetrievalSource.TOLERANCE],
            QueryIntent.FIT_SELECTION: [RetrievalSource.TOLERANCE],
            QueryIntent.FIT_CALCULATION: [RetrievalSource.TOLERANCE],
            QueryIntent.THREAD_SPEC: [RetrievalSource.THREADS],
            QueryIntent.BEARING_SPEC: [RetrievalSource.BEARINGS],
            QueryIntent.SEAL_SPEC: [RetrievalSource.SEALS],
            QueryIntent.CUTTING_PARAMETERS: [RetrievalSource.MACHINING, RetrievalSource.MATERIALS],
            QueryIntent.TOOL_SELECTION: [RetrievalSource.MACHINING],
            QueryIntent.PROCESS_ROUTE: [RetrievalSource.MACHINING, RetrievalSource.MATERIALS],
        }
        return mapping.get(intent, [RetrievalSource.MATERIALS])

    def _retrieve_materials(self, query: AnalyzedQuery) -> List[RetrievalResult]:
        """Retrieve from materials database."""
        results = []

        try:
            from src.core.materials.classifier import MATERIAL_DATABASE, MATERIAL_COST_DATA
        except ImportError:
            return results

        # Look for material grade in entities
        material_grade = query.entities.get("material_grade", "").upper()

        if material_grade:
            # Direct lookup
            for grade, props in MATERIAL_DATABASE.items():
                if material_grade in grade.upper() or material_grade in props.get("aliases", []):
                    cost_data = MATERIAL_COST_DATA.get(grade, {})
                    results.append(RetrievalResult(
                        source=RetrievalSource.MATERIALS,
                        relevance=0.95,
                        data={
                            "grade": grade,
                            "properties": props,
                            "cost": cost_data,
                        },
                        summary=f"材料 {grade}: {props.get('name', '')}",
                        metadata={"match_type": "direct"},
                    ))
                    break
        else:
            # Keyword search in material names
            for keyword in query.keywords:
                for grade, props in MATERIAL_DATABASE.items():
                    name = props.get("name", "")
                    if keyword.lower() in grade.lower() or keyword.lower() in name.lower():
                        results.append(RetrievalResult(
                            source=RetrievalSource.MATERIALS,
                            relevance=0.7,
                            data={"grade": grade, "properties": props},
                            summary=f"材料 {grade}: {name}",
                            metadata={"match_type": "keyword", "keyword": keyword},
                        ))
                        if len(results) >= 3:
                            break

        return results

    def _retrieve_tolerance(self, query: AnalyzedQuery) -> List[RetrievalResult]:
        """Retrieve from tolerance knowledge base."""
        results = []

        try:
            from src.core.knowledge.tolerance import (
                get_tolerance_value,
                get_fit_deviations,
                get_common_fits,
                select_fit_for_application,
            )
        except ImportError:
            return results

        # Check for IT grade query
        it_grade = query.entities.get("it_grade")
        diameter = query.entities.get("diameter")

        if it_grade:
            grade_str = f"IT{it_grade}"
            if diameter:
                try:
                    d = float(diameter)
                    value = get_tolerance_value(d, grade_str)
                    if value:
                        results.append(RetrievalResult(
                            source=RetrievalSource.TOLERANCE,
                            relevance=0.95,
                            data={"grade": grade_str, "diameter": d, "tolerance_um": value},
                            summary=f"{grade_str}公差 @ {d}mm = {value}μm",
                        ))
                except ValueError:
                    pass

        # Check for fit query
        hole_tol = query.entities.get("hole_tolerance")
        shaft_tol = query.entities.get("shaft_tolerance")

        if hole_tol and shaft_tol:
            fit_code = f"{hole_tol}/{shaft_tol}"
            if diameter:
                try:
                    d = float(diameter)
                    deviations = get_fit_deviations(fit_code, d)
                    if deviations:
                        results.append(RetrievalResult(
                            source=RetrievalSource.TOLERANCE,
                            relevance=0.95,
                            data={
                                "fit_code": fit_code,
                                "diameter": d,
                                "deviations": {
                                    "hole_upper": deviations.hole_upper_deviation_um,
                                    "hole_lower": deviations.hole_lower_deviation_um,
                                    "shaft_upper": deviations.shaft_upper_deviation_um,
                                    "shaft_lower": deviations.shaft_lower_deviation_um,
                                    "max_clearance": deviations.max_clearance_um,
                                    "min_clearance": deviations.min_clearance_um,
                                },
                            },
                            summary=f"{fit_code} @ {d}mm: 间隙 {deviations.min_clearance_um}~{deviations.max_clearance_um}μm",
                        ))
                except (ValueError, AttributeError):
                    pass

        # If no specific query, return common fits info
        if not results and query.intent in [QueryIntent.FIT_SELECTION, QueryIntent.FIT_CALCULATION]:
            fits = get_common_fits()
            for code, data in list(fits.items())[:3]:
                results.append(RetrievalResult(
                    source=RetrievalSource.TOLERANCE,
                    relevance=0.6,
                    data={"fit_code": code, "info": data},
                    summary=f"{code}: {data.get('name_zh', '')}",
                ))

        return results

    def _retrieve_threads(self, query: AnalyzedQuery) -> List[RetrievalResult]:
        """Retrieve from thread specifications."""
        results = []

        try:
            from src.core.knowledge.standards import get_thread_spec, get_tap_drill_size
        except ImportError:
            return results

        thread_d = query.entities.get("thread_diameter")
        thread_p = query.entities.get("thread_pitch")

        if thread_d:
            if thread_p:
                designation = f"M{thread_d}x{thread_p}"
            else:
                designation = f"M{thread_d}"

            spec = get_thread_spec(designation)
            if spec:
                results.append(RetrievalResult(
                    source=RetrievalSource.THREADS,
                    relevance=0.95,
                    data={
                        "designation": spec.designation,
                        "nominal_diameter": spec.nominal_diameter,
                        "pitch": spec.pitch,
                        "tap_drill": spec.tap_drill_size,
                        "pitch_diameter": spec.pitch_diameter,
                        "minor_diameter": spec.minor_diameter_ext,
                    },
                    summary=f"{spec.designation}: 螺距{spec.pitch}mm, 底孔{spec.tap_drill_size}mm",
                ))

        return results

    def _retrieve_bearings(self, query: AnalyzedQuery) -> List[RetrievalResult]:
        """Retrieve from bearing specifications."""
        results = []

        try:
            from src.core.knowledge.standards import get_bearing_spec, get_bearing_by_bore
        except ImportError:
            return results

        bearing_id = query.entities.get("bearing")
        diameter = query.entities.get("diameter")

        if bearing_id:
            spec = get_bearing_spec(bearing_id)
            if spec:
                results.append(RetrievalResult(
                    source=RetrievalSource.BEARINGS,
                    relevance=0.95,
                    data={
                        "designation": spec.designation,
                        "bore": spec.bore_d,
                        "outer_d": spec.outer_d,
                        "width": spec.width_b,
                        "dynamic_load": spec.dynamic_load_c,
                        "static_load": spec.static_load_c0,
                    },
                    summary=f"{spec.designation}: d={spec.bore_d}mm, D={spec.outer_d}mm, B={spec.width_b}mm",
                ))
        elif diameter:
            try:
                d = float(diameter)
                bearings = get_bearing_by_bore(d)
                for b in bearings[:3]:
                    results.append(RetrievalResult(
                        source=RetrievalSource.BEARINGS,
                        relevance=0.8,
                        data={
                            "designation": b.designation,
                            "bore": b.bore_d,
                            "outer_d": b.outer_d,
                            "width": b.width_b,
                        },
                        summary=f"{b.designation}: d={b.bore_d}mm, D={b.outer_d}mm",
                    ))
            except ValueError:
                pass

        return results

    def _retrieve_seals(self, query: AnalyzedQuery) -> List[RetrievalResult]:
        """Retrieve from seal specifications."""
        results = []

        try:
            from src.core.knowledge.standards import get_oring_spec, get_oring_by_id
        except ImportError:
            return results

        oring_id = query.entities.get("oring_id")
        oring_cs = query.entities.get("oring_cs")

        if oring_id and oring_cs:
            designation = f"{oring_id}x{oring_cs}"
            spec = get_oring_spec(designation)
            if spec:
                results.append(RetrievalResult(
                    source=RetrievalSource.SEALS,
                    relevance=0.95,
                    data={
                        "designation": designation,
                        "inner_diameter": spec.inner_diameter,
                        "cross_section": spec.cross_section,
                        "groove_width": spec.groove_width_static,
                        "groove_depth": spec.groove_depth_static,
                    },
                    summary=f"O形圈 {designation}: ID={spec.inner_diameter}mm, CS={spec.cross_section}mm",
                ))
        elif oring_id:
            try:
                d = float(oring_id)
                orings = get_oring_by_id(d)
                for o in orings[:3]:
                    results.append(RetrievalResult(
                        source=RetrievalSource.SEALS,
                        relevance=0.7,
                        data={
                            "inner_diameter": o.inner_diameter,
                            "cross_section": o.cross_section,
                        },
                        summary=f"O形圈 {o.inner_diameter}x{o.cross_section}",
                    ))
            except ValueError:
                pass

        return results

    def _retrieve_machining(self, query: AnalyzedQuery) -> List[RetrievalResult]:
        """Retrieve from machining knowledge base."""
        results = []

        try:
            from src.core.knowledge.machining import (
                get_cutting_parameters,
                get_tool_recommendation,
                get_machinability,
            )
        except ImportError:
            return results

        material_grade = query.entities.get("material_grade", "").upper()

        # Try to get machinability data
        if material_grade:
            # Map common grades to machinability keys
            grade_mapping = {
                "304": "austenitic_stainless",
                "316": "austenitic_stainless",
                "316L": "austenitic_stainless",
                "Q235": "low_carbon_steel",
                "45": "medium_carbon_steel",
                "40CR": "alloy_steel",
                "6061": "aluminum_wrought",
                "7075": "aluminum_wrought",
            }

            mat_key = grade_mapping.get(material_grade)
            if mat_key:
                mat = get_machinability(mat_key)
                if mat:
                    # Get cutting parameters
                    params = get_cutting_parameters("turning_rough", mat.material_group)
                    tool = get_tool_recommendation(mat.material_group, "roughing")

                    results.append(RetrievalResult(
                        source=RetrievalSource.MACHINING,
                        relevance=0.9,
                        data={
                            "material": mat_key,
                            "machinability_rating": mat.machinability_rating,
                            "iso_group": mat.material_group,
                            "cutting_speed": params.cutting_speed_recommended if params else None,
                            "feed": params.feed_recommended if params else None,
                            "tool_material": tool.tool_material.value if tool else None,
                        },
                        summary=f"{material_grade}加工: 切削速度{params.cutting_speed_recommended if params else 'N/A'}m/min, 可加工性{mat.machinability_rating}%",
                    ))

        return results

    def retrieve_by_source(
        self,
        source: RetrievalSource,
        query: AnalyzedQuery,
    ) -> List[RetrievalResult]:
        """Retrieve from a specific knowledge source."""
        retriever_fn = self._retrievers.get(source)
        if retriever_fn:
            return retriever_fn(query)
        return []
