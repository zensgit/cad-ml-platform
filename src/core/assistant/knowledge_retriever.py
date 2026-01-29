"""
Knowledge Retriever for CAD-ML Assistant.

Retrieves relevant knowledge from domain-specific databases based on
analyzed query intent and extracted entities.

Supports both keyword-based and semantic (embedding) retrieval.
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
    DESIGN_STANDARDS = "design_standards"
    GDT = "gdt"


class RetrievalMode(str, Enum):
    """Retrieval mode."""

    KEYWORD = "keyword"  # Traditional keyword matching
    SEMANTIC = "semantic"  # Embedding-based similarity
    HYBRID = "hybrid"  # Combine both approaches


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
    - Design standards (design_standards module)

    Retrieval modes:
    - keyword: Traditional entity/keyword matching (fast, precise)
    - semantic: Embedding-based similarity search (flexible, semantic)
    - hybrid: Combine both approaches (best accuracy)

    Example:
        >>> retriever = KnowledgeRetriever()
        >>> results = retriever.retrieve(analyzed_query)
        >>> for r in results:
        ...     print(f"{r.source}: {r.summary}")
    """

    def __init__(self, mode: RetrievalMode = RetrievalMode.HYBRID):
        self.mode = mode
        self._retrievers = {
            RetrievalSource.MATERIALS: self._retrieve_materials,
            RetrievalSource.TOLERANCE: self._retrieve_tolerance,
            RetrievalSource.THREADS: self._retrieve_threads,
            RetrievalSource.BEARINGS: self._retrieve_bearings,
            RetrievalSource.SEALS: self._retrieve_seals,
            RetrievalSource.MACHINING: self._retrieve_machining,
            RetrievalSource.DESIGN_STANDARDS: self._retrieve_design_standards,
        }
        self._semantic_retriever = None

    def _get_semantic_retriever(self):
        """Lazy-load semantic retriever."""
        if self._semantic_retriever is None:
            try:
                from .embedding_retriever import get_semantic_retriever
                self._semantic_retriever = get_semantic_retriever()
            except Exception:
                self._semantic_retriever = None
        return self._semantic_retriever

    def retrieve(
        self,
        query: AnalyzedQuery,
        max_results: int = 5,
        mode: Optional[RetrievalMode] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant knowledge based on analyzed query.

        Args:
            query: Analyzed query with intent and entities
            max_results: Maximum number of results to return
            mode: Override retrieval mode (optional)

        Returns:
            List of RetrievalResult sorted by relevance
        """
        effective_mode = mode or self.mode
        results = []

        # 1. Keyword-based retrieval (always run for precise entity matches)
        if effective_mode in [RetrievalMode.KEYWORD, RetrievalMode.HYBRID]:
            keyword_results = self._retrieve_by_keyword(query)
            results.extend(keyword_results)

        # 2. Semantic retrieval (for flexible matching)
        if effective_mode in [RetrievalMode.SEMANTIC, RetrievalMode.HYBRID]:
            semantic_results = self._retrieve_by_semantic(query)

            # In hybrid mode, add semantic results that don't duplicate keyword results
            if effective_mode == RetrievalMode.HYBRID:
                existing_ids = {r.metadata.get("id") for r in results}
                for r in semantic_results:
                    if r.metadata.get("id") not in existing_ids:
                        # Slightly lower relevance for semantic-only matches
                        r.relevance *= 0.9
                        results.append(r)
            else:
                results.extend(semantic_results)

        # Sort by relevance and limit
        results.sort(key=lambda x: x.relevance, reverse=True)
        return results[:max_results]

    def _retrieve_by_keyword(self, query: AnalyzedQuery) -> List[RetrievalResult]:
        """Retrieve using traditional keyword/entity matching."""
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

        return results

    def _retrieve_by_semantic(self, query: AnalyzedQuery) -> List[RetrievalResult]:
        """Retrieve using semantic similarity search."""
        results = []

        semantic_retriever = self._get_semantic_retriever()
        if semantic_retriever is None:
            return results

        try:
            # Use original query text for semantic search
            search_results = semantic_retriever.search(
                query=query.original_query,
                max_results=10,
            )

            for item, score in search_results:
                # Map source string to RetrievalSource enum
                source_map = {
                    "materials": RetrievalSource.MATERIALS,
                    "tolerance": RetrievalSource.TOLERANCE,
                    "threads": RetrievalSource.THREADS,
                    "bearings": RetrievalSource.BEARINGS,
                    "seals": RetrievalSource.SEALS,
                    "machining": RetrievalSource.MACHINING,
                    "design_standards": RetrievalSource.DESIGN_STANDARDS,
                }
                source = source_map.get(item.source, RetrievalSource.MATERIALS)

                results.append(RetrievalResult(
                    source=source,
                    relevance=score,
                    data=item.data,
                    summary=item.text[:100],
                    metadata={
                        "id": item.id,
                        "match_type": "semantic",
                        "similarity_score": score,
                    },
                ))

        except Exception as e:
            print(f"Semantic retrieval error: {e}")

        return results

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

    def _retrieve_design_standards(self, query: AnalyzedQuery) -> List[RetrievalResult]:
        """Retrieve from design standards knowledge base."""
        results = []

        try:
            from src.core.knowledge.design_standards import (
                get_ra_value,
                get_surface_finish_for_application,
                get_linear_tolerance,
                get_angular_tolerance,
                get_preferred_diameter,
                get_standard_chamfer,
                get_standard_fillet,
                SurfaceFinishGrade,
                GeneralToleranceClass,
            )
        except ImportError:
            return results

        # Surface finish query
        surface_grade = query.entities.get("surface_grade")
        if surface_grade:
            try:
                grade = SurfaceFinishGrade(surface_grade.upper())
                ra = get_ra_value(grade)
                results.append(RetrievalResult(
                    source=RetrievalSource.DESIGN_STANDARDS,
                    relevance=0.95,
                    data={"grade": grade.value, "ra_um": ra},
                    summary=f"表面粗糙度 {grade.value}: Ra = {ra} μm",
                    metadata={"id": f"surface_{grade.value}"},
                ))
            except (ValueError, KeyError):
                pass

        # General tolerance query
        dimension = query.entities.get("dimension")
        tolerance_class = query.entities.get("tolerance_class", "m")
        if dimension:
            try:
                d = float(dimension)
                tol_class = GeneralToleranceClass(tolerance_class.lower())
                tol = get_linear_tolerance(d, tol_class)
                if tol:
                    results.append(RetrievalResult(
                        source=RetrievalSource.DESIGN_STANDARDS,
                        relevance=0.95,
                        data={"dimension": d, "class": tol_class.value, "tolerance": tol},
                        summary=f"一般公差 {tol_class.value}级 @ {d}mm: ±{tol}mm",
                        metadata={"id": f"gen_tol_{d}_{tol_class.value}"},
                    ))
            except (ValueError, KeyError):
                pass

        # Preferred diameter query
        target_diameter = query.entities.get("target_diameter")
        if target_diameter:
            try:
                d = float(target_diameter)
                preferred = get_preferred_diameter(d)
                results.append(RetrievalResult(
                    source=RetrievalSource.DESIGN_STANDARDS,
                    relevance=0.9,
                    data={"target": d, "preferred": preferred},
                    summary=f"优选直径: {d}mm → {preferred}mm (ISO 497)",
                    metadata={"id": f"pref_dia_{d}"},
                ))
            except ValueError:
                pass

        # Chamfer/fillet query
        chamfer_size = query.entities.get("chamfer_size")
        fillet_size = query.entities.get("fillet_size")

        if chamfer_size:
            try:
                size = float(chamfer_size)
                result = get_standard_chamfer(size)
                if result:
                    results.append(RetrievalResult(
                        source=RetrievalSource.DESIGN_STANDARDS,
                        relevance=0.9,
                        data=result,
                        summary=f"标准倒角: {size}mm → {result['designation']}",
                        metadata={"id": f"chamfer_{size}"},
                    ))
            except ValueError:
                pass

        if fillet_size:
            try:
                size = float(fillet_size)
                result = get_standard_fillet(size)
                if result:
                    results.append(RetrievalResult(
                        source=RetrievalSource.DESIGN_STANDARDS,
                        relevance=0.9,
                        data=result,
                        summary=f"标准圆角: {size}mm → {result['designation']}",
                        metadata={"id": f"fillet_{size}"},
                    ))
            except ValueError:
                pass

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
