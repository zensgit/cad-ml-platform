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
    WELDING = "welding"
    HEAT_TREATMENT = "heat_treatment"
    SURFACE_TREATMENT = "surface_treatment"
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
            RetrievalSource.WELDING: self._retrieve_welding,
            RetrievalSource.HEAT_TREATMENT: self._retrieve_heat_treatment,
            RetrievalSource.SURFACE_TREATMENT: self._retrieve_surface_treatment,
            RetrievalSource.GDT: self._retrieve_gdt,
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
            QueryIntent.WELDING_PARAMETERS: [RetrievalSource.WELDING],
            QueryIntent.WELDING_JOINT: [RetrievalSource.WELDING],
            QueryIntent.WELDABILITY: [RetrievalSource.WELDING, RetrievalSource.MATERIALS],
            QueryIntent.HEAT_TREATMENT: [RetrievalSource.HEAT_TREATMENT, RetrievalSource.MATERIALS],
            QueryIntent.HARDENING: [RetrievalSource.HEAT_TREATMENT],
            QueryIntent.ANNEALING: [RetrievalSource.HEAT_TREATMENT],
            QueryIntent.ELECTROPLATING: [RetrievalSource.SURFACE_TREATMENT],
            QueryIntent.ANODIZING: [RetrievalSource.SURFACE_TREATMENT],
            QueryIntent.COATING: [RetrievalSource.SURFACE_TREATMENT],
            QueryIntent.GDT_INTERPRETATION: [RetrievalSource.GDT],
            QueryIntent.GDT_APPLICATION: [RetrievalSource.GDT, RetrievalSource.TOLERANCE],
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
                get_fundamental_deviation,
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

        # Fundamental deviation lookup (e.g., H7 25mm 基本偏差)
        tol_symbol = query.entities.get("tolerance_symbol")
        tol_grade = query.entities.get("tolerance_grade")
        if not results and tol_symbol and tol_grade and diameter:
            try:
                d = float(diameter)
                grade_str = f"IT{tol_grade}"
                tolerance = get_tolerance_value(d, grade_str)
                symbol = tol_symbol.strip()
                if tolerance is not None and symbol:
                    symbol_upper = symbol.upper()
                    symbol_lower = symbol.lower()
                    is_hole = symbol.isupper()

                    if is_hole:
                        if symbol_upper == "H":
                            lower = 0.0
                            upper = tolerance
                        elif symbol_upper == "JS":
                            upper = tolerance / 2
                            lower = -tolerance / 2
                        else:
                            fund_dev = get_fundamental_deviation(symbol_upper, d)
                            if fund_dev is None:
                                fund_dev = 0.0
                            lower = fund_dev
                            upper = fund_dev + tolerance

                        summary = (
                            f"{symbol_upper}{tol_grade} @ {d}mm: "
                            f"EI={lower:.0f}μm, ES={upper:.0f}μm"
                        )
                        results.append(
                            RetrievalResult(
                                source=RetrievalSource.TOLERANCE,
                                relevance=0.9,
                                data={
                                    "symbol": symbol_upper,
                                    "grade": tol_grade,
                                    "diameter": d,
                                    "tolerance_um": tolerance,
                                    "lower_deviation_um": lower,
                                    "upper_deviation_um": upper,
                                    "type": "hole",
                                },
                                summary=summary,
                            )
                        )
                    else:
                        fund_dev = get_fundamental_deviation(symbol_lower, d)
                        if fund_dev is None:
                            fund_dev = 0.0
                        if symbol_lower in ["g", "f", "e", "d", "c", "h", "a", "b"]:
                            upper = fund_dev
                            lower = fund_dev - tolerance
                        else:
                            lower = fund_dev
                            upper = fund_dev + tolerance

                        summary = (
                            f"{symbol_lower}{tol_grade} @ {d}mm: "
                            f"ei={lower:.0f}μm, es={upper:.0f}μm"
                        )
                        results.append(
                            RetrievalResult(
                                source=RetrievalSource.TOLERANCE,
                                relevance=0.9,
                                data={
                                    "symbol": symbol_lower,
                                    "grade": tol_grade,
                                    "diameter": d,
                                    "tolerance_um": tolerance,
                                    "lower_deviation_um": lower,
                                    "upper_deviation_um": upper,
                                    "type": "shaft",
                                },
                                summary=summary,
                            )
                        )
            except ValueError:
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

        # Precision rules (e.g., GB/T 1804/1184) from knowledge seed
        results.extend(self._retrieve_precision_rules(query))

        return results

    def _retrieve_precision_rules(self, query: AnalyzedQuery) -> List[RetrievalResult]:
        """Retrieve precision/tolerance rules from knowledge seed."""
        results: List[RetrievalResult] = []
        rules_path = "data/knowledge/precision_rules.json"
        try:
            import json
            from pathlib import Path
            path = Path(rules_path)
            if not path.exists():
                return results
            data = json.loads(path.read_text(encoding="utf-8"))
            rules = data.get("rules", [])
        except Exception:
            return results

        text = query.original_query
        text_lower = text.lower()

        for rule in rules:
            if not rule.get("enabled", True):
                continue
            matched = False

            for keyword in rule.get("keywords", []):
                if keyword.lower() in text_lower:
                    matched = True
                    break

            if not matched:
                for pattern in rule.get("ocr_patterns", []):
                    try:
                        if __import__("re").search(pattern, text, __import__("re").IGNORECASE):
                            matched = True
                            break
                    except Exception:
                        continue

            if not matched:
                continue

            metadata = rule.get("metadata", {})
            tolerance_class = metadata.get("tolerance_class")
            gdt_class = metadata.get("gdt_class")
            if tolerance_class:
                summary = f"未注公差按 GB/T 1804-{tolerance_class.upper()}（一般公差{tolerance_class.upper()}级）"
            elif gdt_class:
                summary = f"未注形位公差按 GB/T 1184-{gdt_class.upper()}"
            else:
                summary = rule.get("description") or rule.get("name") or "公差规则"

            results.append(
                RetrievalResult(
                    source=RetrievalSource.TOLERANCE,
                    relevance=0.75,
                    data={
                        "rule_id": rule.get("id"),
                        "name": rule.get("name"),
                        "description": rule.get("description"),
                        "metadata": metadata,
                    },
                    summary=summary,
                )
            )

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

    def _retrieve_welding(self, query: AnalyzedQuery) -> List[RetrievalResult]:
        """Retrieve from welding knowledge base."""
        results = []

        try:
            from src.core.knowledge.welding import (
                get_welding_parameters,
                get_filler_material,
                get_joint_design,
                get_weldability,
                get_preheat_temperature,
                calculate_heat_input,
                WeldingProcess,
                GrooveType,
            )
        except ImportError:
            return results

        material_grade = query.entities.get("material_grade", "").upper()
        thickness = query.entities.get("thickness")

        # Map material to welding material type
        material_type_mapping = {
            "Q235": "carbon_steel",
            "Q345": "carbon_steel",
            "45": "carbon_steel",
            "304": "stainless_steel",
            "316": "stainless_steel",
            "316L": "stainless_steel",
            "6061": "aluminum",
            "7075": "aluminum",
        }

        # Welding parameters query
        if query.intent == QueryIntent.WELDING_PARAMETERS:
            base_material = material_type_mapping.get(material_grade, "carbon_steel")
            t = float(thickness) if thickness else 6.0

            for process in [WeldingProcess.GMAW, WeldingProcess.SMAW, WeldingProcess.GTAW]:
                params = get_welding_parameters(process, base_material, t)
                if params:
                    results.append(RetrievalResult(
                        source=RetrievalSource.WELDING,
                        relevance=0.9,
                        data={
                            "process": process.value,
                            "material": base_material,
                            "thickness": t,
                            "current": params.current_recommended,
                            "voltage": params.voltage_recommended,
                            "speed": params.speed_recommended,
                            "electrode_diameter": params.electrode_diameter,
                        },
                        summary=f"{process.value}焊接 {base_material} {t}mm: {params.current_recommended}A, {params.voltage_recommended}V",
                        metadata={"id": f"weld_{process.value}_{base_material}"},
                    ))

            # Filler material
            fillers = get_filler_material(base_material, "GMAW")
            if fillers:
                results.append(RetrievalResult(
                    source=RetrievalSource.WELDING,
                    relevance=0.85,
                    data={"filler_materials": fillers, "base_material": base_material},
                    summary=f"{base_material}焊丝推荐: {', '.join(fillers[:3])}",
                    metadata={"id": f"filler_{base_material}"},
                ))

        # Joint design query
        elif query.intent == QueryIntent.WELDING_JOINT:
            t = float(thickness) if thickness else 10.0
            for groove_type in [GrooveType.SINGLE_V, GrooveType.DOUBLE_V, GrooveType.SQUARE]:
                design = get_joint_design(groove_type, t)
                if design:
                    results.append(RetrievalResult(
                        source=RetrievalSource.WELDING,
                        relevance=0.9,
                        data={
                            "groove_type": groove_type.value,
                            "thickness": t,
                            "groove_angle": design.groove_angle,
                            "root_gap": design.root_gap,
                            "root_face": design.root_face,
                        },
                        summary=f"{groove_type.value}坡口 {t}mm: 角度{design.groove_angle}°, 根部间隙{design.root_gap}mm",
                        metadata={"id": f"joint_{groove_type.value}_{t}"},
                    ))

        # Weldability query
        elif query.intent == QueryIntent.WELDABILITY:
            if material_grade:
                # Map to weldability key
                weld_key_mapping = {
                    "Q235": "Q235",
                    "Q345": "Q345",
                    "45": "45",
                    "304": "304",
                    "316L": "316L",
                    "40CR": "40Cr",
                }
                weld_key = weld_key_mapping.get(material_grade, material_grade)
                weldability = get_weldability(weld_key)

                if weldability:
                    t = float(thickness) if thickness else 20.0
                    preheat = get_preheat_temperature(weld_key, thickness=t)

                    results.append(RetrievalResult(
                        source=RetrievalSource.WELDING,
                        relevance=0.95,
                        data={
                            "material": weld_key,
                            "weldability_class": weldability.weldability.value,
                            "preheat_required": weldability.preheat_required,
                            "preheat_temp": preheat,
                            "carbon_equivalent": weldability.carbon_equivalent,
                            "special_requirements": weldability.special_requirements,
                        },
                        summary=f"{weld_key}焊接性: {weldability.weldability.value}, 预热{preheat[0]}-{preheat[1]}°C",
                        metadata={"id": f"weldability_{weld_key}"},
                    ))

        return results

    def _retrieve_heat_treatment(self, query: AnalyzedQuery) -> List[RetrievalResult]:
        """Retrieve from heat treatment knowledge base."""
        results = []

        try:
            from src.core.knowledge.heat_treatment import (
                get_heat_treatment_parameters,
                get_hardenability,
                get_tempering_temperature,
                get_annealing_parameters,
                calculate_hardness_after_tempering,
                HeatTreatmentProcess,
                AnnealingType,
            )
        except ImportError:
            return results

        material_grade = query.entities.get("material_grade", "").upper()
        target_hardness = query.entities.get("target_hardness")

        # Map material to heat treatment key
        ht_key_mapping = {
            "45": "45",
            "45钢": "45",
            "40CR": "40Cr",
            "CR12MOV": "Cr12MoV",
            "304": "304",
            "6061": "6061",
            "GCR15": "GCr15",
        }
        ht_key = ht_key_mapping.get(material_grade, material_grade)

        # Hardening query
        if query.intent in [QueryIntent.HARDENING, QueryIntent.HEAT_TREATMENT]:
            params = get_heat_treatment_parameters(ht_key, HeatTreatmentProcess.QUENCH_HARDENING)
            if params:
                results.append(RetrievalResult(
                    source=RetrievalSource.HEAT_TREATMENT,
                    relevance=0.95,
                    data={
                        "material": ht_key,
                        "process": "quench_hardening",
                        "temperature": params.temperature_recommended,
                        "temperature_range": (params.temperature_min, params.temperature_max),
                        "quench_media": params.quench_media.value if params.quench_media else None,
                        "hardness_range": params.hardness_range,
                    },
                    summary=f"{ht_key}淬火: {params.temperature_recommended}°C, {params.quench_media.value if params.quench_media else 'N/A'}",
                    metadata={"id": f"quench_{ht_key}"},
                ))

            # Hardenability data
            hardenability = get_hardenability(ht_key)
            if hardenability:
                results.append(RetrievalResult(
                    source=RetrievalSource.HEAT_TREATMENT,
                    relevance=0.9,
                    data={
                        "material": ht_key,
                        "hardenability_class": hardenability.hardenability_class.value,
                        "critical_diameter_water": hardenability.critical_diameter_water,
                        "critical_diameter_oil": hardenability.critical_diameter_oil,
                        "as_quenched_hardness": hardenability.as_quenched_hardness_max,
                    },
                    summary=f"{ht_key}淬透性: {hardenability.hardenability_class.value}, 临界直径(油){hardenability.critical_diameter_oil}mm",
                    metadata={"id": f"hardenability_{ht_key}"},
                ))

            # Tempering recommendation
            if target_hardness:
                try:
                    h = float(target_hardness)
                    temp_range = get_tempering_temperature(ht_key, h)
                    if temp_range:
                        results.append(RetrievalResult(
                            source=RetrievalSource.HEAT_TREATMENT,
                            relevance=0.9,
                            data={
                                "material": ht_key,
                                "target_hardness": h,
                                "tempering_temp_range": temp_range,
                            },
                            summary=f"{ht_key}回火至HRC{h}: {temp_range[0]}-{temp_range[1]}°C",
                            metadata={"id": f"temper_{ht_key}_{h}"},
                        ))
                except ValueError:
                    pass

        # Annealing query
        elif query.intent == QueryIntent.ANNEALING:
            for ann_type in [AnnealingType.FULL, AnnealingType.SPHEROIDIZING, AnnealingType.STRESS_RELIEF]:
                params = get_annealing_parameters(ht_key, ann_type)
                if params:
                    results.append(RetrievalResult(
                        source=RetrievalSource.HEAT_TREATMENT,
                        relevance=0.9,
                        data={
                            "material": ht_key,
                            "annealing_type": ann_type.value,
                            "temperature": params.temperature_recommended,
                            "temperature_range": (params.temperature_min, params.temperature_max),
                            "cooling_method": params.cooling_method,
                            "hardness_after": params.hardness_after,
                        },
                        summary=f"{ht_key}{ann_type.value}: {params.temperature_recommended}°C, {params.cooling_method}",
                        metadata={"id": f"anneal_{ht_key}_{ann_type.value}"},
                    ))

        return results

    def _retrieve_surface_treatment(self, query: AnalyzedQuery) -> List[RetrievalResult]:
        """Retrieve from surface treatment knowledge base."""
        results = []

        try:
            from src.core.knowledge.surface_treatment import (
                get_plating_parameters,
                get_plating_thickness,
                recommend_plating_for_application,
                get_anodizing_parameters,
                get_anodizing_colors,
                recommend_anodizing_for_application,
                get_coating_parameters,
                get_coating_for_environment,
                calculate_coating_life,
                PlatingType,
                AnodizingType,
                CoatingType,
            )
        except ImportError:
            return results

        application = query.entities.get("application", "")
        environment = query.entities.get("environment", "")

        # Electroplating query
        if query.intent == QueryIntent.ELECTROPLATING:
            # Common plating types
            plating_types = [
                (PlatingType.ZINC_YELLOW, "镀黄锌"),
                (PlatingType.ZINC_NICKEL, "镀锌镍合金"),
                (PlatingType.NICKEL_BRIGHT, "镀亮镍"),
                (PlatingType.CHROME_HARD, "镀硬铬"),
            ]

            for plating_type, name_zh in plating_types:
                params = get_plating_parameters(plating_type)
                if params:
                    results.append(RetrievalResult(
                        source=RetrievalSource.SURFACE_TREATMENT,
                        relevance=0.85,
                        data={
                            "plating_type": plating_type.value,
                            "name_zh": name_zh,
                            "thickness_typical": params.thickness_typical,
                            "corrosion_resistance_hours": params.corrosion_resistance_hours,
                            "hardness_hv": params.hardness_hv,
                            "current_density": params.current_density_typical,
                        },
                        summary=f"{name_zh}: 厚度{params.thickness_typical}μm, 盐雾{params.corrosion_resistance_hours or 'N/A'}h",
                        metadata={"id": f"plating_{plating_type.value}"},
                    ))

            # Thickness recommendation
            if application:
                thickness_info = get_plating_thickness(application, environment or "indoor")
                if thickness_info:
                    results.append(RetrievalResult(
                        source=RetrievalSource.SURFACE_TREATMENT,
                        relevance=0.9,
                        data=thickness_info,
                        summary=f"{application}镀层厚度推荐: {thickness_info.get('thickness_um', 'N/A')}μm",
                        metadata={"id": f"plating_thickness_{application}"},
                    ))

        # Anodizing query
        elif query.intent == QueryIntent.ANODIZING:
            anodize_types = [
                (AnodizingType.TYPE_II, "Type II硫酸阳极氧化"),
                (AnodizingType.TYPE_III, "Type III硬质阳极氧化"),
                (AnodizingType.TYPE_I, "Type I铬酸阳极氧化"),
            ]

            for anodize_type, name_zh in anodize_types:
                params = get_anodizing_parameters(anodize_type)
                if params:
                    results.append(RetrievalResult(
                        source=RetrievalSource.SURFACE_TREATMENT,
                        relevance=0.85,
                        data={
                            "anodize_type": anodize_type.value,
                            "name_zh": name_zh,
                            "thickness_typical": params.thickness_typical,
                            "hardness_hv": params.hardness_hv,
                            "acid_type": params.acid_type,
                            "temperature": params.temperature_typical,
                        },
                        summary=f"{name_zh}: 厚度{params.thickness_typical}μm, {params.acid_type}",
                        metadata={"id": f"anodize_{anodize_type.value}"},
                    ))

            # Available colors for Type II
            colors = get_anodizing_colors(AnodizingType.TYPE_II)
            if colors:
                results.append(RetrievalResult(
                    source=RetrievalSource.SURFACE_TREATMENT,
                    relevance=0.8,
                    data={"colors": colors, "anodize_type": "type_ii"},
                    summary=f"Type II可染色: {', '.join(c['name_zh'] for c in colors[:5])}",
                    metadata={"id": "anodize_colors_type_ii"},
                ))

        # Coating query
        elif query.intent == QueryIntent.COATING:
            coating_types = [
                (CoatingType.POWDER_POLYESTER, "聚酯粉末涂料"),
                (CoatingType.EPOXY, "环氧涂料"),
                (CoatingType.POLYURETHANE, "聚氨酯涂料"),
                (CoatingType.ZINC_FLAKE, "锌铬涂层"),
            ]

            for coating_type, name_zh in coating_types:
                params = get_coating_parameters(coating_type)
                if params:
                    results.append(RetrievalResult(
                        source=RetrievalSource.SURFACE_TREATMENT,
                        relevance=0.85,
                        data={
                            "coating_type": coating_type.value,
                            "name_zh": name_zh,
                            "dft_recommended": params.dft_recommended,
                            "cure_temperature": params.cure_temperature,
                            "application_method": params.application_method,
                        },
                        summary=f"{name_zh}: 干膜厚度{params.dft_recommended}μm",
                        metadata={"id": f"coating_{coating_type.value}"},
                    ))

            # Environment-based recommendation
            if environment:
                env_info = get_coating_for_environment(environment)
                if env_info:
                    results.append(RetrievalResult(
                        source=RetrievalSource.SURFACE_TREATMENT,
                        relevance=0.9,
                        data=env_info,
                        summary=f"{environment}环境涂层: 最小厚度{env_info.get('min_dft_um', 'N/A')}μm",
                        metadata={"id": f"coating_env_{environment}"},
                    ))

        return results

    def _retrieve_gdt(self, query: AnalyzedQuery) -> List[RetrievalResult]:
        """Retrieve from GD&T knowledge base."""
        results = []

        try:
            from src.core.knowledge.gdt import (
                GDTCharacteristic,
                GDTCategory,
                get_gdt_symbol,
                get_all_symbols,
                get_tolerance_zone,
                get_recommended_tolerance,
                get_gdt_for_feature,
                interpret_feature_control_frame,
            )
            from src.core.knowledge.gdt.tolerances import ToleranceGrade
            from src.core.knowledge.gdt.application import FeatureType, get_gdt_rule_one_guidance
        except ImportError:
            return results

        # Map keywords to characteristics
        char_keywords = {
            "平面度": GDTCharacteristic.FLATNESS,
            "直线度": GDTCharacteristic.STRAIGHTNESS,
            "圆度": GDTCharacteristic.CIRCULARITY,
            "圆柱度": GDTCharacteristic.CYLINDRICITY,
            "垂直度": GDTCharacteristic.PERPENDICULARITY,
            "平行度": GDTCharacteristic.PARALLELISM,
            "位置度": GDTCharacteristic.POSITION,
            "同心度": GDTCharacteristic.CONCENTRICITY,
            "对称度": GDTCharacteristic.SYMMETRY,
            "圆跳动": GDTCharacteristic.CIRCULAR_RUNOUT,
            "全跳动": GDTCharacteristic.TOTAL_RUNOUT,
            "倾斜度": GDTCharacteristic.ANGULARITY,
        }

        # Check for specific characteristic queries
        matched_char = None
        for keyword, char in char_keywords.items():
            if keyword in query.original_query:
                matched_char = char
                break

        if matched_char:
            # Get symbol info
            info = get_gdt_symbol(matched_char)
            if info:
                results.append(RetrievalResult(
                    source=RetrievalSource.GDT,
                    relevance=0.95,
                    data={
                        "characteristic": matched_char.value,
                        "name_zh": info.name_zh,
                        "name_en": info.name_en,
                        "symbol": info.symbol_unicode,
                        "category": info.category.value,
                        "requires_datum": info.requires_datum,
                        "applications": info.typical_applications,
                    },
                    summary=f"{info.name_zh} ({info.symbol_unicode}): {info.description_zh}",
                    metadata={"id": f"gdt_{matched_char.value}"},
                ))

                # Get recommended tolerance
                size = query.entities.get("diameter")
                if size:
                    try:
                        s = float(size)
                        tol = get_recommended_tolerance(matched_char, s, ToleranceGrade.K)
                        if tol:
                            results.append(RetrievalResult(
                                source=RetrievalSource.GDT,
                                relevance=0.9,
                                data={
                                    "characteristic": matched_char.value,
                                    "size": s,
                                    "grade": "K",
                                    "recommended_tolerance": tol,
                                },
                                summary=f"{info.name_zh}推荐值 @ {s}mm (K级): {tol}mm",
                                metadata={"id": f"gdt_tol_{matched_char.value}_{s}"},
                            ))
                    except ValueError:
                        pass

        # GD&T interpretation query - return category overview
        if query.intent == QueryIntent.GDT_INTERPRETATION:
            for category in [GDTCategory.FORM, GDTCategory.ORIENTATION, GDTCategory.LOCATION, GDTCategory.RUNOUT]:
                symbols = get_all_symbols(category)
                if symbols:
                    category_names = {
                        GDTCategory.FORM: "形状公差",
                        GDTCategory.ORIENTATION: "方向公差",
                        GDTCategory.LOCATION: "位置公差",
                        GDTCategory.RUNOUT: "跳动公差",
                    }
                    results.append(RetrievalResult(
                        source=RetrievalSource.GDT,
                        relevance=0.8,
                        data={
                            "category": category.value,
                            "name_zh": category_names.get(category, category.value),
                            "symbols": [
                                {"name": s.name_zh, "symbol": s.symbol_unicode, "requires_datum": s.requires_datum}
                                for s in symbols
                            ],
                        },
                        summary=f"{category_names.get(category)}: {', '.join(s.name_zh for s in symbols[:4])}",
                        metadata={"id": f"gdt_category_{category.value}"},
                    ))

            # Add Rule #1 guidance
            rule_one = get_gdt_rule_one_guidance()
            results.append(RetrievalResult(
                source=RetrievalSource.GDT,
                relevance=0.75,
                data=rule_one,
                summary=f"{rule_one['name_zh']}: {rule_one['principle']}",
                metadata={"id": "gdt_rule_one"},
            ))

        # GD&T application query - feature-based recommendations
        if query.intent == QueryIntent.GDT_APPLICATION:
            feature_keywords = {
                "孔": FeatureType.HOLE,
                "轴": FeatureType.SHAFT,
                "平面": FeatureType.FLAT_SURFACE,
                "槽": FeatureType.SLOT,
                "螺纹": FeatureType.THREAD,
            }

            for keyword, feature_type in feature_keywords.items():
                if keyword in query.original_query:
                    app = get_gdt_for_feature(feature_type)
                    results.append(RetrievalResult(
                        source=RetrievalSource.GDT,
                        relevance=0.9,
                        data={
                            "feature_type": feature_type.value,
                            "recommended_characteristics": [c.value for c in app.recommended_characteristics],
                            "typical_tolerances": {
                                c.value: t for c, t in app.typical_tolerances.items()
                            },
                            "inspection_methods": [m.value for m in app.inspection_methods],
                        },
                        summary=f"{keyword}特征GD&T: {', '.join(c.value for c in app.recommended_characteristics[:3])}",
                        metadata={"id": f"gdt_app_{feature_type.value}"},
                    ))
                    break

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
