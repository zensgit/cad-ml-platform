"""
Automated AI quality evaluation suite for the CAD-ML Platform.

Evaluates the system's ability to answer manufacturing questions correctly
using golden Q&A datasets, cost estimation sanity checks, and hybrid
intelligence consistency tests.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class AIEvaluationSuite:
    """Automated evaluation of AI answer quality.

    Tests the Copilot's ability to answer manufacturing questions correctly
    using a golden Q&A dataset, validates cost estimation logic, and checks
    hybrid classifier intelligence behaviour.
    """

    def __init__(self) -> None:
        self.test_cases = self._build_test_cases()

    # ------------------------------------------------------------------
    # Golden Q&A test cases
    # ------------------------------------------------------------------

    def _build_test_cases(self) -> List[Dict[str, Any]]:
        """Build golden Q&A test cases for evaluation."""
        return [
            # ---- Material queries (6) ------------------------------------
            {
                "question": "SUS304的密度是多少？",
                "expected_contains": ["7930", "kg"],
                "category": "material",
            },
            {
                "question": "铝合金6061的抗拉强度？",
                "expected_contains": ["310", "MPa"],
                "category": "material",
            },
            {
                "question": "Q235和SUS304哪个更贵？",
                "expected_contains": ["SUS304", "贵"],
                "category": "material",
            },
            {
                "question": "TC4钛合金的价格？",
                "expected_contains": ["180"],
                "category": "material",
            },
            {
                "question": "45号钢的抗拉强度？",
                "expected_contains": ["600"],
                "category": "material",
            },
            {
                "question": "黄铜H62的密度？",
                "expected_contains": ["8430"],
                "category": "material",
            },
            # ---- Process queries (5) -------------------------------------
            {
                "question": "法兰盘一般用什么工艺加工？",
                "expected_contains": ["车削", "CNC"],
                "category": "process",
            },
            {
                "question": "钛合金适合线切割吗？",
                "expected_contains": ["适合"],
                "category": "process",
            },
            {
                "question": "Ra1.6需要什么加工？",
                "expected_contains": ["磨削"],
                "category": "process",
            },
            {
                "question": "壳体通常用什么加工？",
                "expected_contains": ["铣削"],
                "category": "process",
            },
            {
                "question": "齿轮精加工用什么工艺？",
                "expected_contains": ["磨削"],
                "category": "process",
            },
            # ---- Cost queries (2) ----------------------------------------
            {
                "question": "钢件10cm3大概多少钱？",
                "expected_contains": ["CNY"],
                "category": "cost",
            },
            {
                "question": "批量生产能降低成本吗？",
                "expected_contains": ["批量"],
                "category": "cost",
            },
            # ---- GD&T queries (3) ----------------------------------------
            {
                "question": "什么是平面度？",
                "expected_contains": ["flatness", "平面"],
                "category": "gdt",
            },
            {
                "question": "圆柱度和圆度有什么区别？",
                "expected_contains": ["cylindricity", "circularity"],
                "category": "gdt",
            },
            {
                "question": "位置度公差怎么标注？",
                "expected_contains": ["位置"],
                "category": "gdt",
            },
            # ---- Tolerance queries (2) -----------------------------------
            {
                "question": "IT7的公差值大约是多少？",
                "expected_contains": ["IT7"],
                "category": "tolerance",
            },
            {
                "question": "IT6和IT8哪个更精密？",
                "expected_contains": ["IT6"],
                "category": "tolerance",
            },
            # ---- Welding queries (3) -------------------------------------
            {
                "question": "SUS304焊接用什么焊丝？",
                "expected_contains": ["焊"],
                "category": "welding",
            },
            {
                "question": "铝合金能焊接吗？",
                "expected_contains": ["铝"],
                "category": "welding",
            },
            {
                "question": "碳钢焊接需要预热吗？",
                "expected_contains": ["碳钢"],
                "category": "welding",
            },
            # ---- Knowledge graph queries (15) ----------------------------
            {
                "question": "SUS304适合什么工艺？",
                "expected_contains": ["CNC", "车削"],
                "category": "knowledge_graph",
            },
            {
                "question": "法兰盘常用什么材料？",
                "expected_contains": ["钢", "碳钢"],
                "category": "knowledge_graph",
            },
            {
                "question": "CNC车削能达到什么精度？",
                "expected_contains": ["IT"],
                "category": "knowledge_graph",
            },
            {
                "question": "齿轮通常用什么材料？",
                "expected_contains": ["钢"],
                "category": "knowledge_graph",
            },
            {
                "question": "轴用什么材料做？",
                "expected_contains": ["45", "钢"],
                "category": "knowledge_graph",
            },
            {
                "question": "壳体常用什么材料？",
                "expected_contains": ["铝"],
                "category": "knowledge_graph",
            },
            {
                "question": "支架一般用什么材料？",
                "expected_contains": ["钢"],
                "category": "knowledge_graph",
            },
            {
                "question": "SUS304能做什么零件？",
                "expected_contains": ["法兰", "轴"],
                "category": "knowledge_graph",
            },
            {
                "question": "Q235适合什么工艺？",
                "expected_contains": ["CNC"],
                "category": "knowledge_graph",
            },
            {
                "question": "铝合金6061适合什么加工？",
                "expected_contains": ["CNC"],
                "category": "knowledge_graph",
            },
            {
                "question": "磨削能达到什么粗糙度？",
                "expected_contains": ["Ra"],
                "category": "knowledge_graph",
            },
            {
                "question": "5轴加工能达到什么精度？",
                "expected_contains": ["IT"],
                "category": "knowledge_graph",
            },
            {
                "question": "法兰盘用SUS304做，推荐什么工艺？",
                "expected_contains": ["车削"],
                "category": "knowledge_graph",
            },
            {
                "question": "轴用45号钢做，推荐什么工艺？",
                "expected_contains": ["车削"],
                "category": "knowledge_graph",
            },
            {
                "question": "连接件常用什么材料？",
                "expected_contains": ["钢"],
                "category": "knowledge_graph",
            },
            # ---- Cross-domain queries (5) --------------------------------
            {
                "question": "钛合金适合什么工艺？",
                "expected_contains": ["5轴", "线切割"],
                "category": "knowledge_graph",
            },
            {
                "question": "Ra0.8需要什么加工方式？",
                "expected_contains": ["磨削"],
                "category": "knowledge_graph",
            },
            {
                "question": "IT6需要什么加工方式？",
                "expected_contains": ["磨削"],
                "category": "knowledge_graph",
            },
        ]

    # ------------------------------------------------------------------
    # Knowledge graph evaluation
    # ------------------------------------------------------------------

    async def evaluate_knowledge_graph(self) -> Dict[str, Any]:
        """Evaluate knowledge graph query quality."""
        from src.ml.knowledge import ManufacturingKnowledgeGraph, GraphQueryEngine

        graph = ManufacturingKnowledgeGraph()
        graph.build_default_graph()
        engine = GraphQueryEngine(graph)

        results: List[Dict[str, Any]] = []
        for tc in self.test_cases:
            if tc["category"] != "knowledge_graph":
                continue
            result = engine.query(tc["question"])
            answer = result.answer
            hits = sum(1 for kw in tc["expected_contains"] if kw in answer)
            score = hits / len(tc["expected_contains"])
            results.append(
                {
                    "question": tc["question"],
                    "answer": answer[:200],
                    "score": score,
                    "confidence": result.confidence,
                }
            )

        avg_score = sum(r["score"] for r in results) / len(results) if results else 0
        return {
            "category": "knowledge_graph",
            "cases": len(results),
            "avg_score": avg_score,
            "passed": sum(1 for r in results if r["score"] > 0.5),
            "details": results,
        }

    # ------------------------------------------------------------------
    # Cost estimation evaluation
    # ------------------------------------------------------------------

    async def evaluate_cost_estimation(self) -> Dict[str, Any]:
        """Evaluate cost estimation consistency."""
        from src.ml.cost.estimator import CostEstimator
        from src.ml.cost.models import CostEstimateRequest

        est = CostEstimator()

        checks: List[Dict[str, Any]] = []

        # Check 1: titanium > steel
        r1 = est.estimate(
            CostEstimateRequest(
                material="steel",
                bounding_volume_mm3=10000,
                entity_count=20,
                batch_size=1,
            )
        )
        r2 = est.estimate(
            CostEstimateRequest(
                material="titanium",
                bounding_volume_mm3=10000,
                entity_count=20,
                batch_size=1,
            )
        )
        checks.append(
            {"test": "titanium > steel", "pass": r2.estimate.total > r1.estimate.total}
        )

        # Check 2: batch reduces cost per unit
        r3 = est.estimate(
            CostEstimateRequest(
                material="steel",
                bounding_volume_mm3=10000,
                entity_count=20,
                batch_size=100,
            )
        )
        checks.append(
            {"test": "batch reduces unit cost", "pass": r3.estimate.total < r1.estimate.total}
        )

        # Check 3: optimistic < estimate < pessimistic
        checks.append(
            {
                "test": "cost range order",
                "pass": r1.optimistic.total < r1.estimate.total < r1.pessimistic.total,
            }
        )

        # Check 4: confidence > 0
        checks.append({"test": "confidence positive", "pass": r1.confidence > 0})

        # Check 5: higher entity count yields higher complexity
        r4 = est.estimate(
            CostEstimateRequest(
                material="steel",
                bounding_volume_mm3=10000,
                entity_count=200,
                batch_size=1,
            )
        )
        checks.append(
            {
                "test": "more entities -> higher complexity",
                "pass": r4.complexity_score > r1.complexity_score,
            }
        )

        # Check 6: all costs are positive
        checks.append(
            {
                "test": "all cost components positive",
                "pass": all(
                    v > 0
                    for v in [
                        r1.estimate.material_cost,
                        r1.estimate.machining_cost,
                        r1.estimate.setup_cost,
                        r1.estimate.overhead,
                    ]
                ),
            }
        )

        passed = sum(1 for c in checks if c["pass"])
        return {
            "category": "cost",
            "cases": len(checks),
            "passed": passed,
            "avg_score": passed / len(checks),
            "details": checks,
        }

    # ------------------------------------------------------------------
    # Hybrid intelligence evaluation
    # ------------------------------------------------------------------

    async def evaluate_hybrid_intelligence(self) -> Dict[str, Any]:
        """Evaluate hybrid classifier intelligence."""
        from src.ml.hybrid.intelligence import HybridIntelligence

        hi = HybridIntelligence()

        checks: List[Dict[str, Any]] = []

        # Unanimous agreement -> low uncertainty
        unanimous = {
            "filename": {"label": "法兰盘", "confidence": 0.9},
            "graph2d": {"label": "法兰盘", "confidence": 0.8},
            "titleblock": {"label": "法兰盘", "confidence": 0.85},
        }
        u = hi.analyze_ensemble_uncertainty(unanimous)
        checks.append(
            {"test": "unanimous -> low uncertainty", "pass": u.severity == "low"}
        )

        # Disagreement -> detected
        mixed = {
            "filename": {"label": "法兰盘", "confidence": 0.9},
            "graph2d": {"label": "壳体", "confidence": 0.8},
            "titleblock": {"label": "轴", "confidence": 0.7},
        }
        d = hi.detect_disagreement(mixed)
        checks.append(
            {"test": "mixed -> disagreement detected", "pass": d.has_disagreement}
        )

        # Calibration reduces on disagreement
        cal_high = hi.compute_calibrated_confidence(0.9, unanimous)
        cal_low = hi.compute_calibrated_confidence(0.9, mixed)
        checks.append(
            {
                "test": "disagreement lowers confidence",
                "pass": cal_low.calibrated_confidence < cal_high.calibrated_confidence,
            }
        )

        # Cross-validation detects inconsistency
        prediction = {"label": "法兰盘", "confidence": 0.9}
        cv = hi.cross_validate_prediction(prediction, mixed)
        checks.append(
            {
                "test": "cross-validation detects inconsistency",
                "pass": not cv.is_consistent or len(cv.warnings) > 0,
            }
        )

        # Explanation is non-empty
        explanation = hi.generate_smart_explanation(prediction, unanimous, u)
        checks.append(
            {
                "test": "smart explanation non-empty",
                "pass": len(explanation) > 10,
            }
        )

        passed = sum(1 for c in checks if c["pass"])
        return {
            "category": "intelligence",
            "cases": len(checks),
            "passed": passed,
            "avg_score": passed / len(checks),
            "details": checks,
        }

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    async def run_full_evaluation(self) -> Dict[str, Any]:
        """Run all evaluation suites and generate report."""
        kg_result = await self.evaluate_knowledge_graph()
        cost_result = await self.evaluate_cost_estimation()
        intel_result = await self.evaluate_hybrid_intelligence()

        results = [kg_result, cost_result, intel_result]
        overall_score = sum(r["avg_score"] for r in results) / len(results)

        return {
            "overall_score": round(overall_score, 4),
            "categories": results,
            "total_cases": sum(r["cases"] for r in results),
            "verdict": "PASS" if overall_score >= 0.6 else "NEEDS_IMPROVEMENT",
        }

    # ------------------------------------------------------------------
    # Markdown report generation
    # ------------------------------------------------------------------

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate Markdown evaluation report."""
        lines = ["# AI Quality Evaluation Report\n"]
        lines.append(
            f"**Overall Score: {results['overall_score']:.1%}** "
            f"({results['verdict']})\n"
        )
        lines.append(f"Total test cases: {results['total_cases']}\n")

        for cat in results["categories"]:
            lines.append(f"\n## {cat['category'].replace('_', ' ').title()}")
            lines.append(
                f"Score: {cat['avg_score']:.1%} "
                f"({cat.get('passed', cat['cases'])}/{cat['cases']})\n"
            )
            for d in cat.get("details", []):
                if "pass" in d:
                    status = "PASS" if d["pass"] else "FAIL"
                    test_name = d.get("test", "")[:60]
                else:
                    status = "PASS" if d.get("score", 0) > 0.5 else "FAIL"
                    test_name = d.get("question", "")[:60]
                lines.append(f"- [{status}] {test_name}")

        return "\n".join(lines)
