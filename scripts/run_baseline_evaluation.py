#!/usr/bin/env python3
"""
运行基线评测脚本
评测装配理解AI的各项指标
"""

import json
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.assembly.assembly_graph_builder import AssemblyGraphBuilder
from src.assembly.rules.assembly_rules import AssemblyRuleEngine
from src.assembly.evidence_collector import EvidenceCollector
from src.assembly.graph_normalizer import AssemblyGraphNormalizer
from src.evaluation.metrics import AssemblyMetrics


class BaselineEvaluator:
    """基线评测器"""

    def __init__(self):
        self.graph_builder = AssemblyGraphBuilder()
        self.rule_engine = AssemblyRuleEngine()
        self.evidence_collector = EvidenceCollector()
        self.normalizer = AssemblyGraphNormalizer()
        self.metrics = AssemblyMetrics()

    def run_baseline_tests(self):
        """运行基线测试"""

        # 黄金测试用例
        test_cases = self._load_golden_cases()

        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_count": len(test_cases),
            "individual_results": [],
            "aggregate_metrics": {}
        }

        print("=" * 50)
        print("🎯 装配理解AI基线评测")
        print("=" * 50)

        all_metrics = []

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📊 测试用例 {i}/{len(test_cases)}: {test_case['name']}")
            print("-" * 40)

            # 运行测试
            case_result = self._evaluate_single_case(test_case)

            # 记录结果
            results["individual_results"].append(case_result)
            all_metrics.append(case_result["metrics"])

            # 打印摘要
            self._print_case_summary(case_result)

        # 计算聚合指标
        results["aggregate_metrics"] = self._calculate_aggregate_metrics(all_metrics)

        # 打印总体报告
        self._print_overall_report(results)

        # 保存结果
        self._save_results(results)

        return results

    def _load_golden_cases(self):
        """加载黄金测试用例"""

        return [
            {
                "name": "simple_gear_train",
                "description": "简单齿轮系",
                "input": {
                    "parts": [
                        {"id": "gear1", "type": "gear", "label": "主动轮 Z=20",
                         "center_of_mass": [0, 0, 0], "volume": 100},
                        {"id": "gear2", "type": "gear", "label": "从动轮 Z=60",
                         "center_of_mass": [100, 0, 0], "volume": 300},
                        {"id": "shaft1", "type": "shaft", "label": "输入轴",
                         "center_of_mass": [0, 0, -50], "volume": 50},
                        {"id": "shaft2", "type": "shaft", "label": "输出轴",
                         "center_of_mass": [100, 0, -50], "volume": 60}
                    ],
                    "mates": [
                        {"id": "m1", "part1": "gear1", "part2": "gear2", "type": "gear_mesh"},
                        {"id": "m2", "part1": "gear1", "part2": "shaft1", "type": "fixed"},
                        {"id": "m3", "part1": "gear2", "part2": "shaft2", "type": "fixed"}
                    ]
                },
                "expected": {
                    "function": "齿轮传动",
                    "transmission_ratio": 3.0,
                    "degrees_of_freedom": 1,
                    "joint_types": ["gear_mesh", "fixed", "fixed"]
                }
            },
            {
                "name": "belt_drive_system",
                "description": "皮带传动系统",
                "input": {
                    "parts": [
                        {"id": "pulley1", "type": "pulley", "label": "主动轮 D=100",
                         "center_of_mass": [0, 0, 0], "volume": 80},
                        {"id": "pulley2", "type": "pulley", "label": "从动轮 D=200",
                         "center_of_mass": [300, 0, 0], "volume": 160},
                        {"id": "belt", "type": "belt", "label": "同步带",
                         "center_of_mass": [150, 0, 0], "volume": 20}
                    ],
                    "mates": [
                        {"id": "m1", "part1": "pulley1", "part2": "belt", "type": "belt_contact"},
                        {"id": "m2", "part1": "pulley2", "part2": "belt", "type": "belt_contact"}
                    ]
                },
                "expected": {
                    "function": "皮带传动",
                    "transmission_ratio": 2.0,
                    "joint_types": ["belt_contact", "belt_contact"]
                }
            },
            {
                "name": "bearing_supported_shaft",
                "description": "轴承支撑轴系",
                "input": {
                    "parts": [
                        {"id": "shaft", "type": "shaft", "label": "传动轴 D=30",
                         "center_of_mass": [0, 0, 0], "volume": 200},
                        {"id": "bearing1", "type": "bearing", "label": "轴承6206",
                         "center_of_mass": [-100, 0, 0], "volume": 30},
                        {"id": "bearing2", "type": "bearing", "label": "轴承6206",
                         "center_of_mass": [100, 0, 0], "volume": 30},
                        {"id": "housing", "type": "housing", "label": "轴承座",
                         "center_of_mass": [0, 0, -50], "volume": 500}
                    ],
                    "mates": [
                        {"id": "m1", "part1": "shaft", "part2": "bearing1", "type": "bearing_fit"},
                        {"id": "m2", "part1": "shaft", "part2": "bearing2", "type": "bearing_fit"},
                        {"id": "m3", "part1": "bearing1", "part2": "housing", "type": "fixed"},
                        {"id": "m4", "part1": "bearing2", "part2": "housing", "type": "fixed"}
                    ]
                },
                "expected": {
                    "function": "轴承支撑",
                    "degrees_of_freedom": 1,
                    "joint_types": ["bearing_fit", "bearing_fit", "fixed", "fixed"]
                }
            },
            {
                "name": "complex_gearbox",
                "description": "复杂齿轮箱",
                "input": {
                    "parts": [
                        {"id": "motor", "type": "motor", "label": "电机"},
                        {"id": "gear1", "type": "gear", "label": "齿轮1 Z=15"},
                        {"id": "gear2", "type": "gear", "label": "齿轮2 Z=45"},
                        {"id": "gear3", "type": "gear", "label": "齿轮3 Z=20"},
                        {"id": "gear4", "type": "gear", "label": "齿轮4 Z=80"},
                        {"id": "shaft1", "type": "shaft", "label": "输入轴"},
                        {"id": "shaft2", "type": "shaft", "label": "中间轴"},
                        {"id": "shaft3", "type": "shaft", "label": "输出轴"}
                    ],
                    "mates": [
                        {"id": "m1", "part1": "motor", "part2": "shaft1", "type": "coupling"},
                        {"id": "m2", "part1": "gear1", "part2": "shaft1", "type": "keyed"},
                        {"id": "m3", "part1": "gear2", "part2": "shaft2", "type": "keyed"},
                        {"id": "m4", "part1": "gear3", "part2": "shaft2", "type": "keyed"},
                        {"id": "m5", "part1": "gear4", "part2": "shaft3", "type": "keyed"},
                        {"id": "m6", "part1": "gear1", "part2": "gear2", "type": "gear_mesh"},
                        {"id": "m7", "part1": "gear3", "part2": "gear4", "type": "gear_mesh"}
                    ]
                },
                "expected": {
                    "function": "二级齿轮减速器",
                    "transmission_ratio": 12.0,  # (45/15) * (80/20)
                    "degrees_of_freedom": 1
                }
            },
            {
                "name": "linkage_mechanism",
                "description": "连杆机构",
                "input": {
                    "parts": [
                        {"id": "crank", "type": "link", "label": "曲柄"},
                        {"id": "coupler", "type": "link", "label": "连杆"},
                        {"id": "rocker", "type": "link", "label": "摇杆"},
                        {"id": "frame", "type": "frame", "label": "机架"}
                    ],
                    "mates": [
                        {"id": "m1", "part1": "crank", "part2": "frame", "type": "revolute"},
                        {"id": "m2", "part1": "crank", "part2": "coupler", "type": "revolute"},
                        {"id": "m3", "part1": "coupler", "part2": "rocker", "type": "revolute"},
                        {"id": "m4", "part1": "rocker", "part2": "frame", "type": "revolute"}
                    ]
                },
                "expected": {
                    "function": "四杆机构",
                    "degrees_of_freedom": 1,
                    "joint_types": ["revolute", "revolute", "revolute", "revolute"]
                }
            },
            {
                "name": "cam_follower",
                "description": "凸轮机构",
                "input": {
                    "parts": [
                        {"id": "cam", "type": "cam", "label": "凸轮"},
                        {"id": "follower", "type": "follower", "label": "从动件"},
                        {"id": "spring", "type": "spring", "label": "复位弹簧"},
                        {"id": "guide", "type": "guide", "label": "导轨"}
                    ],
                    "mates": [
                        {"id": "m1", "part1": "cam", "part2": "follower", "type": "cam_contact"},
                        {"id": "m2", "part1": "follower", "part2": "guide", "type": "prismatic"},
                        {"id": "m3", "part1": "spring", "part2": "follower", "type": "spring_force"}
                    ]
                },
                "expected": {
                    "function": "凸轮机构",
                    "degrees_of_freedom": 1,
                    "joint_types": ["cam_contact", "prismatic", "spring_force"]
                }
            }
        ]

    def _evaluate_single_case(self, test_case):
        """评测单个测试用例"""

        start_time = time.time()

        # 构建装配图
        assembly_graph = self.graph_builder.build_from_parsed_data(test_case["input"])

        # 收集证据
        evidence_chains = []
        for mate in test_case["input"]["mates"]:
            part1 = next(p for p in test_case["input"]["parts"] if p["id"] == mate["part1"])
            part2 = next(p for p in test_case["input"]["parts"] if p["id"] == mate["part2"])
            evidence = self.evidence_collector.collect_evidence(part1, part2, {})
            evidence_chains.append({
                "mate": mate["id"],
                "evidence": evidence
            })

        # 规范化处理
        normalized_graph = self.normalizer.normalize(assembly_graph)

        # 规则验证
        validation = self.rule_engine.validate_assembly(normalized_graph)

        processing_time = time.time() - start_time

        # 添加处理信息
        normalized_graph["processing_time"] = processing_time
        normalized_graph["evidence_chains"] = evidence_chains

        # 计算评测指标
        metrics = self.metrics.evaluate(normalized_graph, test_case["expected"])

        return {
            "name": test_case["name"],
            "description": test_case["description"],
            "predicted": normalized_graph,
            "expected": test_case["expected"],
            "metrics": metrics,
            "processing_time": processing_time,
            "validation": validation
        }

    def _print_case_summary(self, case_result):
        """打印用例摘要"""

        metrics = case_result["metrics"]
        graph_metrics = metrics.get("graph_quality", {})
        physics_metrics = metrics.get("physics_consistency", {})
        evidence_metrics = metrics.get("evidence_quality", {})

        print(f"  ✓ Edge F1: {graph_metrics.get('edge_f1', 0):.3f}")
        print(f"  ✓ Joint Type Accuracy: {graph_metrics.get('joint_type_accuracy', 0):.3f}")
        print(f"  ✓ Evidence Coverage: {evidence_metrics.get('evidence_coverage', 0):.3f}")
        print(f"  ✓ Processing Time: {case_result['processing_time']:.3f}s")
        print(f"  ✓ Overall Score: {metrics.get('overall_score', 0):.3f}")

        # 检查是否达标
        if metrics.get('overall_score', 0) >= 0.75:
            print("  ✅ PASSED")
        else:
            print("  ❌ FAILED")

    def _calculate_aggregate_metrics(self, all_metrics):
        """计算聚合指标"""

        aggregate = {
            "avg_edge_f1": 0,
            "avg_joint_accuracy": 0,
            "avg_evidence_coverage": 0,
            "avg_confidence": 0,
            "avg_overall_score": 0,
            "pass_rate": 0
        }

        if not all_metrics:
            return aggregate

        n = len(all_metrics)

        # 计算平均值
        for metrics in all_metrics:
            graph = metrics.get("graph_quality", {})
            evidence = metrics.get("evidence_quality", {})

            aggregate["avg_edge_f1"] += graph.get("edge_f1", 0) / n
            aggregate["avg_joint_accuracy"] += graph.get("joint_type_accuracy", 0) / n
            aggregate["avg_evidence_coverage"] += evidence.get("evidence_coverage", 0) / n
            aggregate["avg_confidence"] += evidence.get("average_confidence", 0) / n
            aggregate["avg_overall_score"] += metrics.get("overall_score", 0) / n

        # 计算通过率
        passed = sum(1 for m in all_metrics if m.get("overall_score", 0) >= 0.75)
        aggregate["pass_rate"] = passed / n

        return aggregate

    def _print_overall_report(self, results):
        """打印总体报告"""

        print("\n" + "=" * 50)
        print("📊 总体评测报告")
        print("=" * 50)

        agg = results["aggregate_metrics"]

        print(f"\n测试用例数: {results['test_count']}")
        print(f"通过率: {agg['pass_rate']:.1%}")
        print(f"\n关键指标:")
        print(f"  • 平均Edge F1: {agg['avg_edge_f1']:.3f}")
        print(f"  • 平均关节类型准确率: {agg['avg_joint_accuracy']:.3f}")
        print(f"  • 平均证据覆盖率: {agg['avg_evidence_coverage']:.3f}")
        print(f"  • 平均置信度: {agg['avg_confidence']:.3f}")
        print(f"  • 平均总分: {agg['avg_overall_score']:.3f}")

        # 判定是否达到基线
        baseline_met = (
            agg['avg_edge_f1'] >= 0.75 and
            agg['pass_rate'] >= 0.8
        )

        print("\n基线状态:")
        if baseline_met:
            print("  ✅ 达到基线要求 (Edge F1 ≥ 0.75)")
        else:
            print("  ❌ 未达到基线要求")

    def _save_results(self, results):
        """保存评测结果"""

        output_path = Path("evaluation_results") / f"baseline_{time.strftime('%Y%m%d_%H%M%S')}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n💾 结果已保存到: {output_path}")


if __name__ == "__main__":
    evaluator = BaselineEvaluator()
    results = evaluator.run_baseline_tests()

    # 退出码：0表示成功，1表示失败
    exit_code = 0 if results["aggregate_metrics"]["pass_rate"] >= 0.8 else 1
    sys.exit(exit_code)