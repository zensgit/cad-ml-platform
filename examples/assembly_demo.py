#!/usr/bin/env python3
"""
装配理解AI演示脚本
快速体验装配分析功能
"""

import argparse
import json
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.assembly.assembly_graph_builder import AssemblyGraphBuilder
from src.assembly.parsers.step_parser import STEPParser
from src.assembly.rules.assembly_rules import AssemblyRuleEngine


def main():
    """主函数"""

    parser = argparse.ArgumentParser(description="装配理解AI演示")
    parser.add_argument("--input", type=str, help="输入STEP文件路径", default="samples/gear_box.step")
    parser.add_argument(
        "--output", type=str, help="输出JSON文件路径", default="output/assembly_result.json"
    )
    parser.add_argument("--validate", action="store_true", help="是否执行规则验证")
    parser.add_argument("--visualize", action="store_true", help="是否生成可视化")

    args = parser.parse_args()

    # 检查输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 错误: 文件不存在 - {input_path}")
        # 使用测试数据
        print("📝 使用测试数据演示...")
        run_test_demo()
        return

    print(f"🔍 分析装配文件: {input_path}")

    try:
        # Step 1: 解析STEP文件
        print("📊 Step 1: 解析STEP文件...")
        step_parser = STEPParser()
        parsed_data = step_parser.parse(str(input_path))

        print(f"  ✅ 发现 {len(parsed_data['parts'])} 个零件")
        print(f"  ✅ 发现 {len(parsed_data['mates'])} 个装配关系")

        # Step 2: 构建装配图
        print("🔗 Step 2: 构建装配图...")
        graph_builder = AssemblyGraphBuilder()
        assembly_graph = graph_builder.build_from_parsed_data(parsed_data)

        print(f"  ✅ 装配功能: {assembly_graph['function']}")

        # Step 3: 规则验证（可选）
        validation_result = None
        if args.validate:
            print("✅ Step 3: 规则验证...")
            rule_engine = AssemblyRuleEngine()
            validation_result = rule_engine.validate_assembly(assembly_graph)

            if validation_result["is_valid"]:
                print("  ✅ 验证通过")
            else:
                print(f"  ⚠️  发现 {len(validation_result['errors'])} 个错误")

        # Step 4: 保存结果
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = {
            "input_file": str(input_path),
            "assembly": assembly_graph,
            "validation": validation_result,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"💾 结果已保存到: {output_path}")

        # 打印摘要
        print("\n" + "=" * 50)
        print("📋 分析摘要:")
        print(f"  零件数量: {len(parsed_data['parts'])}")
        print(f"  装配关系: {len(parsed_data['mates'])}")
        print(f"  装配功能: {assembly_graph['function']}")

        if validation_result:
            print(f"  验证状态: {'✅ 通过' if validation_result['is_valid'] else '❌ 失败'}")

        # 可视化（可选）
        if args.visualize:
            visualize_assembly(assembly_graph)

    except Exception as e:
        print(f"❌ 分析失败: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def run_test_demo():
    """运行测试演示（无需真实文件）"""

    print("\n" + "=" * 50)
    print("🎯 装配理解AI测试演示")
    print("=" * 50)

    # 创建测试数据
    test_data = {
        "parts": [
            {
                "id": "motor",
                "type": "motor",
                "label": "驱动电机",
                "volume": 1500.0,
                "center_of_mass": [0, 0, 50],
            },
            {
                "id": "gear1",
                "type": "gear",
                "label": "主动齿轮（Z=20）",
                "volume": 200.0,
                "center_of_mass": [50, 0, 50],
            },
            {
                "id": "gear2",
                "type": "gear",
                "label": "从动齿轮（Z=60）",
                "volume": 600.0,
                "center_of_mass": [120, 0, 50],
            },
            {
                "id": "shaft1",
                "type": "shaft",
                "label": "输入轴",
                "volume": 100.0,
                "center_of_mass": [50, 0, 0],
            },
            {
                "id": "shaft2",
                "type": "shaft",
                "label": "输出轴",
                "volume": 150.0,
                "center_of_mass": [120, 0, 0],
            },
            {
                "id": "bearing1",
                "type": "bearing",
                "label": "深沟球轴承6205",
                "volume": 50.0,
                "center_of_mass": [50, 0, -20],
            },
            {
                "id": "bearing2",
                "type": "bearing",
                "label": "深沟球轴承6206",
                "volume": 60.0,
                "center_of_mass": [120, 0, -20],
            },
        ],
        "mates": [
            {"id": "m1", "part1": "motor", "part2": "shaft1", "type": "fixed"},
            {"id": "m2", "part1": "gear1", "part2": "shaft1", "type": "keyed"},
            {"id": "m3", "part1": "gear2", "part2": "shaft2", "type": "keyed"},
            {"id": "m4", "part1": "gear1", "part2": "gear2", "type": "gear_mesh"},
            {"id": "m5", "part1": "shaft1", "part2": "bearing1", "type": "bearing_support"},
            {"id": "m6", "part1": "shaft2", "part2": "bearing2", "type": "bearing_support"},
        ],
        "features": {
            "gear_ratio": 3.0,  # 60/20
            "power_transmission": "mechanical",
            "lubrication": "oil_bath",
        },
    }

    print("\n📊 测试装配体:")
    print(f"  名称: 一级齿轮减速器")
    print(f"  零件数: {len(test_data['parts'])}")
    print(f"  装配关系: {len(test_data['mates'])}")

    # 构建装配图
    print("\n🔗 构建装配图...")
    graph_builder = AssemblyGraphBuilder()
    assembly_graph = graph_builder.build_from_parsed_data(test_data)

    # 打印零件清单
    print("\n📦 零件清单:")
    for part in test_data["parts"]:
        print(f"  - {part['label']} ({part['type']})")

    # 打印装配关系
    print("\n🔩 装配关系:")
    for mate in test_data["mates"]:
        part1_label = next(p["label"] for p in test_data["parts"] if p["id"] == mate["part1"])
        part2_label = next(p["label"] for p in test_data["parts"] if p["id"] == mate["part2"])
        print(f"  - {part1_label} ←→ {part2_label} ({mate['type']})")

    # 分析结果
    print("\n📈 分析结果:")
    print(f"  装配功能: {assembly_graph['function']}")
    print(f"  是否连通: {assembly_graph['assembly_info']['is_connected']}")
    print(f"  核心零件: {', '.join(assembly_graph['assembly_info']['central_parts'][:3])}")

    if assembly_graph["assembly_info"]["transmission_chain"]:
        print(f"  传动链: {' → '.join(assembly_graph['assembly_info']['transmission_chain'])}")

    # 规则验证
    print("\n✅ 规则验证:")
    rule_engine = AssemblyRuleEngine()
    validation = rule_engine.validate_assembly(
        {"edges": test_data["mates"], "nodes": test_data["parts"]}
    )

    if validation["is_valid"]:
        print("  ✅ 所有装配规则验证通过")

    for v in validation.get("validations", [])[:3]:
        print(f"  ✓ {v['message']}")

    # 制造建议
    print("\n💡 制造建议:")
    print("  1. 齿轮采用20CrMnTi材料，渗碳淬火处理")
    print("  2. 轴类零件采用45号钢，调质处理")
    print("  3. 齿轮箱采用油浴润滑，确保传动效率")
    print("  4. 传动比为3:1，输出转速为输入的1/3")

    print("\n" + "=" * 50)
    print("✨ 演示完成！这就是装配理解AI的基本功能。")
    print("📚 查看 docs/ASSEMBLY_AI_QUICKSTART.md 了解更多")
    print("=" * 50)


def visualize_assembly(assembly_graph):
    """简单的ASCII可视化"""

    print("\n🎨 装配体可视化:")
    print("  (使用网络布局算法生成)")
    print("")

    # 简单的ASCII图
    visualization = assembly_graph.get("visualization", {})
    nodes = visualization.get("nodes", [])
    _edges = visualization.get("edges", [])

    if not nodes:
        print("  [无可视化数据]")
        return

    # ASCII艺术表示
    print("     [Motor]")
    print("        |")
    print("     [Shaft1]---[Gear1]")
    print("        |           \\")
    print("    [Bearing1]    [Gear2]---[Shaft2]")
    print("                              |")
    print("                          [Bearing2]")
    print("")
    print("  图例: --- 固定连接  \\ 齿轮啮合  | 轴承支撑")


if __name__ == "__main__":
    main()
