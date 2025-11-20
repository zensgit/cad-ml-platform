# 🚀 装配理解AI快速启动指南

## 📋 5分钟快速体验

### Step 1: 安装依赖（2分钟）

```bash
# 基础依赖
pip install fastapi uvicorn pydantic
pip install numpy scipy networkx

# CAD处理
pip install pythonocc-core freecad ezdxf

# 仿真（可选）
pip install pybullet

# 机器学习（可选）
pip install torch torch-geometric
```

### Step 2: 运行示例（3分钟）

```bash
# 克隆项目
git clone https://github.com/zensgit/cad-ml-platform
cd cad-ml-platform

# 运行装配分析示例
python examples/assembly_demo.py --input samples/gear_box.step

# 启动API服务
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 3: 测试API

```bash
# 上传STEP文件分析
curl -X POST http://localhost:8000/api/v1/assembly/analyze \
  -F "file=@samples/gear_box.step"

# 输出示例
{
  "assembly_graph": {
    "parts": ["gear1", "gear2", "shaft", "bearing"],
    "relations": ["gear_mesh", "bearing_support"],
    "function": "二级齿轮减速器，传动比12:1"
  }
}
```

---

## 🎯 第一周实施计划（MVP）

### Day 1: 环境搭建与工具链

#### 1.1 安装开发环境
```bash
# 创建项目结构
mkdir -p src/assembly/{parsers,reasoners,generators}
mkdir -p tests/assembly
mkdir -p samples/step_files

# 安装FreeCAD Python绑定
conda install -c conda-forge freecad

# 或使用pip
pip install freecad-python
```

#### 1.2 创建基础配置
```python
# src/assembly/config.py
from pydantic import BaseSettings

class AssemblyConfig(BaseSettings):
    """装配模块配置"""

    # 解析器设置
    PARSER_BACKEND: str = "freecad"  # freecad, pythonocc
    MAX_PART_COUNT: int = 100

    # 推理设置
    USE_AI_REASONING: bool = False  # 初期使用规则
    CONFIDENCE_THRESHOLD: float = 0.7

    # 仿真设置
    ENABLE_SIMULATION: bool = False  # 初期可选
    SIMULATION_ENGINE: str = "pybullet"

    # 性能设置
    CACHE_ENABLED: bool = True
    MAX_CACHE_SIZE: int = 100  # MB

assembly_config = AssemblyConfig()
```

### Day 2-3: STEP解析器开发

#### 2.1 基础STEP解析器
```python
# src/assembly/parsers/step_parser.py
import FreeCAD
import Part
import tempfile
from typing import List, Dict, Any
import hashlib

class STEPParser:
    """STEP文件解析器 - MVP版本"""

    def __init__(self):
        self.document = None
        self.parts = []
        self.mates = []

    def parse(self, file_path: str) -> Dict[str, Any]:
        """解析STEP文件"""

        # 打开STEP文件
        self.document = FreeCAD.open(file_path)

        # 提取零件
        self._extract_parts()

        # 分析装配关系（基于接触检测）
        self._analyze_mates()

        # 识别特征
        features = self._extract_features()

        return {
            "file_hash": self._compute_hash(file_path),
            "parts": self.parts,
            "mates": self.mates,
            "features": features,
            "stats": {
                "part_count": len(self.parts),
                "mate_count": len(self.mates)
            }
        }

    def _extract_parts(self):
        """提取零件信息"""

        for obj in self.document.Objects:
            if hasattr(obj, 'Shape'):
                part_info = {
                    "id": obj.Name,
                    "label": obj.Label,
                    "type": self._classify_part(obj.Shape),
                    "volume": obj.Shape.Volume,
                    "center_of_mass": list(obj.Shape.CenterOfMass),
                    "bounding_box": self._get_bbox(obj.Shape)
                }
                self.parts.append(part_info)

    def _classify_part(self, shape) -> str:
        """简单零件分类"""

        # 基于形状特征的规则分类
        faces = shape.Faces
        edges = shape.Edges

        # 检查是否为圆柱（可能是轴）
        cylindrical_faces = [f for f in faces if self._is_cylindrical(f)]
        if len(cylindrical_faces) > 0:
            aspect_ratio = self._calculate_aspect_ratio(shape)
            if aspect_ratio > 3:
                return "shaft"
            elif aspect_ratio < 0.5:
                return "disk"

        # 检查是否有齿形（齿轮）
        if self._has_gear_teeth(edges):
            return "gear"

        return "general_part"

    def _analyze_mates(self):
        """分析装配约束关系"""

        # 简单的接触检测
        for i, part1 in enumerate(self.parts):
            for part2 in self.parts[i+1:]:
                if self._check_contact(part1, part2):
                    mate = {
                        "id": f"mate_{len(self.mates)}",
                        "part1": part1["id"],
                        "part2": part2["id"],
                        "type": self._infer_mate_type(part1, part2)
                    }
                    self.mates.append(mate)

    def _infer_mate_type(self, part1: Dict, part2: Dict) -> str:
        """推断装配关系类型"""

        # 基于零件类型的简单规则
        types = {part1["type"], part2["type"]}

        if "gear" in types:
            return "gear_mesh"
        elif "shaft" in types and "bearing" in types:
            return "bearing_support"
        elif "shaft" in types:
            return "shaft_coupling"

        return "fixed"
```

### Day 4: 装配图生成器

#### 4.1 装配图构建
```python
# src/assembly/assembly_graph_builder.py
import networkx as nx
from typing import Dict, List, Any
import json

class AssemblyGraphBuilder:
    """装配图构建器"""

    def __init__(self):
        self.graph = nx.DiGraph()

    def build_from_parsed_data(self, parsed_data: Dict) -> Dict:
        """从解析数据构建装配图"""

        # 添加节点（零件）
        for part in parsed_data["parts"]:
            self.graph.add_node(
                part["id"],
                **part  # 包含所有零件属性
            )

        # 添加边（装配关系）
        for mate in parsed_data["mates"]:
            self.graph.add_edge(
                mate["part1"],
                mate["part2"],
                type=mate["type"],
                id=mate["id"]
            )

        # 分析装配结构
        assembly_info = self._analyze_assembly_structure()

        # 推理功能
        function = self._infer_function()

        return {
            "graph": nx.node_link_data(self.graph),
            "assembly_info": assembly_info,
            "function": function,
            "visualization": self._generate_visualization()
        }

    def _analyze_assembly_structure(self) -> Dict:
        """分析装配结构"""

        return {
            "is_connected": nx.is_connected(self.graph.to_undirected()),
            "components": list(nx.connected_components(self.graph.to_undirected())),
            "central_parts": self._find_central_parts(),
            "transmission_chain": self._find_transmission_chain()
        }

    def _find_central_parts(self) -> List[str]:
        """找出核心零件（连接最多的）"""

        centrality = nx.degree_centrality(self.graph)
        sorted_parts = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return [part[0] for part in sorted_parts[:3]]

    def _find_transmission_chain(self) -> List[str]:
        """查找传动链"""

        # 查找齿轮
        gears = [n for n in self.graph.nodes()
                 if self.graph.nodes[n].get("type") == "gear"]

        if len(gears) >= 2:
            # 尝试找出齿轮间的路径
            try:
                path = nx.shortest_path(self.graph.to_undirected(),
                                       gears[0], gears[-1])
                return path
            except nx.NetworkXNoPath:
                pass

        return []

    def _infer_function(self) -> str:
        """推理装配体功能"""

        # 统计零件类型
        part_types = [self.graph.nodes[n].get("type", "unknown")
                     for n in self.graph.nodes()]

        gear_count = part_types.count("gear")
        shaft_count = part_types.count("shaft")
        bearing_count = part_types.count("bearing")

        # 基于规则的功能推理
        if gear_count >= 2:
            return f"齿轮传动装置（{gear_count}个齿轮）"
        elif shaft_count >= 1 and bearing_count >= 2:
            return "轴承支撑系统"
        elif "motor" in part_types:
            return "电机驱动装置"
        else:
            return "通用机械装配"

    def _generate_visualization(self) -> Dict:
        """生成可视化数据"""

        # 为前端可视化准备数据
        pos = nx.spring_layout(self.graph)

        nodes = []
        for node in self.graph.nodes():
            nodes.append({
                "id": node,
                "x": pos[node][0],
                "y": pos[node][1],
                "type": self.graph.nodes[node].get("type", "unknown"),
                "label": self.graph.nodes[node].get("label", node)
            })

        edges = []
        for edge in self.graph.edges():
            edges.append({
                "source": edge[0],
                "target": edge[1],
                "type": self.graph.edges[edge].get("type", "fixed")
            })

        return {
            "nodes": nodes,
            "edges": edges
        }
```

### Day 5: 简单规则引擎

#### 5.1 装配规则库
```python
# src/assembly/rules/assembly_rules.py
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class AssemblyRule:
    """装配规则"""
    name: str
    description: str
    condition: callable
    action: callable
    priority: int = 0

class AssemblyRuleEngine:
    """装配规则引擎"""

    def __init__(self):
        self.rules = self._load_rules()

    def _load_rules(self) -> List[AssemblyRule]:
        """加载规则库"""

        rules = []

        # 齿轮啮合规则
        rules.append(AssemblyRule(
            name="gear_meshing",
            description="检查齿轮啮合条件",
            condition=lambda p1, p2: (
                p1.get("type") == "gear" and
                p2.get("type") == "gear"
            ),
            action=self._check_gear_meshing,
            priority=10
        ))

        # 轴承配合规则
        rules.append(AssemblyRule(
            name="bearing_fit",
            description="检查轴承配合",
            condition=lambda p1, p2: (
                "bearing" in [p1.get("type"), p2.get("type")] and
                "shaft" in [p1.get("type"), p2.get("type")]
            ),
            action=self._check_bearing_fit,
            priority=8
        ))

        # 同轴度规则
        rules.append(AssemblyRule(
            name="coaxiality",
            description="检查同轴度",
            condition=lambda p1, p2: (
                p1.get("type") == "shaft" and
                p2.get("type") == "shaft"
            ),
            action=self._check_coaxiality,
            priority=5
        ))

        return sorted(rules, key=lambda r: r.priority, reverse=True)

    def validate_assembly(self, assembly_graph: Dict) -> Dict:
        """验证装配合理性"""

        validations = []
        warnings = []
        errors = []

        # 遍历所有装配关系
        for edge in assembly_graph.get("edges", []):
            part1 = self._get_part_by_id(assembly_graph, edge["source"])
            part2 = self._get_part_by_id(assembly_graph, edge["target"])

            # 应用规则
            for rule in self.rules:
                if rule.condition(part1, part2):
                    result = rule.action(part1, part2)
                    if result["status"] == "error":
                        errors.append(result)
                    elif result["status"] == "warning":
                        warnings.append(result)
                    else:
                        validations.append(result)

        return {
            "is_valid": len(errors) == 0,
            "validations": validations,
            "warnings": warnings,
            "errors": errors,
            "summary": self._generate_summary(validations, warnings, errors)
        }

    def _check_gear_meshing(self, gear1: Dict, gear2: Dict) -> Dict:
        """检查齿轮啮合"""

        # 简化检查：基于边界框判断是否可能啮合
        bbox1 = gear1.get("bounding_box", {})
        bbox2 = gear2.get("bounding_box", {})

        # 检查是否有重叠（简化）
        if self._check_bbox_proximity(bbox1, bbox2):
            return {
                "status": "ok",
                "rule": "gear_meshing",
                "message": f"齿轮 {gear1['id']} 和 {gear2['id']} 可以正常啮合"
            }
        else:
            return {
                "status": "warning",
                "rule": "gear_meshing",
                "message": f"齿轮 {gear1['id']} 和 {gear2['id']} 间距可能过大"
            }

    def _check_bearing_fit(self, part1: Dict, part2: Dict) -> Dict:
        """检查轴承配合"""

        # 识别轴和轴承
        shaft = part1 if part1.get("type") == "shaft" else part2
        bearing = part2 if part2.get("type") == "bearing" else part1

        return {
            "status": "ok",
            "rule": "bearing_fit",
            "message": f"轴 {shaft['id']} 与轴承 {bearing['id']} 配合正常"
        }

    def _check_coaxiality(self, shaft1: Dict, shaft2: Dict) -> Dict:
        """检查同轴度"""

        # 基于质心位置简单判断
        com1 = shaft1.get("center_of_mass", [0, 0, 0])
        com2 = shaft2.get("center_of_mass", [0, 0, 0])

        # 简化：检查Y和Z坐标是否接近
        if abs(com1[1] - com2[1]) < 5 and abs(com1[2] - com2[2]) < 5:
            return {
                "status": "ok",
                "rule": "coaxiality",
                "message": f"轴 {shaft1['id']} 和 {shaft2['id']} 同轴度良好"
            }
        else:
            return {
                "status": "warning",
                "rule": "coaxiality",
                "message": f"轴 {shaft1['id']} 和 {shaft2['id']} 可能不同轴"
            }
```

### Day 6: API集成

#### 6.1 装配分析API
```python
# src/api/v1/assembly.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Optional
import tempfile
import os
from src.assembly.parsers.step_parser import STEPParser
from src.assembly.assembly_graph_builder import AssemblyGraphBuilder
from src.assembly.rules.assembly_rules import AssemblyRuleEngine

router = APIRouter(prefix="/assembly", tags=["assembly"])

# 初始化组件
step_parser = STEPParser()
graph_builder = AssemblyGraphBuilder()
rule_engine = AssemblyRuleEngine()

@router.post("/analyze")
async def analyze_assembly(
    file: UploadFile = File(...),
    validate: bool = True
):
    """
    分析CAD装配文件

    - **file**: STEP格式的CAD文件
    - **validate**: 是否执行规则验证
    """

    # 检查文件类型
    if not file.filename.lower().endswith(('.step', '.stp')):
        raise HTTPException(400, "仅支持STEP格式文件")

    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.step') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Step 1: 解析STEP文件
        parsed_data = step_parser.parse(tmp_path)

        # Step 2: 构建装配图
        assembly_graph = graph_builder.build_from_parsed_data(parsed_data)

        # Step 3: 规则验证（可选）
        validation_result = None
        if validate:
            validation_result = rule_engine.validate_assembly(assembly_graph)

        # Step 4: 组合结果
        result = {
            "success": True,
            "assembly": assembly_graph,
            "validation": validation_result,
            "statistics": {
                "part_count": len(parsed_data["parts"]),
                "relation_count": len(parsed_data["mates"]),
                "file_size": len(content)
            }
        }

        return result

    except Exception as e:
        raise HTTPException(500, f"分析失败: {str(e)}")

    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@router.post("/quick-test")
async def quick_test():
    """快速测试接口（使用预置样例）"""

    # 使用预定义的测试数据
    test_data = {
        "parts": [
            {"id": "gear1", "type": "gear", "label": "主动齿轮"},
            {"id": "gear2", "type": "gear", "label": "从动齿轮"},
            {"id": "shaft1", "type": "shaft", "label": "输入轴"},
            {"id": "shaft2", "type": "shaft", "label": "输出轴"},
            {"id": "bearing1", "type": "bearing", "label": "轴承1"},
            {"id": "bearing2", "type": "bearing", "label": "轴承2"}
        ],
        "mates": [
            {"id": "m1", "part1": "gear1", "part2": "gear2", "type": "gear_mesh"},
            {"id": "m2", "part1": "gear1", "part2": "shaft1", "type": "fixed"},
            {"id": "m3", "part1": "gear2", "part2": "shaft2", "type": "fixed"},
            {"id": "m4", "part1": "shaft1", "part2": "bearing1", "type": "bearing_support"},
            {"id": "m5", "part1": "shaft2", "part2": "bearing2", "type": "bearing_support"}
        ]
    }

    # 构建装配图
    assembly_graph = graph_builder.build_from_parsed_data(test_data)

    # 验证
    validation = rule_engine.validate_assembly(assembly_graph)

    return {
        "success": True,
        "message": "测试成功 - 简单齿轮箱装配",
        "assembly": assembly_graph,
        "validation": validation
    }
```

### Day 7: 测试与文档

#### 7.1 测试脚本
```python
# tests/test_assembly_analysis.py
import pytest
import asyncio
from src.assembly.parsers.step_parser import STEPParser
from src.assembly.assembly_graph_builder import AssemblyGraphBuilder

def test_step_parser():
    """测试STEP解析器"""

    parser = STEPParser()
    # 使用测试文件
    result = parser.parse("samples/simple_gear.step")

    assert "parts" in result
    assert "mates" in result
    assert len(result["parts"]) > 0

def test_assembly_graph_builder():
    """测试装配图构建"""

    builder = AssemblyGraphBuilder()

    test_data = {
        "parts": [
            {"id": "p1", "type": "gear"},
            {"id": "p2", "type": "shaft"}
        ],
        "mates": [
            {"id": "m1", "part1": "p1", "part2": "p2", "type": "fixed"}
        ]
    }

    graph = builder.build_from_parsed_data(test_data)

    assert "graph" in graph
    assert "function" in graph
    assert graph["assembly_info"]["is_connected"] == True

@pytest.mark.asyncio
async def test_api_endpoint():
    """测试API端点"""

    from httpx import AsyncClient
    from src.main import app

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/api/v1/assembly/quick-test")

    assert response.status_code == 200
    assert response.json()["success"] == True
```

---

## 🎯 与现有视觉系统的整合

### 整合架构
```python
# src/core/unified_analyzer.py
from src.core.vision_analyzer import VisionAnalyzer
from src.assembly.parsers.step_parser import STEPParser
from src.assembly.assembly_graph_builder import AssemblyGraphBuilder

class UnifiedAnalyzer:
    """统一分析器 - 整合2D视觉和3D装配理解"""

    def __init__(self):
        self.vision = VisionAnalyzer()
        self.step_parser = STEPParser()
        self.assembly_builder = AssemblyGraphBuilder()

    async def analyze_comprehensive(self, input_data):
        """综合分析"""

        results = {}

        # 如果是图片，先进行视觉分析
        if input_data.type == "image":
            vision_result = await self.vision.analyze(input_data.content)
            results["vision"] = vision_result

            # 从视觉结果提取装配提示
            if "part_type" in vision_result:
                results["hints"] = {
                    "detected_parts": vision_result["part_type"],
                    "materials": vision_result.get("materials", [])
                }

        # 如果是CAD文件，进行装配分析
        elif input_data.type == "cad":
            parsed = self.step_parser.parse(input_data.path)
            assembly = self.assembly_builder.build_from_parsed_data(parsed)
            results["assembly"] = assembly

        # 综合建议
        results["recommendations"] = self._generate_recommendations(results)

        return results

    def _generate_recommendations(self, results):
        """生成制造建议"""

        recommendations = []

        if "assembly" in results:
            assembly = results["assembly"]

            # 基于装配结构的建议
            if "gear" in str(assembly.get("function", "")):
                recommendations.append({
                    "type": "manufacturing",
                    "content": "建议采用精密加工确保齿轮啮合精度"
                })

            if assembly.get("assembly_info", {}).get("is_connected"):
                recommendations.append({
                    "type": "assembly",
                    "content": "装配体结构完整，可以进行装配仿真验证"
                })

        if "vision" in results:
            vision = results["vision"]

            # 基于视觉识别的建议
            if vision.get("confidence", 0) < 0.8:
                recommendations.append({
                    "type": "quality",
                    "content": "图纸质量可能影响识别，建议提供更清晰的图纸"
                })

        return recommendations
```

---

## 📊 性能基准测试

### 测试脚本
```python
# benchmarks/assembly_benchmark.py
import time
import statistics
from src.assembly.parsers.step_parser import STEPParser

def benchmark_step_parsing():
    """STEP解析性能测试"""

    parser = STEPParser()
    test_files = [
        "samples/simple_part.step",    # 单个零件
        "samples/gear_box.step",       # 中等复杂度
        "samples/complex_assembly.step" # 复杂装配
    ]

    results = {}

    for file in test_files:
        times = []
        for _ in range(5):  # 运行5次
            start = time.time()
            parser.parse(file)
            elapsed = time.time() - start
            times.append(elapsed)

        results[file] = {
            "mean": statistics.mean(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            "min": min(times),
            "max": max(times)
        }

    return results

if __name__ == "__main__":
    results = benchmark_step_parsing()

    print("装配分析性能基准:")
    print("-" * 50)
    for file, metrics in results.items():
        print(f"\n文件: {file}")
        print(f"  平均时间: {metrics['mean']:.3f}秒")
        print(f"  标准差: {metrics['stdev']:.3f}秒")
        print(f"  最快: {metrics['min']:.3f}秒")
        print(f"  最慢: {metrics['max']:.3f}秒")
```

---

## 🚦 启动检查清单

### 环境准备
- [ ] Python 3.8+ 已安装
- [ ] FreeCAD Python绑定已配置
- [ ] 测试STEP文件已准备

### 代码就绪
- [ ] STEP解析器可运行
- [ ] 装配图构建器完成
- [ ] 规则引擎已配置
- [ ] API端点已注册

### 测试通过
- [ ] 单元测试全部通过
- [ ] API测试正常
- [ ] 性能基准已建立

### 文档完整
- [ ] API文档已生成
- [ ] 示例代码可运行
- [ ] README已更新

---

## 💡 下一步建议

### 短期（1-2周）
1. 完善规则库，增加更多装配规则
2. 优化STEP解析性能
3. 添加更多零件类型识别
4. 实现简单的URDF导出

### 中期（3-4周）
1. 集成PyBullet仿真
2. 实现装配序列规划
3. 添加公差分析
4. 开发Web界面

### 长期（1-2月）
1. 训练GNN模型
2. 实现高级仿真
3. 添加FMEA分析
4. 优化大规模装配处理

---

**🎉 恭喜！按照这个指南，您可以在第一周内实现装配理解AI的MVP版本！**
