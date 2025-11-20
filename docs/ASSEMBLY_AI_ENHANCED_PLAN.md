# 🚀 装配理解AI增强版实施方案

> 基于专业反馈的改进版本，强化证据驱动、可解释性和评测基线

---

## 📊 核心增强点

### 1. 证据链系统（Evidence Chain）

#### 1.1 证据数据结构
```python
# src/models/evidence.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum

class EvidenceType(str, Enum):
    """证据类型"""
    GEOMETRIC = "geometric"      # 几何证据（接触面、轴线）
    DIMENSIONAL = "dimensional"   # 尺寸证据（标注、公差）
    TEXTUAL = "textual"          # 文本证据（标签、说明）
    RULE_BASED = "rule_based"    # 规则推理
    LEARNED = "learned"          # 机器学习推断

class Evidence(BaseModel):
    """证据对象"""
    type: EvidenceType
    source: str                  # 证据来源（face_123, dimension_456）
    confidence: float            # 置信度 [0,1]
    description: str             # 人类可读描述
    raw_data: Optional[Dict[str, Any]] = None  # 原始数据

    class Config:
        json_schema_extra = {
            "example": {
                "type": "geometric",
                "source": "face_123_124_contact",
                "confidence": 0.95,
                "description": "圆柱面123与124同轴接触，判定为轴承配合",
                "raw_data": {
                    "contact_area": 314.15,
                    "axis_deviation": 0.01
                }
            }
        }

class EvidencedRelation(BaseModel):
    """带证据的装配关系"""
    id: str
    source_part: str
    target_part: str
    relation_type: str
    evidence_chain: List[Evidence]  # 支撑此关系的所有证据
    overall_confidence: float       # 综合置信度

    def calculate_confidence(self) -> float:
        """计算综合置信度"""
        if not self.evidence_chain:
            return 0.0
        # 加权平均，几何证据权重更高
        weights = {
            EvidenceType.GEOMETRIC: 0.4,
            EvidenceType.DIMENSIONAL: 0.3,
            EvidenceType.TEXTUAL: 0.1,
            EvidenceType.RULE_BASED: 0.15,
            EvidenceType.LEARNED: 0.05
        }
        total = sum(e.confidence * weights.get(e.type, 0.1)
                   for e in self.evidence_chain)
        return min(total, 1.0)
```

#### 1.2 证据收集器实现
```python
# src/assembly/evidence_collector.py
import numpy as np
from typing import List, Dict, Any, Tuple
from src.models.evidence import Evidence, EvidenceType

class EvidenceCollector:
    """装配证据收集器"""

    def collect_evidence(
        self,
        part1: Dict,
        part2: Dict,
        geometry_data: Dict
    ) -> List[Evidence]:
        """收集两个零件间的所有证据"""

        evidence_list = []

        # 1. 几何证据
        geometric_evidence = self._collect_geometric_evidence(part1, part2, geometry_data)
        evidence_list.extend(geometric_evidence)

        # 2. 尺寸证据
        dimensional_evidence = self._collect_dimensional_evidence(part1, part2)
        evidence_list.extend(dimensional_evidence)

        # 3. 文本证据
        textual_evidence = self._collect_textual_evidence(part1, part2)
        evidence_list.extend(textual_evidence)

        return evidence_list

    def _collect_geometric_evidence(
        self,
        part1: Dict,
        part2: Dict,
        geometry_data: Dict
    ) -> List[Evidence]:
        """收集几何证据"""

        evidence = []

        # 检查接触面
        contact_faces = self._find_contact_faces(part1, part2, geometry_data)
        for face_pair in contact_faces:
            evidence.append(Evidence(
                type=EvidenceType.GEOMETRIC,
                source=f"face_{face_pair[0]}_{face_pair[1]}",
                confidence=self._calculate_contact_confidence(face_pair),
                description=f"检测到接触面 {face_pair[0]} 和 {face_pair[1]}",
                raw_data={
                    "face1_id": face_pair[0],
                    "face2_id": face_pair[1],
                    "contact_type": self._identify_contact_type(face_pair)
                }
            ))

        # 检查同轴性
        if self._check_coaxiality(part1, part2):
            evidence.append(Evidence(
                type=EvidenceType.GEOMETRIC,
                source="axis_alignment",
                confidence=0.9,
                description="检测到同轴关系",
                raw_data={
                    "axis_deviation": self._calculate_axis_deviation(part1, part2)
                }
            ))

        return evidence

    def _collect_dimensional_evidence(
        self,
        part1: Dict,
        part2: Dict
    ) -> List[Evidence]:
        """收集尺寸证据"""

        evidence = []

        # 检查配合尺寸
        if "dimensions" in part1 and "dimensions" in part2:
            matching_dims = self._find_matching_dimensions(
                part1["dimensions"],
                part2["dimensions"]
            )

            for dim_match in matching_dims:
                evidence.append(Evidence(
                    type=EvidenceType.DIMENSIONAL,
                    source=f"dimension_match_{dim_match['id']}",
                    confidence=dim_match["confidence"],
                    description=f"尺寸匹配: {dim_match['value']}",
                    raw_data=dim_match
                ))

        return evidence

    def _collect_textual_evidence(
        self,
        part1: Dict,
        part2: Dict
    ) -> List[Evidence]:
        """收集文本证据"""

        evidence = []

        # 从标签提取
        label1 = part1.get("label", "")
        label2 = part2.get("label", "")

        # 关键词匹配
        keywords = {
            "gear": ["齿轮", "gear", "Z="],
            "bearing": ["轴承", "bearing", "6201", "6202"],
            "shaft": ["轴", "shaft", "axis"]
        }

        for key, words in keywords.items():
            if any(word in label1 for word in words) and \
               any(word in label2 for word in words):
                evidence.append(Evidence(
                    type=EvidenceType.TEXTUAL,
                    source="label_analysis",
                    confidence=0.7,
                    description=f"标签中检测到 {key} 相关关键词",
                    raw_data={"keyword": key, "labels": [label1, label2]}
                ))

        return evidence

    def _calculate_contact_confidence(self, face_pair: Tuple) -> float:
        """计算接触置信度"""
        # 基于接触面积、法向量等计算
        return 0.85  # 简化实现

    def _identify_contact_type(self, face_pair: Tuple) -> str:
        """识别接触类型"""
        # 平面接触、圆柱面接触等
        return "cylindrical"  # 简化实现
```

---

### 2. 装配图规范化（Canonicalization）

#### 2.1 规范化处理器
```python
# src/assembly/graph_normalizer.py
import numpy as np
from typing import Dict, List, Any
import hashlib

class AssemblyGraphNormalizer:
    """装配图规范化处理器"""

    def __init__(self):
        self.unit_conversion = {
            "mm": 1.0,
            "cm": 10.0,
            "m": 1000.0,
            "inch": 25.4
        }

    def normalize(self, assembly_graph: Dict) -> Dict:
        """规范化装配图"""

        normalized = assembly_graph.copy()

        # 1. 坐标系对齐
        normalized = self._align_coordinate_system(normalized)

        # 2. 单位统一
        normalized = self._unify_units(normalized)

        # 3. 去重处理
        normalized = self._remove_duplicates(normalized)

        # 4. 特征ID稳定化
        normalized = self._stabilize_feature_ids(normalized)

        # 5. 计算规范哈希
        normalized["canonical_hash"] = self._compute_canonical_hash(normalized)

        return normalized

    def _align_coordinate_system(self, graph: Dict) -> Dict:
        """对齐坐标系到标准方向"""

        # 找到主轴（最长的轴类零件）
        main_axis = self._find_main_axis(graph)

        if main_axis:
            # 计算旋转矩阵，使主轴与X轴对齐
            rotation_matrix = self._compute_alignment_matrix(main_axis)

            # 应用变换到所有零件
            for part in graph.get("parts", []):
                if "center_of_mass" in part:
                    part["center_of_mass"] = self._transform_point(
                        part["center_of_mass"],
                        rotation_matrix
                    )
                if "bounding_box" in part:
                    part["bounding_box"] = self._transform_bbox(
                        part["bounding_box"],
                        rotation_matrix
                    )

        return graph

    def _unify_units(self, graph: Dict) -> Dict:
        """统一到毫米单位"""

        detected_unit = self._detect_unit(graph)
        conversion_factor = self.unit_conversion.get(detected_unit, 1.0)

        if conversion_factor != 1.0:
            for part in graph.get("parts", []):
                # 转换所有尺寸相关字段
                if "dimensions" in part:
                    part["dimensions"] = {
                        k: v * conversion_factor
                        for k, v in part["dimensions"].items()
                    }
                if "volume" in part:
                    part["volume"] *= (conversion_factor ** 3)

        graph["units"] = "mm"
        return graph

    def _remove_duplicates(self, graph: Dict) -> Dict:
        """去除重复的面和轴"""

        # 去重接触面
        unique_mates = []
        seen_pairs = set()

        for mate in graph.get("mates", []):
            # 创建标准化的配对ID（顺序无关）
            pair_id = tuple(sorted([mate["part1"], mate["part2"]]))

            if pair_id not in seen_pairs:
                seen_pairs.add(pair_id)
                unique_mates.append(mate)

        graph["mates"] = unique_mates
        return graph

    def _stabilize_feature_ids(self, graph: Dict) -> Dict:
        """稳定化特征ID"""

        # 基于几何特征生成稳定ID
        for part in graph.get("parts", []):
            # 基于形状特征生成稳定ID
            shape_hash = self._compute_shape_hash(part)
            part["stable_id"] = f"part_{shape_hash[:8]}"

        # 更新引用
        for mate in graph.get("mates", []):
            mate["part1"] = self._get_stable_id(graph, mate["part1"])
            mate["part2"] = self._get_stable_id(graph, mate["part2"])

        return graph

    def _compute_canonical_hash(self, graph: Dict) -> str:
        """计算规范哈希值"""

        # 提取关键特征
        canonical_features = {
            "part_count": len(graph.get("parts", [])),
            "mate_count": len(graph.get("mates", [])),
            "part_types": sorted([p.get("type", "unknown")
                                for p in graph.get("parts", [])]),
            "mate_types": sorted([m.get("type", "unknown")
                                for m in graph.get("mates", [])])
        }

        # 计算哈希
        feature_str = str(canonical_features)
        return hashlib.sha256(feature_str.encode()).hexdigest()[:16]

    def _detect_unit(self, graph: Dict) -> str:
        """检测当前单位"""

        # 基于尺寸范围启发式判断
        all_dims = []
        for part in graph.get("parts", []):
            if "bounding_box" in part:
                bbox = part["bounding_box"]
                size = max(bbox.get("max", [0])) - min(bbox.get("min", [0]))
                all_dims.append(size)

        if all_dims:
            avg_dim = np.mean(all_dims)
            if avg_dim < 10:  # 可能是米
                return "m"
            elif avg_dim > 1000:  # 可能是微米
                return "um"
            else:  # 默认毫米
                return "mm"

        return "mm"
```

---

### 3. 版本化知识库

#### 3.1 知识库结构
```yaml
# knowledge_base/assembly/rules/v1.0.0/gear_meshing.yaml
version: 1.0.0
name: gear_meshing_rules
description: 齿轮啮合规则库
created: 2025-01-10
author: CAD-ML-Platform Team

rules:
  - id: spur_gear_mesh
    name: 直齿轮啮合
    conditions:
      - type: both_parts_are_gears
      - type: parallel_axes
      - type: center_distance_matches_pitch_circles
    parameters:
      module:
        required: true
        unit: mm
      pressure_angle:
        default: 20
        unit: degree
      backlash:
        min: 0.05
        max: 0.3
        unit: mm
    evidence_requirements:
      - geometric: cylindrical_contact
      - dimensional: matching_module
    confidence_weight: 0.9

  - id: helical_gear_mesh
    name: 斜齿轮啮合
    conditions:
      - type: both_parts_are_gears
      - type: parallel_or_crossing_axes
      - type: helix_angle_matches
    parameters:
      helix_angle:
        required: true
        unit: degree
      hand:
        options: [left, right]
    evidence_requirements:
      - geometric: helical_contact_pattern
    confidence_weight: 0.85

  - id: bevel_gear_mesh
    name: 锥齿轮啮合
    conditions:
      - type: both_parts_are_gears
      - type: intersecting_axes
      - type: cone_angles_sum_to_shaft_angle
    parameters:
      shaft_angle:
        default: 90
        unit: degree
    confidence_weight: 0.8

mappings:
  # CAD软件mate类型到标准关节的映射
  solidworks:
    - mate_type: "Gear"
      maps_to: gear_mesh
      extract_params:
        - ratio: from_property("GearRatio")
        - module: from_dimension("Module")

  fusion360:
    - mate_type: "Motion:Rotation"
      with_conditions:
        - has_teeth_geometry
      maps_to: gear_mesh
      extract_params:
        - ratio: calculate_from_teeth_count

  creo:
    - constraint_type: "Gear Pair"
      maps_to: gear_mesh
      extract_params:
        - module: from_parameter("d10")
```

#### 3.2 知识库管理器
```python
# src/assembly/knowledge_manager.py
import yaml
import semver
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

class KnowledgeBaseManager:
    """知识库版本管理器"""

    def __init__(self, kb_path: str = "knowledge_base/assembly"):
        self.kb_path = Path(kb_path)
        self.current_version = None
        self.loaded_rules = {}
        self.change_log = []

    def load_version(self, version: str = "latest") -> Dict:
        """加载指定版本的知识库"""

        if version == "latest":
            version = self._get_latest_version()

        version_path = self.kb_path / "rules" / f"v{version}"

        if not version_path.exists():
            raise ValueError(f"Version {version} not found")

        # 加载所有规则文件
        rules = {}
        for rule_file in version_path.glob("*.yaml"):
            with open(rule_file, 'r', encoding='utf-8') as f:
                rule_data = yaml.safe_load(f)
                rules[rule_data["name"]] = rule_data

        self.current_version = version
        self.loaded_rules = rules

        # 加载变更日志
        self._load_changelog(version)

        return rules

    def _get_latest_version(self) -> str:
        """获取最新版本号"""

        versions = []
        rules_path = self.kb_path / "rules"

        for version_dir in rules_path.iterdir():
            if version_dir.is_dir() and version_dir.name.startswith("v"):
                version_str = version_dir.name[1:]  # 去掉'v'前缀
                try:
                    versions.append(semver.VersionInfo.parse(version_str))
                except:
                    continue

        if versions:
            latest = max(versions)
            return str(latest)

        return "1.0.0"

    def _load_changelog(self, version: str):
        """加载变更日志"""

        changelog_file = self.kb_path / "CHANGELOG.md"
        if changelog_file.exists():
            with open(changelog_file, 'r', encoding='utf-8') as f:
                self.change_log = f.readlines()

    def get_rule(self, rule_name: str, rule_id: str) -> Optional[Dict]:
        """获取特定规则"""

        if rule_name in self.loaded_rules:
            rules = self.loaded_rules[rule_name].get("rules", [])
            for rule in rules:
                if rule["id"] == rule_id:
                    return rule

        return None

    def get_mapping(self, cad_system: str, mate_type: str) -> Optional[Dict]:
        """获取CAD系统映射"""

        for rule_set in self.loaded_rules.values():
            mappings = rule_set.get("mappings", {})
            if cad_system in mappings:
                for mapping in mappings[cad_system]:
                    if mapping["mate_type"] == mate_type:
                        return mapping

        return None

    def validate_rule_update(self, new_rules: Dict) -> Dict:
        """验证规则更新"""

        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        # 检查向后兼容性
        for rule_name, rule_data in new_rules.items():
            if rule_name in self.loaded_rules:
                old_version = self.loaded_rules[rule_name].get("version", "0.0.0")
                new_version = rule_data.get("version", "0.0.0")

                if semver.compare(new_version, old_version) <= 0:
                    validation_result["errors"].append(
                        f"Version must be higher than {old_version}"
                    )
                    validation_result["valid"] = False

        return validation_result
```

---

### 4. 评测基线与CI集成

#### 4.1 评测指标定义
```python
# src/evaluation/metrics.py
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import precision_recall_fscore_support

class AssemblyMetrics:
    """装配分析评测指标"""

    def __init__(self):
        self.results = []

    def evaluate(self, predicted: Dict, ground_truth: Dict) -> Dict:
        """评测单个样本"""

        metrics = {
            "graph_quality": self._evaluate_graph_quality(predicted, ground_truth),
            "physics_consistency": self._evaluate_physics(predicted, ground_truth),
            "evidence_quality": self._evaluate_evidence(predicted),
            "performance": self._evaluate_performance(predicted)
        }

        # 计算总分
        metrics["overall_score"] = self._calculate_overall_score(metrics)

        return metrics

    def _evaluate_graph_quality(self, pred: Dict, truth: Dict) -> Dict:
        """评测装配图质量"""

        # 提取边类型
        pred_edges = [(e["part1"], e["part2"], e["type"])
                     for e in pred.get("mates", [])]
        truth_edges = [(e["part1"], e["part2"], e["type"])
                      for e in truth.get("mates", [])]

        # 计算准确率、召回率、F1
        true_positives = len(set(pred_edges) & set(truth_edges))
        false_positives = len(set(pred_edges) - set(truth_edges))
        false_negatives = len(set(truth_edges) - set(pred_edges))

        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        # 关节类型准确率
        joint_accuracy = self._calculate_joint_type_accuracy(pred, truth)

        # 连通性检查
        connectivity_score = self._check_connectivity(pred, truth)

        return {
            "edge_precision": precision,
            "edge_recall": recall,
            "edge_f1": f1,
            "joint_type_accuracy": joint_accuracy,
            "connectivity_score": connectivity_score
        }

    def _evaluate_physics(self, pred: Dict, truth: Dict) -> Dict:
        """评测物理一致性"""

        physics_metrics = {
            "simulation_ready": False,
            "dof_match": 0.0,
            "transmission_ratio_error": 0.0
        }

        # 检查是否可仿真
        if "urdf" in pred or "simulation_ready" in pred:
            physics_metrics["simulation_ready"] = True

        # 自由度匹配
        pred_dof = pred.get("degrees_of_freedom", -1)
        truth_dof = truth.get("degrees_of_freedom", -1)
        if pred_dof == truth_dof and pred_dof >= 0:
            physics_metrics["dof_match"] = 1.0

        # 传动比误差
        pred_ratio = self._extract_transmission_ratio(pred)
        truth_ratio = self._extract_transmission_ratio(truth)
        if pred_ratio and truth_ratio:
            error = abs(pred_ratio - truth_ratio) / truth_ratio
            physics_metrics["transmission_ratio_error"] = 1.0 - min(error, 1.0)

        return physics_metrics

    def _evaluate_evidence(self, pred: Dict) -> Dict:
        """评测证据质量"""

        evidence_metrics = {
            "evidence_coverage": 0.0,
            "average_confidence": 0.0,
            "evidence_diversity": 0.0
        }

        all_evidence = []
        for mate in pred.get("mates", []):
            if "evidence_chain" in mate:
                all_evidence.extend(mate["evidence_chain"])

        if all_evidence:
            # 证据覆盖率
            mates_with_evidence = sum(1 for m in pred.get("mates", [])
                                     if "evidence_chain" in m)
            total_mates = len(pred.get("mates", []))
            evidence_metrics["evidence_coverage"] = mates_with_evidence / (total_mates + 1e-10)

            # 平均置信度
            confidences = [e.get("confidence", 0) for e in all_evidence]
            evidence_metrics["average_confidence"] = np.mean(confidences)

            # 证据多样性
            evidence_types = set(e.get("type", "") for e in all_evidence)
            evidence_metrics["evidence_diversity"] = len(evidence_types) / 5.0  # 假设5种类型

        return evidence_metrics

    def _evaluate_performance(self, pred: Dict) -> Dict:
        """评测性能指标"""

        return {
            "processing_time": pred.get("processing_time", -1),
            "cache_hit": pred.get("cache_hit", False),
            "cost": pred.get("cost", 0.0)
        }

    def _calculate_overall_score(self, metrics: Dict) -> float:
        """计算综合得分"""

        weights = {
            "edge_f1": 0.3,
            "joint_type_accuracy": 0.2,
            "simulation_ready": 0.15,
            "evidence_coverage": 0.15,
            "average_confidence": 0.1,
            "connectivity_score": 0.1
        }

        score = 0.0
        for key, weight in weights.items():
            # 递归查找指标值
            value = self._find_metric_value(metrics, key)
            if value is not None:
                score += value * weight

        return score

    def _find_metric_value(self, metrics: Dict, key: str) -> Optional[float]:
        """递归查找指标值"""

        for k, v in metrics.items():
            if k == key:
                return float(v) if isinstance(v, bool) else v
            elif isinstance(v, dict):
                result = self._find_metric_value(v, key)
                if result is not None:
                    return result

        return None
```

#### 4.2 CI集成配置
```yaml
# .github/workflows/assembly_tests.yml
name: Assembly AI Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run unit tests
      run: |
        pytest tests/assembly/ -v --cov=src/assembly

    - name: Run evaluation baseline
      run: |
        python scripts/run_baseline_evaluation.py

    - name: Check metrics threshold
      run: |
        python scripts/check_metrics.py --min-f1 0.75 --min-confidence 0.7

    - name: Generate report
      run: |
        python scripts/generate_evaluation_report.py > evaluation_report.md

    - name: Upload artifacts
      uses: actions/upload-artifact@v2
      with:
        name: evaluation-report
        path: evaluation_report.md
```

#### 4.3 基准测试集
```python
# tests/assembly/golden_cases.py
"""黄金测试用例集"""

GOLDEN_CASES = [
    {
        "name": "simple_gear_train",
        "description": "简单齿轮系",
        "input": {
            "parts": [
                {"id": "gear1", "type": "gear", "teeth": 20},
                {"id": "gear2", "type": "gear", "teeth": 60},
                {"id": "shaft1", "type": "shaft"},
                {"id": "shaft2", "type": "shaft"}
            ],
            "mates": [
                {"part1": "gear1", "part2": "gear2", "type": "gear_mesh"},
                {"part1": "gear1", "part2": "shaft1", "type": "fixed"},
                {"part1": "gear2", "part2": "shaft2", "type": "fixed"}
            ]
        },
        "expected": {
            "function": "齿轮减速器",
            "transmission_ratio": 3.0,
            "degrees_of_freedom": 1,
            "is_valid": True
        }
    },
    {
        "name": "belt_drive",
        "description": "皮带传动",
        "input": {
            "parts": [
                {"id": "pulley1", "type": "pulley", "diameter": 100},
                {"id": "pulley2", "type": "pulley", "diameter": 200},
                {"id": "belt", "type": "belt"}
            ],
            "mates": [
                {"part1": "pulley1", "part2": "belt", "type": "belt_contact"},
                {"part1": "pulley2", "part2": "belt", "type": "belt_contact"}
            ]
        },
        "expected": {
            "function": "皮带传动系统",
            "transmission_ratio": 2.0,
            "is_valid": True
        }
    },
    {
        "name": "bearing_support",
        "description": "轴承支撑",
        "input": {
            "parts": [
                {"id": "shaft", "type": "shaft", "diameter": 30},
                {"id": "bearing1", "type": "bearing", "inner_diameter": 30},
                {"id": "bearing2", "type": "bearing", "inner_diameter": 30},
                {"id": "housing", "type": "housing"}
            ],
            "mates": [
                {"part1": "shaft", "part2": "bearing1", "type": "bearing_fit"},
                {"part1": "shaft", "part2": "bearing2", "type": "bearing_fit"},
                {"part1": "bearing1", "part2": "housing", "type": "fixed"},
                {"part1": "bearing2", "part2": "housing", "type": "fixed"}
            ]
        },
        "expected": {
            "function": "轴承支撑系统",
            "is_valid": True,
            "constraints": ["proper_bearing_spacing", "adequate_support"]
        }
    }
]

def get_golden_case(name: str):
    """获取指定的黄金测试用例"""
    for case in GOLDEN_CASES:
        if case["name"] == name:
            return case
    return None

def run_golden_tests():
    """运行所有黄金测试"""
    from src.assembly.assembly_graph_builder import AssemblyGraphBuilder
    from src.evaluation.metrics import AssemblyMetrics

    builder = AssemblyGraphBuilder()
    evaluator = AssemblyMetrics()

    results = []
    for case in GOLDEN_CASES:
        print(f"Testing {case['name']}...")

        # 构建装配图
        predicted = builder.build_from_parsed_data(case["input"])

        # 评测
        metrics = evaluator.evaluate(predicted, case["expected"])

        results.append({
            "case": case["name"],
            "metrics": metrics,
            "passed": metrics["overall_score"] >= 0.75
        })

    return results
```

---

### 5. 增强的API契约

```python
# src/api/v1/assembly_enhanced.py
from fastapi import APIRouter, UploadFile, File, Header
from typing import Optional, Dict
from src.models.evidence import EvidencedRelation
from pydantic import BaseModel
import hashlib

router = APIRouter(prefix="/assembly", tags=["assembly-enhanced"])

class EnhancedAnalysisRequest(BaseModel):
    """增强分析请求"""
    idempotency_key: Optional[str] = None
    enable_evidence: bool = True
    enable_normalization: bool = True
    confidence_threshold: float = 0.7
    cache_mode: str = "auto"  # auto, force_refresh, cache_only

class EnhancedAnalysisResponse(BaseModel):
    """增强分析响应"""
    request_id: str
    input_hash: str
    assembly_graph: Dict
    evidence: List[EvidencedRelation]
    uncertainty: Dict[str, float]
    canonical_hash: str
    suggestions: List[Dict]
    engine_support_matrix: Dict
    cache_hit: bool
    delta_report: Optional[Dict] = None

@router.post("/analyze", response_model=EnhancedAnalysisResponse)
async def analyze_with_evidence(
    file: UploadFile = File(...),
    request: EnhancedAnalysisRequest = None,
    idempotency_key: Optional[str] = Header(None)
):
    """带证据链的装配分析"""

    # 幂等性处理
    if idempotency_key:
        cached = await check_idempotency(idempotency_key)
        if cached:
            return cached

    # 计算输入哈希
    content = await file.read()
    input_hash = hashlib.sha256(content).hexdigest()

    # 检查缓存
    if request.cache_mode != "force_refresh":
        cached_result = await get_from_cache(input_hash)
        if cached_result:
            cached_result["cache_hit"] = True

            # 生成Delta报告
            if request.cache_mode == "auto":
                cached_result["delta_report"] = await generate_delta(
                    cached_result,
                    input_hash
                )

            return cached_result

    # 执行分析...
    # (实现细节省略)

    return response

@router.get("/engine-support-matrix")
async def get_engine_support():
    """获取引擎支持矩阵"""

    return {
        "urdf": {
            "supported_joints": ["fixed", "revolute", "prismatic", "continuous"],
            "unsupported": ["gear", "belt", "chain"],
            "workarounds": {
                "gear": "Use revolute with transmission ratio annotation",
                "belt": "Use coupled revolute joints"
            }
        },
        "pybullet": {
            "supported_joints": ["all_urdf", "custom_constraints"],
            "performance": "fast",
            "accuracy": "medium"
        },
        "chrono": {
            "supported_joints": ["all", "gear_pairs", "belt_drives"],
            "performance": "medium",
            "accuracy": "high"
        },
        "mujoco": {
            "supported_joints": ["all_urdf", "tendons", "actuators"],
            "performance": "very_fast",
            "accuracy": "high"
        }
    }
```

---

## 📊 实施路线图（1周内可完成）

### Day 1-2: 证据系统
- [ ] 实现Evidence数据模型
- [ ] 开发EvidenceCollector
- [ ] 集成到现有分析流程

### Day 3: 规范化处理
- [ ] 实现AssemblyGraphNormalizer
- [ ] 添加单位转换和坐标对齐
- [ ] 生成规范哈希

### Day 4: 知识库与映射
- [ ] 创建YAML格式知识库
- [ ] 实现版本管理
- [ ] 添加CAD系统映射表

### Day 5: 评测基线
- [ ] 定义6个黄金测试用例
- [ ] 实现评测指标计算
- [ ] 建立F1≥0.75基线

### Day 6: API增强
- [ ] 添加证据输出
- [ ] 实现幂等性
- [ ] 缓存与Delta报告

### Day 7: CI集成
- [ ] 配置GitHub Actions
- [ ] 自动运行评测
- [ ] 生成报告

---

## 🎯 预期成果

通过这些增强，系统将达到：

1. **可解释性**：每个装配关系都有完整证据链
2. **稳定性**：规范化确保结果一致性
3. **可维护性**：版本化知识库便于迭代
4. **可测量性**：量化指标驱动改进
5. **生产就绪**：幂等性、缓存、Delta报告

---

**您的建议非常专业，这个增强版方案将让装配理解AI真正达到生产级别！**