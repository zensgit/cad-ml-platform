"""
Delta报告标准化
基于RFC 6902 JSON Patch标准 + 自定义evidence diff
"""

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import jsonpatch


class ChangeType(str, Enum):
    """变更类型"""

    ADD = "add"
    REMOVE = "remove"
    REPLACE = "replace"
    MOVE = "move"
    COPY = "copy"
    TEST = "test"


@dataclass
class EvidenceChange:
    """证据变更"""

    type: ChangeType
    path: str
    evidence_id: str
    old_value: Optional[Dict] = None
    new_value: Optional[Dict] = None
    confidence_delta: Optional[float] = None
    reason: Optional[str] = None


@dataclass
class DeltaReport:
    """Delta报告"""

    from_version: str
    to_version: str
    from_hash: str
    to_hash: str
    timestamp: str

    # RFC 6902标准部分
    json_patch: List[Dict]

    # 自定义证据diff
    evidence_changes: List[EvidenceChange]

    # 汇总统计
    summary: Dict[str, Any]

    # 可审阅的变更描述
    human_readable: List[str]


class DeltaReporter:
    """Delta报告生成器"""

    def __init__(self):
        self.cache = {}  # 缓存历史版本用于比对

    def generate_delta(
        self, old_assembly: Dict, new_assembly: Dict, include_evidence: bool = True
    ) -> DeltaReport:
        """
        生成Delta报告

        Args:
            old_assembly: 旧版本装配图
            new_assembly: 新版本装配图
            include_evidence: 是否包含证据变更

        Returns:
            标准化的Delta报告
        """
        import datetime

        # 计算版本哈希
        old_hash = self._compute_hash(old_assembly)
        new_hash = self._compute_hash(new_assembly)

        # 生成RFC 6902 JSON Patch
        json_patch = self._generate_json_patch(old_assembly, new_assembly)

        # 生成证据变更
        evidence_changes = []
        if include_evidence:
            evidence_changes = self._generate_evidence_diff(old_assembly, new_assembly)

        # 生成汇总
        summary = self._generate_summary(json_patch, evidence_changes)

        # 生成人类可读描述
        human_readable = self._generate_human_readable(json_patch, evidence_changes)

        return DeltaReport(
            from_version=old_assembly.get("version", "unknown"),
            to_version=new_assembly.get("version", "unknown"),
            from_hash=old_hash,
            to_hash=new_hash,
            timestamp=datetime.datetime.now().isoformat(),
            json_patch=json_patch,
            evidence_changes=evidence_changes,
            summary=summary,
            human_readable=human_readable,
        )

    def _generate_json_patch(self, old: Dict, new: Dict) -> List[Dict]:
        """生成RFC 6902标准的JSON Patch"""

        # 使用jsonpatch库生成标准patch
        patch = jsonpatch.make_patch(old, new)

        # 转换为列表格式
        patch_list = []
        for operation in patch:
            patch_list.append(dict(operation))

        return patch_list

    def _generate_evidence_diff(
        self, old_assembly: Dict, new_assembly: Dict
    ) -> List[EvidenceChange]:
        """生成证据差异"""

        changes = []

        # 提取旧的证据
        old_evidence = self._extract_all_evidence(old_assembly)
        new_evidence = self._extract_all_evidence(new_assembly)

        # 找出新增的证据
        for eid, evidence in new_evidence.items():
            if eid not in old_evidence:
                changes.append(
                    EvidenceChange(
                        type=ChangeType.ADD,
                        path=evidence["path"],
                        evidence_id=eid,
                        new_value=evidence["data"],
                        confidence_delta=evidence["data"].get("confidence", 0),
                        reason="新增证据",
                    )
                )

        # 找出删除的证据
        for eid, evidence in old_evidence.items():
            if eid not in new_evidence:
                changes.append(
                    EvidenceChange(
                        type=ChangeType.REMOVE,
                        path=evidence["path"],
                        evidence_id=eid,
                        old_value=evidence["data"],
                        confidence_delta=-evidence["data"].get("confidence", 0),
                        reason="移除证据",
                    )
                )

        # 找出变更的证据
        for eid in old_evidence.keys() & new_evidence.keys():
            old_ev = old_evidence[eid]
            new_ev = new_evidence[eid]

            if old_ev["data"] != new_ev["data"]:
                old_conf = old_ev["data"].get("confidence", 0)
                new_conf = new_ev["data"].get("confidence", 0)

                # 判断变更类型
                if new_conf < old_conf:
                    reason = f"置信度降低 {old_conf:.2f} → {new_conf:.2f}"
                elif new_conf > old_conf:
                    reason = f"置信度提升 {old_conf:.2f} → {new_conf:.2f}"
                else:
                    reason = "证据内容更新"

                changes.append(
                    EvidenceChange(
                        type=ChangeType.REPLACE,
                        path=new_ev["path"],
                        evidence_id=eid,
                        old_value=old_ev["data"],
                        new_value=new_ev["data"],
                        confidence_delta=new_conf - old_conf,
                        reason=reason,
                    )
                )

        return changes

    def _extract_all_evidence(self, assembly: Dict) -> Dict[str, Dict]:
        """提取所有证据"""

        evidence_map = {}

        # 从mates中提取证据
        for mate in assembly.get("mates", []):
            if "evidence_chain" in mate:
                for i, evidence in enumerate(mate["evidence_chain"]):
                    eid = f"{mate['id']}_evidence_{i}"
                    evidence_map[eid] = {
                        "path": f"/mates/{mate['id']}/evidence_chain/{i}",
                        "data": evidence,
                    }

        return evidence_map

    def _generate_summary(
        self, json_patch: List[Dict], evidence_changes: List[EvidenceChange]
    ) -> Dict[str, Any]:
        """生成汇总统计"""

        # 统计JSON Patch操作
        patch_stats = {}
        for op in json_patch:
            op_type = op.get("op", "unknown")
            patch_stats[op_type] = patch_stats.get(op_type, 0) + 1

        # 统计证据变更
        evidence_stats = {
            "added": sum(1 for c in evidence_changes if c.type == ChangeType.ADD),
            "removed": sum(1 for c in evidence_changes if c.type == ChangeType.REMOVE),
            "modified": sum(1 for c in evidence_changes if c.type == ChangeType.REPLACE),
            "total_confidence_delta": sum(c.confidence_delta or 0 for c in evidence_changes),
        }

        # 识别重要变更
        critical_changes = []

        # 检查关节类型变更
        for op in json_patch:
            if "mates" in op.get("path", "") and "/type" in op.get("path", ""):
                critical_changes.append(f"关节类型变更: {op.get('path')}")

        # 检查置信度大幅下降
        for change in evidence_changes:
            if change.confidence_delta and change.confidence_delta < -0.3:
                critical_changes.append(f"置信度大幅下降: {change.evidence_id}")

        return {
            "patch_operations": patch_stats,
            "evidence_changes": evidence_stats,
            "critical_changes": critical_changes,
            "total_changes": len(json_patch) + len(evidence_changes),
        }

    def _generate_human_readable(
        self, json_patch: List[Dict], evidence_changes: List[EvidenceChange]
    ) -> List[str]:
        """生成人类可读的变更描述"""

        descriptions = []

        # 描述结构变更
        for op in json_patch:
            op_type = op.get("op")
            path = op.get("path", "")

            if op_type == "add":
                if "/parts/" in path:
                    descriptions.append(f"✅ 新增零件: {path}")
                elif "/mates/" in path:
                    descriptions.append(f"✅ 新增装配关系: {path}")

            elif op_type == "remove":
                if "/parts/" in path:
                    descriptions.append(f"❌ 移除零件: {path}")
                elif "/mates/" in path:
                    descriptions.append(f"❌ 移除装配关系: {path}")

            elif op_type == "replace":
                value = op.get("value")
                if "/type" in path:
                    descriptions.append(f"🔄 类型变更: {path} → {value}")
                elif "/confidence" in path:
                    descriptions.append(f"🔄 置信度更新: {path} → {value}")

        # 描述证据变更
        for change in evidence_changes:
            if change.type == ChangeType.ADD:
                descriptions.append(
                    f"📎 新增{change.evidence_id}证据 (置信度: {change.new_value.get('confidence', 0):.2f})"
                )
            elif change.type == ChangeType.REMOVE:
                descriptions.append(f"🗑️ 移除{change.evidence_id}证据")
            elif change.type == ChangeType.REPLACE and change.reason:
                descriptions.append(f"♻️ {change.evidence_id}: {change.reason}")

        return descriptions

    def _compute_hash(self, assembly: Dict) -> str:
        """计算装配图哈希"""

        # 提取关键特征
        key_features = {
            "parts": sorted([p.get("id", "") for p in assembly.get("parts", [])]),
            "mates": sorted([m.get("id", "") for m in assembly.get("mates", [])]),
            "types": sorted([m.get("type", "") for m in assembly.get("mates", [])]),
        }

        # 计算哈希
        feature_str = json.dumps(key_features, sort_keys=True)
        return hashlib.sha256(feature_str.encode()).hexdigest()[:16]

    def apply_patch(self, assembly: Dict, patch: List[Dict]) -> Dict:
        """应用JSON Patch到装配图"""

        # 创建副本避免修改原始数据
        result = json.loads(json.dumps(assembly))

        # 应用patch
        json_patch = jsonpatch.JsonPatch(patch)
        result = json_patch.apply(result)

        return result

    def validate_patch(self, assembly: Dict, patch: List[Dict]) -> Tuple[bool, Optional[str]]:
        """验证patch是否可以应用"""

        try:
            self.apply_patch(assembly, patch)
            return True, None
        except Exception as e:
            return False, str(e)


class DeltaReviewUI:
    """Delta审阅界面生成器"""

    @staticmethod
    def generate_review_html(delta_report: DeltaReport) -> str:
        """生成可审阅的HTML界面"""

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>装配图Delta报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 10px; }}
                .section {{ margin: 20px 0; }}
                .add {{ color: green; }}
                .remove {{ color: red; }}
                .modify {{ color: orange; }}
                .evidence {{ background: #f9f9f9; padding: 5px; margin: 5px 0; }}
                .critical {{ background: #ffe0e0; padding: 10px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>装配图Delta报告</h1>
                <p>从版本: {delta_report.from_hash[:8]} → 到版本: {delta_report.to_hash[:8]}</p>
                <p>生成时间: {delta_report.timestamp}</p>
            </div>

            <div class="section">
                <h2>📊 变更汇总</h2>
                <ul>
                    <li>总变更数: {delta_report.summary['total_changes']}</li>
                    <li>证据新增: {delta_report.summary['evidence_changes']['added']}</li>
                    <li>证据移除: {delta_report.summary['evidence_changes']['removed']}</li>
                    <li>证据修改: {delta_report.summary['evidence_changes']['modified']}</li>
                    <li>置信度总变化: {delta_report.summary['evidence_changes']['total_confidence_delta']:.3f}</li>
                </ul>
            </div>

            {"<div class='section critical'><h2>⚠️ 重要变更</h2><ul>" +
             "".join(f"<li>{c}</li>" for c in delta_report.summary['critical_changes']) +
             "</ul></div>" if delta_report.summary['critical_changes'] else ""}

            <div class="section">
                <h2>📝 详细变更</h2>
                <ul>
                    {"".join(f"<li>{desc}</li>" for desc in delta_report.human_readable)}
                </ul>
            </div>

            <div class="section">
                <h2>🔧 JSON Patch (RFC 6902)</h2>
                <pre>{json.dumps(delta_report.json_patch, indent=2)}</pre>
            </div>
        </body>
        </html>
        """

        return html


# 使用示例
if __name__ == "__main__":
    # 创建Delta报告器
    reporter = DeltaReporter()

    # 模拟旧版本
    old_assembly = {
        "version": "1.0.0",
        "parts": [{"id": "gear1", "type": "gear"}, {"id": "shaft1", "type": "shaft"}],
        "mates": [
            {
                "id": "m1",
                "part1": "gear1",
                "part2": "shaft1",
                "type": "fixed",
                "evidence_chain": [{"type": "geometric", "confidence": 0.8}],
            }
        ],
    }

    # 模拟新版本
    new_assembly = {
        "version": "1.0.1",
        "parts": [
            {"id": "gear1", "type": "gear"},
            {"id": "shaft1", "type": "shaft"},
            {"id": "bearing1", "type": "bearing"},  # 新增
        ],
        "mates": [
            {
                "id": "m1",
                "part1": "gear1",
                "part2": "shaft1",
                "type": "keyed",  # 类型变更
                "evidence_chain": [
                    {"type": "geometric", "confidence": 0.9},  # 置信度提升
                    {"type": "textual", "confidence": 0.7},  # 新增证据
                ],
            }
        ],
    }

    # 生成Delta报告
    delta = reporter.generate_delta(old_assembly, new_assembly)

    # 打印报告
    print("Delta报告汇总:")
    print(json.dumps(delta.summary, indent=2, ensure_ascii=False))
    print("\n人类可读变更:")
    for desc in delta.human_readable:
        print(f"  {desc}")
