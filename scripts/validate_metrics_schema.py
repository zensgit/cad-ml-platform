#!/usr/bin/env python3
"""
Validate Metrics Against Federation Schema
验证指标是否符合联邦模式标准
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime
import yaml


@dataclass
class ValidationError:
    """验证错误"""
    metric_name: str
    error_type: str
    message: str
    severity: str  # error, warning, info


@dataclass
class ValidationReport:
    """验证报告"""
    timestamp: datetime
    total_metrics: int
    valid_metrics: int
    errors: List[ValidationError]
    warnings: List[ValidationError]
    compliance_score: float  # 0-100


class MetricsSchemaValidator:
    """指标模式验证器"""

    def __init__(self, schema_file: str = None):
        self.project_root = Path(__file__).parent.parent

        if schema_file:
            self.schema_file = Path(schema_file)
        else:
            self.schema_file = self.project_root / "config" / "federation_metrics_schema.json"

        self.schema = self._load_schema()
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []

    def _load_schema(self) -> Dict[str, Any]:
        """加载模式定义"""
        if not self.schema_file.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_file}")

        with open(self.schema_file, 'r') as f:
            return json.load(f)

    def validate_metric(self, metric_name: str, metric_info: Dict[str, Any]) -> bool:
        """验证单个指标"""
        is_valid = True

        # 验证命名
        if not self._validate_metric_name(metric_name):
            is_valid = False

        # 验证命名空间
        if not self._validate_namespace(metric_name):
            is_valid = False

        # 验证类型后缀
        if not self._validate_type_suffix(metric_name, metric_info.get('type', 'gauge')):
            is_valid = False

        # 验证标签
        labels = metric_info.get('labels', [])
        if not self._validate_labels(metric_name, labels):
            is_valid = False

        # 验证单位
        if not self._validate_units(metric_name, metric_info.get('unit')):
            is_valid = False

        # 验证基数限制
        if not self._validate_cardinality(metric_name, labels):
            is_valid = False

        return is_valid

    def _validate_metric_name(self, metric_name: str) -> bool:
        """验证指标名称"""
        rules = self.schema.get('validation_rules', {}).get('metric_name', {})

        # 检查模式
        pattern = rules.get('pattern', r'^[a-zA-Z_][a-zA-Z0-9_]*$')
        if not re.match(pattern, metric_name):
            self.errors.append(ValidationError(
                metric_name=metric_name,
                error_type="invalid_name",
                message=f"Metric name does not match pattern: {pattern}",
                severity="error"
            ))
            return False

        # 检查长度
        max_length = rules.get('max_length', 100)
        if len(metric_name) > max_length:
            self.errors.append(ValidationError(
                metric_name=metric_name,
                error_type="name_too_long",
                message=f"Metric name exceeds max length of {max_length}",
                severity="error"
            ))
            return False

        # 检查保留前缀
        reserved_prefixes = rules.get('reserved_prefixes', [])
        for prefix in reserved_prefixes:
            if metric_name.startswith(prefix):
                self.warnings.append(ValidationError(
                    metric_name=metric_name,
                    error_type="reserved_prefix",
                    message=f"Metric uses reserved prefix: {prefix}",
                    severity="warning"
                ))

        return True

    def _validate_namespace(self, metric_name: str) -> bool:
        """验证命名空间"""
        namespaces = self.schema.get('namespaces', {})

        # 检查是否有有效的命名空间前缀
        has_valid_namespace = False
        for ns_name, ns_config in namespaces.items():
            prefix = ns_config.get('prefix', '')
            if metric_name.startswith(prefix):
                has_valid_namespace = True
                break

        if not has_valid_namespace:
            self.warnings.append(ValidationError(
                metric_name=metric_name,
                error_type="missing_namespace",
                message="Metric does not have a standard namespace prefix",
                severity="warning"
            ))

        return True

    def _validate_type_suffix(self, metric_name: str, metric_type: str) -> bool:
        """验证类型后缀"""
        metric_types = self.schema.get('metric_types', {})

        if metric_type in metric_types:
            expected_suffix = metric_types[metric_type].get('suffix', '')

            if metric_type == "counter" and expected_suffix:
                if not metric_name.endswith(expected_suffix):
                    self.errors.append(ValidationError(
                        metric_name=metric_name,
                        error_type="missing_suffix",
                        message=f"Counter metric should end with '{expected_suffix}'",
                        severity="error"
                    ))
                    return False

        return True

    def _validate_labels(self, metric_name: str, labels: List[str]) -> bool:
        """验证标签"""
        standard_labels = self.schema.get('standard_labels', {})
        required_labels = set(standard_labels.get('required', {}).keys())

        # 检查必需标签
        missing_labels = required_labels - set(labels)
        if missing_labels:
            self.errors.append(ValidationError(
                metric_name=metric_name,
                error_type="missing_required_labels",
                message=f"Missing required labels: {', '.join(missing_labels)}",
                severity="error"
            ))
            return False

        # 检查敏感标签
        sensitive_labels = set(self.schema.get('security', {}).get('sensitive_labels', []))
        used_sensitive = sensitive_labels & set(labels)
        if used_sensitive:
            self.errors.append(ValidationError(
                metric_name=metric_name,
                error_type="sensitive_labels",
                message=f"Using sensitive labels: {', '.join(used_sensitive)}",
                severity="error"
            ))
            return False

        # 验证标签名称
        label_rules = self.schema.get('validation_rules', {}).get('label_name', {})
        pattern = label_rules.get('pattern', r'^[a-zA-Z_][a-zA-Z0-9_]*$')

        for label in labels:
            if not re.match(pattern, label):
                self.errors.append(ValidationError(
                    metric_name=metric_name,
                    error_type="invalid_label_name",
                    message=f"Label '{label}' does not match pattern: {pattern}",
                    severity="error"
                ))
                return False

        return True

    def _validate_units(self, metric_name: str, unit: Optional[str]) -> bool:
        """验证单位"""
        if not unit:
            return True

        units_config = self.schema.get('units', {})

        # 查找单位类别
        unit_valid = False
        for unit_type, unit_info in units_config.items():
            if unit in unit_info.get('allowed', []):
                unit_valid = True

                # 检查后缀
                suffix_map = unit_info.get('suffix_map', {})
                expected_suffix = suffix_map.get(unit, '')

                if expected_suffix and not metric_name.endswith(expected_suffix):
                    self.warnings.append(ValidationError(
                        metric_name=metric_name,
                        error_type="unit_suffix_mismatch",
                        message=f"Metric with unit '{unit}' should end with '{expected_suffix}'",
                        severity="warning"
                    ))
                break

        if not unit_valid:
            self.warnings.append(ValidationError(
                metric_name=metric_name,
                error_type="non_standard_unit",
                message=f"Unit '{unit}' is not in standard units",
                severity="warning"
            ))

        return True

    def _validate_cardinality(self, metric_name: str, labels: List[str]) -> bool:
        """验证基数限制"""
        cardinality_limits = self.schema.get('cardinality_limits', {})
        label_value_limits = cardinality_limits.get('label_value_limits', {})

        for label in labels:
            if label in label_value_limits:
                max_values = label_value_limits[label]
                # 这里只是警告，实际基数需要运行时检查
                self.warnings.append(ValidationError(
                    metric_name=metric_name,
                    error_type="cardinality_check",
                    message=f"Label '{label}' has cardinality limit of {max_values}",
                    severity="info"
                ))

        return True

    def validate_prometheus_config(self, config_file: str) -> ValidationReport:
        """验证Prometheus配置文件中的指标"""
        print(f"🔍 Validating Prometheus config: {config_file}")

        metrics_found = {}

        # 尝试从配置中提取指标定义
        with open(config_file, 'r') as f:
            content = yaml.safe_load(f)

        # 从recording rules提取
        for group in content.get('groups', []):
            for rule in group.get('rules', []):
                if 'record' in rule:
                    metric_name = rule['record']
                    metrics_found[metric_name] = {
                        'type': 'gauge',  # recording rules通常是gauge
                        'labels': list(rule.get('labels', {}).keys())
                    }

        # 验证每个指标
        for metric_name, metric_info in metrics_found.items():
            self.validate_metric(metric_name, metric_info)

        # 生成报告
        return self._generate_report(len(metrics_found))

    def validate_code_metrics(self) -> ValidationReport:
        """验证代码中定义的指标"""
        print("🔍 Scanning code for metric definitions...")

        metrics_found = {}

        # 扫描Python文件
        for py_file in self.project_root.rglob("*.py"):
            try:
                content = py_file.read_text()

                # Counter
                for match in re.finditer(r'Counter\([\'"](\w+)[\'"]', content):
                    metric_name = match.group(1)
                    metrics_found[metric_name] = {'type': 'counter'}

                # Gauge
                for match in re.finditer(r'Gauge\([\'"](\w+)[\'"]', content):
                    metric_name = match.group(1)
                    metrics_found[metric_name] = {'type': 'gauge'}

                # Histogram
                for match in re.finditer(r'Histogram\([\'"](\w+)[\'"]', content):
                    metric_name = match.group(1)
                    metrics_found[metric_name] = {'type': 'histogram'}

            except Exception:
                pass

        # 验证每个指标
        for metric_name, metric_info in metrics_found.items():
            self.validate_metric(metric_name, metric_info)

        # 生成报告
        return self._generate_report(len(metrics_found))

    def _generate_report(self, total_metrics: int) -> ValidationReport:
        """生成验证报告"""
        valid_metrics = total_metrics - len(self.errors)

        # 计算合规分数
        if total_metrics > 0:
            error_penalty = len(self.errors) * 10
            warning_penalty = len(self.warnings) * 2
            compliance_score = max(0, 100 - error_penalty - warning_penalty)
        else:
            compliance_score = 100.0

        return ValidationReport(
            timestamp=datetime.now(),
            total_metrics=total_metrics,
            valid_metrics=valid_metrics,
            errors=self.errors,
            warnings=self.warnings,
            compliance_score=compliance_score
        )

    def generate_markdown_report(self, report: ValidationReport) -> str:
        """生成Markdown格式报告"""
        lines = []
        lines.append("# Metrics Schema Validation Report")
        lines.append(f"\n**Generated**: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Schema Version**: {self.schema.get('version', 'unknown')}\n")

        # 摘要
        lines.append("## 📊 Summary\n")
        lines.append(f"- **Total Metrics**: {report.total_metrics}")
        lines.append(f"- **Valid Metrics**: {report.valid_metrics}")
        lines.append(f"- **Errors**: {len(report.errors)} ❌")
        lines.append(f"- **Warnings**: {len(report.warnings)} ⚠️")
        lines.append(f"- **Compliance Score**: {report.compliance_score:.1f}%\n")

        # 合规等级
        if report.compliance_score >= 90:
            lines.append("**Compliance Level**: ✅ EXCELLENT")
        elif report.compliance_score >= 75:
            lines.append("**Compliance Level**: 🟢 GOOD")
        elif report.compliance_score >= 60:
            lines.append("**Compliance Level**: 🟡 FAIR")
        else:
            lines.append("**Compliance Level**: 🔴 POOR")

        # 错误详情
        if report.errors:
            lines.append("\n## ❌ Errors\n")
            lines.append("The following errors must be fixed:\n")

            for error in report.errors:
                lines.append(f"### `{error.metric_name}`")
                lines.append(f"- **Type**: {error.error_type}")
                lines.append(f"- **Message**: {error.message}\n")

        # 警告详情
        if report.warnings:
            lines.append("\n## ⚠️ Warnings\n")
            lines.append("The following warnings should be reviewed:\n")

            for warning in report.warnings[:10]:  # 只显示前10个
                lines.append(f"- **{warning.metric_name}**: {warning.message}")

            if len(report.warnings) > 10:
                lines.append(f"\n... and {len(report.warnings) - 10} more warnings")

        # 建议
        lines.append("\n## 💡 Recommendations\n")

        if len(report.errors) > 0:
            lines.append("### Fix Critical Issues")
            lines.append("1. Add missing namespace prefixes")
            lines.append("2. Fix counter metrics to end with '_total'")
            lines.append("3. Add required labels (service, environment)")
            lines.append("4. Remove sensitive labels\n")

        if len(report.warnings) > 0:
            lines.append("### Improve Compliance")
            lines.append("1. Standardize unit suffixes")
            lines.append("2. Review cardinality limits")
            lines.append("3. Consider using recommended labels\n")

        lines.append("### Next Steps")
        lines.append("1. Run validation in CI/CD pipeline")
        lines.append("2. Update metrics to match schema")
        lines.append("3. Document any exceptions")

        return "\n".join(lines)


def generate_federation_config(schema_file: str = None) -> str:
    """生成联邦配置"""
    validator = MetricsSchemaValidator(schema_file)
    schema = validator.schema

    config = {
        'global': {
            'scrape_interval': '15s',
            'evaluation_interval': '15s',
            'external_labels': {
                'cluster': 'cad-ml-production',
                'federation': 'enabled'
            }
        },
        'scrape_configs': []
    }

    # 为每个命名空间生成scrape配置
    for ns_name, ns_config in schema.get('namespaces', {}).items():
        scrape_config = {
            'job_name': f"federate_{ns_name}",
            'honor_labels': True,
            'metrics_path': '/federate',
            'params': {
                'match[]': [
                    f'{{{ns_config["prefix"]}*}}'
                ]
            },
            'static_configs': [
                {
                    'targets': ['prometheus-federation.example.com:9090']
                }
            ]
        }
        config['scrape_configs'].append(scrape_config)

    return yaml.dump(config, default_flow_style=False)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate metrics against federation schema"
    )

    parser.add_argument(
        "--schema",
        help="Path to schema file",
        default=None
    )
    parser.add_argument(
        "--config",
        help="Prometheus config file to validate"
    )
    parser.add_argument(
        "--code",
        action="store_true",
        help="Scan code for metric definitions"
    )
    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="Generate federation config"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path"
    )

    args = parser.parse_args()

    validator = MetricsSchemaValidator(args.schema)

    if args.generate_config:
        config = generate_federation_config(args.schema)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(config)
            print(f"✅ Federation config saved to: {args.output}")
        else:
            print(config)
        return 0

    # 执行验证
    if args.config:
        report = validator.validate_prometheus_config(args.config)
    elif args.code:
        report = validator.validate_code_metrics()
    else:
        print("Please specify --config or --code")
        return 1

    # 生成报告
    output = validator.generate_markdown_report(report)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"✅ Report saved to: {args.output}")
    else:
        print(output)

    # 返回退出码
    if len(report.errors) > 0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())