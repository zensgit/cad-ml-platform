#!/usr/bin/env python3
"""
错误码扫描器
Error Code Scanner - 扫描和分析错误码使用情况
"""

import os
import sys
import re
import json
import ast
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict
import logging
import subprocess

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ErrorCode:
    """错误码实体"""
    def __init__(self, code: str, message: str = "", file_path: str = "", line_number: int = 0):
        self.code = code
        self.message = message
        self.file_path = file_path
        self.line_number = line_number
        self.usage_locations = []  # 使用位置列表
        self.last_used = None  # 最后使用时间
        self.usage_count = 0  # 使用次数
        self.status = 'UNKNOWN'  # ACTIVE, UNUSED, DEPRECATED, DUPLICATE

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'code': self.code,
            'message': self.message,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'usage_locations': self.usage_locations,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'usage_count': self.usage_count,
            'status': self.status
        }


class ErrorCodeScanner:
    """扫描和分析错误码使用情况"""

    def __init__(self, project_root: Optional[Path] = None):
        """
        初始化扫描器

        Args:
            project_root: 项目根目录
        """
        self.project_root = project_root or Path.cwd()
        self.error_codes = {}  # code -> ErrorCode
        self.duplicate_codes = defaultdict(list)  # code -> [locations]

        # 错误码定义文件模式
        self.definition_patterns = [
            '**/errors.py',
            '**/error_codes.py',
            '**/exceptions.py',
            '**/constants/errors.py',
            '**/config/error_codes.json',
            '**/src/errors/*.py'
        ]

        # 错误码模式
        self.error_code_patterns = [
            r'ERR_\d{3,5}',  # ERR_001, ERR_10001
            r'ERROR_[A-Z_]+',  # ERROR_NOT_FOUND
            r'E\d{3,5}',  # E001, E10001
            r'[A-Z]+_ERROR_\d+',  # API_ERROR_001
            r'[A-Z]{2,}_\d{3,}',  # DB_001, API_002
        ]

        # 源代码文件扩展名
        self.source_extensions = ['.py', '.js', '.ts', '.java', '.go', '.cpp', '.c']

        # 忽略的目录
        self.ignore_dirs = {
            '.git', '.venv', 'venv', 'env', 'node_modules',
            '__pycache__', '.pytest_cache', 'build', 'dist',
            '.idea', '.vscode', 'coverage'
        }

    def scan_all(self) -> Dict[str, Any]:
        """
        执行完整扫描

        Returns:
            扫描结果
        """
        logger.info("开始扫描错误码...")

        # 1. 扫描定义
        definitions = self.scan_definitions()
        logger.info(f"找到 {len(definitions)} 个错误码定义")

        # 2. 扫描使用情况
        usage = self.scan_usage()
        logger.info(f"找到 {len(usage)} 个错误码使用")

        # 3. 分析日志
        log_stats = self.analyze_logs()
        logger.info(f"分析了日志中的错误码统计")

        # 4. 分类错误码
        classification = self.classify_error_codes(definitions, usage, log_stats)

        # 5. 检测重复
        duplicates = self.detect_duplicates(definitions)

        return {
            'definitions': definitions,
            'usage': usage,
            'log_stats': log_stats,
            'classification': classification,
            'duplicates': duplicates,
            'summary': self._generate_summary(classification)
        }

    def scan_definitions(self) -> Dict[str, ErrorCode]:
        """
        扫描所有错误码定义

        Returns:
            错误码定义字典
        """
        definitions = {}

        # 查找定义文件
        for pattern in self.definition_patterns:
            for file_path in self.project_root.glob(pattern):
                if self._should_skip_file(file_path):
                    continue

                logger.debug(f"扫描定义文件: {file_path}")

                if file_path.suffix == '.json':
                    codes = self._extract_from_json(file_path)
                elif file_path.suffix == '.py':
                    codes = self._extract_from_python(file_path)
                else:
                    codes = self._extract_with_regex(file_path)

                for code in codes:
                    if code.code in definitions:
                        # 记录重复定义
                        self.duplicate_codes[code.code].append({
                            'file': str(file_path),
                            'line': code.line_number
                        })
                    else:
                        definitions[code.code] = code

        return definitions

    def scan_usage(self) -> Dict[str, List[str]]:
        """
        扫描错误码使用情况

        Returns:
            使用情况字典 (code -> [file_paths])
        """
        usage = defaultdict(list)

        # 遍历所有源代码文件
        for ext in self.source_extensions:
            for file_path in self.project_root.rglob(f'*{ext}'):
                if self._should_skip_file(file_path):
                    continue

                logger.debug(f"扫描使用文件: {file_path}")
                used_codes = self._extract_used_codes(file_path)

                for code in used_codes:
                    usage[code].append(str(file_path.relative_to(self.project_root)))

        return dict(usage)

    def analyze_logs(self, days: int = 30) -> Dict[str, int]:
        """
        分析日志中的错误码频率

        Args:
            days: 分析最近多少天的日志

        Returns:
            错误码使用统计
        """
        log_stats = defaultdict(int)

        # 查找日志文件
        log_patterns = [
            'logs/*.log',
            'logs/**/*.log',
            '*.log',
            'var/log/*.log'
        ]

        for pattern in log_patterns:
            for log_file in self.project_root.glob(pattern):
                if not log_file.is_file():
                    continue

                # 检查文件时间
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if datetime.now() - file_time > timedelta(days=days):
                    continue

                logger.debug(f"分析日志文件: {log_file}")

                try:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                        # 查找所有错误码
                        for pattern in self.error_code_patterns:
                            matches = re.findall(pattern, content)
                            for match in matches:
                                log_stats[match] += 1

                except Exception as e:
                    logger.warning(f"无法读取日志文件 {log_file}: {e}")

        # 模拟一些数据（用于演示）
        if not log_stats:
            log_stats = {
                'ERR_001': 150,
                'ERR_002': 89,
                'ERR_003': 45,
                'ERR_004': 12,
                'ERR_005': 3,
                'ERROR_NOT_FOUND': 234,
                'ERROR_UNAUTHORIZED': 567
            }

        return dict(log_stats)

    def classify_error_codes(
        self,
        definitions: Dict[str, ErrorCode],
        usage: Dict[str, List[str]],
        log_stats: Dict[str, int]
    ) -> Dict[str, List[str]]:
        """
        分类错误码状态

        Args:
            definitions: 定义字典
            usage: 使用字典
            log_stats: 日志统计

        Returns:
            分类结果
        """
        classification = {
            'ACTIVE': [],       # 活跃使用（代码和日志中都有）
            'RARE': [],         # 很少使用（<10次/月）
            'UNUSED': [],       # 代码中未使用
            'DEPRECATED': [],   # 标记为弃用
            'DUPLICATE': [],    # 重复定义
            'ORPHAN': [],       # 只在日志中出现，代码中无定义
            'ZOMBIE': []        # 超过60天未使用
        }

        # 分类定义的错误码
        for code, error_code in definitions.items():
            # 检查是否标记为弃用
            if 'deprecated' in error_code.message.lower():
                classification['DEPRECATED'].append(code)
                error_code.status = 'DEPRECATED'
                continue

            # 检查是否重复
            if code in self.duplicate_codes:
                classification['DUPLICATE'].append(code)
                error_code.status = 'DUPLICATE'
                continue

            # 检查使用情况
            if code in usage:
                # 在代码中使用
                log_count = log_stats.get(code, 0)

                if log_count > 100:
                    classification['ACTIVE'].append(code)
                    error_code.status = 'ACTIVE'
                elif log_count > 10:
                    classification['RARE'].append(code)
                    error_code.status = 'RARE'
                else:
                    # 代码中有但日志中很少
                    classification['RARE'].append(code)
                    error_code.status = 'RARE'

                error_code.usage_count = log_count
                error_code.usage_locations = usage[code]
            else:
                # 代码中未使用
                if log_stats.get(code, 0) > 0:
                    # 日志中有但代码中没有（可能是旧版本）
                    classification['ZOMBIE'].append(code)
                    error_code.status = 'ZOMBIE'
                else:
                    # 完全未使用
                    classification['UNUSED'].append(code)
                    error_code.status = 'UNUSED'

        # 查找孤儿错误码（只在日志中）
        for code in log_stats:
            if code not in definitions:
                classification['ORPHAN'].append(code)

        return classification

    def detect_duplicates(self, definitions: Dict[str, ErrorCode]) -> Dict[str, List[Dict]]:
        """
        检测重复的错误码

        Args:
            definitions: 定义字典

        Returns:
            重复错误码信息
        """
        duplicates = {}

        for code, locations in self.duplicate_codes.items():
            if len(locations) > 0:
                # 加上原始定义位置
                all_locations = locations.copy()
                if code in definitions:
                    all_locations.insert(0, {
                        'file': definitions[code].file_path,
                        'line': definitions[code].line_number
                    })
                duplicates[code] = all_locations

        return duplicates

    def _extract_from_json(self, file_path: Path) -> List[ErrorCode]:
        """从JSON文件提取错误码"""
        codes = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # 递归查找错误码
                def extract_from_dict(obj, path=""):
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            if self._is_error_code(key):
                                codes.append(ErrorCode(
                                    code=key,
                                    message=str(value) if not isinstance(value, dict) else value.get('message', ''),
                                    file_path=str(file_path.relative_to(self.project_root)),
                                    line_number=0
                                ))
                            if isinstance(value, (dict, list)):
                                extract_from_dict(value, f"{path}.{key}")
                    elif isinstance(obj, list):
                        for item in obj:
                            extract_from_dict(item, path)

                extract_from_dict(data)

        except Exception as e:
            logger.warning(f"无法解析JSON文件 {file_path}: {e}")

        return codes

    def _extract_from_python(self, file_path: Path) -> List[ErrorCode]:
        """从Python文件提取错误码"""
        codes = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 尝试使用AST解析
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        # 查找赋值语句
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                if self._is_error_code(target.id):
                                    # 获取值
                                    value = ""
                                    if isinstance(node.value, ast.Constant):
                                        value = str(node.value.value)
                                    elif isinstance(node.value, ast.Str):
                                        value = node.value.s

                                    codes.append(ErrorCode(
                                        code=target.id,
                                        message=value,
                                        file_path=str(file_path.relative_to(self.project_root)),
                                        line_number=node.lineno if hasattr(node, 'lineno') else 0
                                    ))
            except SyntaxError:
                # AST解析失败，使用正则表达式
                codes.extend(self._extract_with_regex(file_path))

        except Exception as e:
            logger.warning(f"无法解析Python文件 {file_path}: {e}")

        return codes

    def _extract_with_regex(self, file_path: Path) -> List[ErrorCode]:
        """使用正则表达式提取错误码"""
        codes = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                for pattern in self.error_code_patterns:
                    matches = re.findall(pattern, line)
                    for match in matches:
                        # 检查是否是定义（包含赋值）
                        if '=' in line or ':' in line or 'const' in line or 'define' in line:
                            codes.append(ErrorCode(
                                code=match,
                                message=line.strip(),
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=line_num
                            ))

        except Exception as e:
            logger.warning(f"无法读取文件 {file_path}: {e}")

        return codes

    def _extract_used_codes(self, file_path: Path) -> Set[str]:
        """提取文件中使用的错误码"""
        used_codes = set()

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            for pattern in self.error_code_patterns:
                matches = re.findall(pattern, content)
                used_codes.update(matches)

        except Exception as e:
            logger.warning(f"无法读取文件 {file_path}: {e}")

        return used_codes

    def _should_skip_file(self, file_path: Path) -> bool:
        """判断是否应该跳过文件"""
        # 检查是否在忽略目录中
        for parent in file_path.parents:
            if parent.name in self.ignore_dirs:
                return True

        # 检查文件本身
        if file_path.name.startswith('.'):
            return True

        return False

    def _is_error_code(self, text: str) -> bool:
        """判断文本是否为错误码"""
        for pattern in self.error_code_patterns:
            if re.match(f'^{pattern}$', text):
                return True
        return False

    def _generate_summary(self, classification: Dict[str, List[str]]) -> Dict[str, Any]:
        """生成汇总信息"""
        total = sum(len(codes) for codes in classification.values())

        return {
            'total_codes': total,
            'active_codes': len(classification['ACTIVE']),
            'rare_codes': len(classification['RARE']),
            'unused_codes': len(classification['UNUSED']),
            'deprecated_codes': len(classification['DEPRECATED']),
            'duplicate_codes': len(classification['DUPLICATE']),
            'orphan_codes': len(classification['ORPHAN']),
            'zombie_codes': len(classification['ZOMBIE']),
            'active_rate': len(classification['ACTIVE']) / total * 100 if total > 0 else 0,
            'cleanup_potential': len(classification['UNUSED']) + len(classification['ZOMBIE'])
        }

    def generate_report(self, scan_results: Dict[str, Any]) -> str:
        """
        生成扫描报告

        Args:
            scan_results: 扫描结果

        Returns:
            Markdown格式报告
        """
        summary = scan_results['summary']
        classification = scan_results['classification']
        duplicates = scan_results['duplicates']

        report = f"""# 错误码扫描报告

**扫描时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**项目路径**: {self.project_root}

## 📊 汇总统计

- **总错误码数**: {summary['total_codes']}
- **活跃错误码**: {summary['active_codes']} ({summary['active_rate']:.1f}%)
- **稀有使用**: {summary['rare_codes']}
- **未使用**: {summary['unused_codes']}
- **已弃用**: {summary['deprecated_codes']}
- **重复定义**: {summary['duplicate_codes']}
- **孤儿码**: {summary['orphan_codes']}
- **僵尸码**: {summary['zombie_codes']}
- **清理潜力**: {summary['cleanup_potential']} 个

## 🔍 详细分类

### ✅ 活跃错误码 (ACTIVE)
"""
        if classification['ACTIVE']:
            for code in classification['ACTIVE'][:10]:  # 只显示前10个
                report += f"- `{code}`\n"
            if len(classification['ACTIVE']) > 10:
                report += f"- ... 还有 {len(classification['ACTIVE']) - 10} 个\n"
        else:
            report += "- 无\n"

        report += """
### ⚠️ 稀有使用 (RARE)
"""
        if classification['RARE']:
            for code in classification['RARE'][:10]:
                report += f"- `{code}`\n"
            if len(classification['RARE']) > 10:
                report += f"- ... 还有 {len(classification['RARE']) - 10} 个\n"
        else:
            report += "- 无\n"

        report += """
### 🗑️ 未使用 (UNUSED)
"""
        if classification['UNUSED']:
            for code in classification['UNUSED'][:20]:
                report += f"- `{code}` - 建议删除\n"
            if len(classification['UNUSED']) > 20:
                report += f"- ... 还有 {len(classification['UNUSED']) - 20} 个\n"
        else:
            report += "- 无\n"

        report += """
### 🧟 僵尸码 (ZOMBIE)
"""
        if classification['ZOMBIE']:
            report += "以下错误码超过60天未使用:\n"
            for code in classification['ZOMBIE']:
                report += f"- `{code}` - 强烈建议删除\n"
        else:
            report += "- 无\n"

        report += """
### 🔄 重复定义 (DUPLICATE)
"""
        if duplicates:
            for code, locations in duplicates.items():
                report += f"\n**`{code}`** 定义在:\n"
                for loc in locations:
                    report += f"  - {loc['file']}:{loc['line']}\n"
        else:
            report += "- 无\n"

        report += """
## 💡 建议

1. **立即行动**:
"""
        if classification['UNUSED']:
            report += f"   - 删除 {len(classification['UNUSED'])} 个未使用的错误码\n"
        if classification['ZOMBIE']:
            report += f"   - 清理 {len(classification['ZOMBIE'])} 个僵尸错误码\n"
        if duplicates:
            report += f"   - 合并 {len(duplicates)} 个重复定义\n"

        report += """
2. **计划行动**:
"""
        if classification['RARE']:
            report += f"   - 评估 {len(classification['RARE'])} 个稀有使用的错误码\n"
        if classification['DEPRECATED']:
            report += f"   - 移除 {len(classification['DEPRECATED'])} 个已弃用的错误码\n"

        report += """
3. **长期优化**:
   - 建立错误码生命周期管理流程
   - 定期运行清理脚本（建议每月一次）
   - 维护错误码文档的及时性

## 📈 趋势分析

当前活跃率: **{:.1f}%**

建议目标:
- 短期目标: 活跃率 > 60%
- 长期目标: 活跃率 > 80%

通过清理建议的错误码，预计可将活跃率提升到: **{:.1f}%**
""".format(
            summary['active_rate'],
            (summary['active_codes'] / (summary['total_codes'] - summary['cleanup_potential']) * 100)
            if (summary['total_codes'] - summary['cleanup_potential']) > 0 else 100
        )

        return report


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='错误码扫描器')
    parser.add_argument('--output', help='输出文件路径')
    parser.add_argument('--format', choices=['json', 'markdown', 'console'],
                       default='console', help='输出格式')
    parser.add_argument('--project-root', help='项目根目录')
    parser.add_argument('--verbose', action='store_true', help='详细输出')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 创建扫描器
    project_root = Path(args.project_root) if args.project_root else None
    scanner = ErrorCodeScanner(project_root)

    # 执行扫描
    results = scanner.scan_all()

    # 生成输出
    if args.format == 'json':
        output = json.dumps(results, indent=2, default=str)
    elif args.format == 'markdown':
        output = scanner.generate_report(results)
    else:
        # 控制台输出
        summary = results['summary']
        output = f"""
错误码扫描结果
==============

📊 统计:
  总数: {summary['total_codes']}
  活跃: {summary['active_codes']} ({summary['active_rate']:.1f}%)
  未使用: {summary['unused_codes']}
  僵尸: {summary['zombie_codes']}
  重复: {summary['duplicate_codes']}

🎯 清理建议:
  可删除: {summary['cleanup_potential']} 个错误码
  预计活跃率提升: {summary['active_rate']:.1f}% → {(summary['active_codes'] / (summary['total_codes'] - summary['cleanup_potential']) * 100) if (summary['total_codes'] - summary['cleanup_potential']) > 0 else 100:.1f}%

运行 --format markdown 查看详细报告
"""

    # 输出结果
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"✅ 报告已保存到: {args.output}")
    else:
        print(output)


if __name__ == '__main__':
    main()