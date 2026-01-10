#!/usr/bin/env python3
"""
Validate Prometheus recording rules using promtool.

This script validates the syntax and semantics of Prometheus recording rules
before deployment. It can use either a local promtool installation or Docker.

Exit Codes:
    0: All rules valid
    1: Validation failed
    2: Missing dependencies (promtool not found)
    3: Rules file not found
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ANSI color codes
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BOLD = '\033[1m'
RESET = '\033[0m'


class PromtoolValidator:
    """Validator for Prometheus recording rules."""

    def __init__(self, rules_path: str = "docs/prometheus/recording_rules.yml", json_mode: bool = False):
        """Initialize validator with rules file path."""
        self.rules_path = Path(rules_path)
        self.json_mode = json_mode
        self.promtool_cmd = self._find_promtool(silent=json_mode)
        self.validation_results: Dict[str, any] = {}

    def _find_promtool(self, silent: bool = False) -> Optional[List[str]]:
        """Find promtool command (local install or Docker)."""
        # Try local installation first
        try:
            result = subprocess.run(
                ["promtool", "version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                if not silent:
                    print(f"{GREEN}✓{RESET} Found local promtool installation")
                return ["promtool"]
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        # Try Docker
        try:
            result = subprocess.run(
                ["docker", "version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                if not silent:
                    print(f"{GREEN}✓{RESET} Found Docker, will use prom/prometheus image")
                return [
                    "docker",
                    "run",
                    "--rm",
                    "--entrypoint",
                    "promtool",
                    "-v",
                    f"{os.getcwd()}:/workspace:ro",
                    "prom/prometheus:latest",
                ]
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        if not silent:
            print(f"{RED}✗{RESET} Neither promtool nor Docker found")
        return None

    def validate_syntax(self) -> Tuple[bool, str]:
        """Validate rules file syntax."""
        if not self.rules_path.exists():
            return False, f"Rules file not found: {self.rules_path}"

        if not self.promtool_cmd:
            return False, "promtool not available"

        # Adjust path for Docker mounting
        rules_file = str(self.rules_path)
        if "docker" in self.promtool_cmd:
            rules_file = f"/workspace/{self.rules_path}"

        cmd = self.promtool_cmd + ["check", "rules", rules_file]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                # Parse success output
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if "SUCCESS" in line:
                        # Extract group and rule counts
                        import re
                        match = re.search(r'(\d+) groups?, (\d+) rules?', line)
                        if match:
                            self.validation_results['groups'] = int(match.group(1))
                            self.validation_results['rules'] = int(match.group(2))

                return True, result.stdout
            else:
                return False, result.stderr or result.stdout

        except subprocess.TimeoutExpired:
            return False, "Validation timeout (30s)"
        except Exception as e:
            return False, f"Validation error: {e}"

    def validate_metrics_used(self) -> Dict[str, List[str]]:
        """Extract and validate metrics referenced in rules."""
        metrics_used = set()

        if not self.rules_path.exists():
            return {"error": ["Rules file not found"]}

        with open(self.rules_path, 'r') as f:
            content = f.read()

        # Simple regex to find metric names in expressions
        import re
        import yaml

        try:
            rules_data = yaml.safe_load(content)

            for group in rules_data.get('groups', []):
                for rule in group.get('rules', []):
                    expr = rule.get('expr', '')
                    if expr is None:
                        continue
                    if not isinstance(expr, str):
                        expr = str(expr)
                    # Extract metric names (simplified - won't catch all cases)
                    # Matches: metric_name{...} or metric_name[...]
                    pattern = r'\b([a-z_][a-z0-9_]*(?:_total|_bucket|_count|_sum)?)\s*[{\[]'
                    found_metrics = re.findall(pattern, expr, re.IGNORECASE)
                    metrics_used.update(found_metrics)

            return {
                "metrics_referenced": sorted(list(metrics_used)),
                "count": len(metrics_used)
            }

        except yaml.YAMLError as e:
            return {"error": [f"YAML parse error: {e}"]}

    def validate_recording_rule_names(self) -> Dict[str, any]:
        """Validate recording rule naming conventions."""
        issues = []
        valid_rules = []

        if not self.rules_path.exists():
            return {"error": "Rules file not found"}

        with open(self.rules_path, 'r') as f:
            import yaml
            try:
                rules_data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                return {"error": f"YAML error: {e}"}

        for group in rules_data.get('groups', []):
            group_name = group.get('name', 'unnamed')

            for rule in group.get('rules', []):
                record_name = rule.get('record', '')
                alert_name = rule.get('alert', '')

                # Skip alerting rules; naming rules apply only to recording rules
                if alert_name and not record_name:
                    continue

                # Check naming conventions
                if not record_name:
                    issues.append(f"Missing 'record' field in group '{group_name}'")
                    continue

                # Recording rules should use snake_case or recording rule convention (prefix:name)
                if ":" in record_name:
                    if not re.match(r'^[a-z][a-z0-9_]*:[a-z][a-z0-9_]*$', record_name):
                        issues.append(
                            f"Invalid rule name '{record_name}' (should be prefix:name with snake_case)"
                        )
                else:
                    if not re.match(r'^[a-z][a-z0-9_]*$', record_name):
                        issues.append(f"Invalid rule name '{record_name}' (should be snake_case)")

                # Check for meaningful prefixes
                known_prefixes = [
                    "ocr_",
                    "vision_",
                    "platform_",
                    "provider_",
                    "slo_",
                    "error_",
                    "model_",
                    "circuit_",
                    "dedup2d_",
                    "cad_ml:",
                    "cad_ml_",
                ]
                if not any(record_name.startswith(p) for p in known_prefixes):
                    issues.append(f"Rule '{record_name}' lacks standard prefix")

                valid_rules.append(record_name)

        return {
            "valid_rules": valid_rules,
            "issues": issues,
            "total_rules": len(valid_rules),
            "total_issues": len(issues)
        }

    def validate_expressions(self) -> Dict[str, any]:
        """Validate that expressions are syntactically correct."""
        expression_errors = []
        valid_expressions = 0

        if not self.rules_path.exists():
            return {"error": "Rules file not found"}

        with open(self.rules_path, 'r') as f:
            import yaml
            try:
                rules_data = yaml.safe_load(f)
            except yaml.YAMLError:
                return {"error": "YAML parse error"}

        for group in rules_data.get('groups', []):
            for rule in group.get('rules', []):
                expr = rule.get('expr', '')
                record = rule.get('record') or rule.get('alert') or 'unnamed'
                if expr is None:
                    expression_errors.append(f"Empty expression for rule '{record}'")
                    continue
                if not isinstance(expr, str):
                    expr = str(expr)

                # Basic expression validation
                if not expr:
                    expression_errors.append(f"Empty expression for rule '{record}'")
                    continue

                # Check for balanced parentheses
                if expr.count('(') != expr.count(')'):
                    expression_errors.append(f"Unbalanced parentheses in '{record}'")

                # Check for balanced brackets
                if expr.count('[') != expr.count(']'):
                    expression_errors.append(f"Unbalanced brackets in '{record}'")

                # Check for balanced braces
                if expr.count('{') != expr.count('}'):
                    expression_errors.append(f"Unbalanced braces in '{record}'")

                valid_expressions += 1

        return {
            "valid_expressions": valid_expressions,
            "errors": expression_errors,
            "error_count": len(expression_errors)
        }

    def generate_test_queries(self) -> List[str]:
        """Generate test PromQL queries for validation."""
        queries = []

        if not self.rules_path.exists():
            return queries

        with open(self.rules_path, 'r') as f:
            import yaml
            try:
                rules_data = yaml.safe_load(f)
            except yaml.YAMLError:
                return queries

        for group in rules_data.get('groups', []):
            for rule in group.get('rules', []):
                record = rule.get('record', '')
                if record:
                    # Generate test queries
                    queries.append(f"{record}")
                    queries.append(f"rate({record}[5m])")
                    queries.append(f"increase({record}[1h])")

        return queries

    def print_report(self, json_output: bool = False):
        """Print validation report."""
        if json_output:
            # JSON output for CI parsing
            output = {
                "validation_passed": self.validation_results.get('success', False),
                "groups": self.validation_results.get('groups', 0),
                "rules": self.validation_results.get('rules', 0),
                "metrics_used": self.validation_results.get('metrics_used', {}),
                "naming_validation": self.validation_results.get('naming', {}),
                "expression_validation": self.validation_results.get('expressions', {}),
                "test_queries": self.validation_results.get('test_queries', [])
            }
            print(json.dumps(output, indent=2))
        else:
            # Human-readable output
            print(f"\n{BOLD}Prometheus Recording Rules Validation Report{RESET}")
            print("=" * 60)

            # Syntax validation
            if self.validation_results.get('success'):
                print(f"{GREEN}✓ Syntax Validation: PASSED{RESET}")
                print(f"  Groups: {self.validation_results.get('groups', 0)}")
                print(f"  Rules: {self.validation_results.get('rules', 0)}")
            else:
                print(f"{RED}✗ Syntax Validation: FAILED{RESET}")
                print(f"  Error: {self.validation_results.get('error', 'Unknown error')}")

            # Metrics used
            metrics_used = self.validation_results.get('metrics_used', {})
            if 'error' not in metrics_used:
                print(f"\n{BOLD}Metrics Referenced:{RESET}")
                print(f"  Total unique metrics: {metrics_used.get('count', 0)}")
                if metrics_used.get('metrics_referenced'):
                    for metric in metrics_used['metrics_referenced'][:10]:
                        print(f"    - {metric}")
                    if len(metrics_used['metrics_referenced']) > 10:
                        print(f"    ... and {len(metrics_used['metrics_referenced']) - 10} more")

            # Naming validation
            naming = self.validation_results.get('naming', {})
            if 'error' not in naming:
                print(f"\n{BOLD}Naming Convention:{RESET}")
                print(f"  Valid rules: {naming.get('total_rules', 0)}")
                if naming.get('issues'):
                    print(f"  {YELLOW}Issues found: {naming.get('total_issues', 0)}{RESET}")
                    for issue in naming['issues'][:5]:
                        print(f"    ⚠ {issue}")
                    if len(naming['issues']) > 5:
                        print(f"    ... and {len(naming['issues']) - 5} more issues")
                else:
                    print(f"  {GREEN}All naming conventions followed{RESET}")

            # Expression validation
            expressions = self.validation_results.get('expressions', {})
            if 'error' not in expressions:
                print(f"\n{BOLD}Expression Validation:{RESET}")
                print(f"  Valid expressions: {expressions.get('valid_expressions', 0)}")
                if expressions.get('errors'):
                    print(f"  {YELLOW}Errors found: {expressions.get('error_count', 0)}{RESET}")
                    for error in expressions['errors'][:5]:
                        print(f"    ⚠ {error}")
                else:
                    print(f"  {GREEN}All expressions valid{RESET}")

            print("\n" + "=" * 60)

            # Summary
            if self.validation_results.get('success'):
                print(f"{GREEN}{BOLD}✓ Validation PASSED{RESET}")
            else:
                print(f"{RED}{BOLD}✗ Validation FAILED{RESET}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Validate Prometheus recording rules'
    )
    parser.add_argument(
        '--rules-file',
        default='docs/prometheus/recording_rules.yml',
        help='Path to recording rules file'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    parser.add_argument(
        '--skip-promtool',
        action='store_true',
        help='Skip promtool validation (only do static checks)'
    )

    args = parser.parse_args()

    # Create validator
    validator = PromtoolValidator(args.rules_file, json_mode=args.json)

    # Check if rules file exists
    if not validator.rules_path.exists():
        if args.json:
            print(json.dumps({"error": f"Rules file not found: {args.rules_file}"}))
        else:
            print(f"{RED}Error: Rules file not found: {args.rules_file}{RESET}")
        sys.exit(3)

    # Run validations
    if not args.skip_promtool:
        success, message = validator.validate_syntax()
        validator.validation_results['success'] = success
        if not success:
            validator.validation_results['error'] = message
            if not validator.promtool_cmd:
                # Missing dependency
                if not args.json:
                    print(f"{RED}Error: promtool not found. Install it or use Docker.{RESET}")
                    print(f"  Option 1: Install promtool from https://prometheus.io/download/")
                    print(f"  Option 2: Install Docker and the script will use prom/prometheus image")
                sys.exit(2)
    else:
        validator.validation_results['success'] = True
        if not args.json:
            print(f"{YELLOW}Skipping promtool validation{RESET}")

    # Always run static validations
    validator.validation_results['metrics_used'] = validator.validate_metrics_used()
    validator.validation_results['naming'] = validator.validate_recording_rule_names()
    validator.validation_results['expressions'] = validator.validate_expressions()
    validator.validation_results['test_queries'] = validator.generate_test_queries()

    # Print report
    validator.print_report(json_output=args.json)

    # Exit with appropriate code
    if validator.validation_results.get('success'):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
