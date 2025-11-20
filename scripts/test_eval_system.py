#!/usr/bin/env python3
"""
Comprehensive test suite for the evaluation system.

Tests all components: evaluation, reporting, integrity, validation.

Usage:
    python3 scripts/test_eval_system.py [--quick] [--verbose]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


class Colors:
    """Terminal colors for output."""
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


class TestRunner:
    """Run and report on evaluation system tests."""

    def __init__(self, verbose: bool = False, quick: bool = False):
        self.verbose = verbose
        self.quick = quick
        self.results = []
        self.start_time = time.time()

    def run_command(self, cmd: List[str], description: str) -> Tuple[bool, str]:
        """Run a command and capture output."""
        if self.verbose:
            print(f"{Colors.BLUE}Running: {' '.join(cmd)}{Colors.RESET}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            success = result.returncode == 0
            output = result.stdout + result.stderr
            return success, output
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)

    def test_config_integrity(self) -> bool:
        """Test configuration file integrity."""
        print(f"\n{Colors.BOLD}Testing Configuration Integrity...{Colors.RESET}")

        # Check config file exists
        config_path = Path("config/eval_frontend.json")
        if not config_path.exists():
            self.results.append(("Config file", False, "Not found"))
            return False

        # Validate JSON
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            self.results.append(("Config JSON", True, "Valid"))
        except json.JSONDecodeError as e:
            self.results.append(("Config JSON", False, str(e)))
            return False

        # Check required fields
        required = ["chartjs", "schema_version", "validation", "retention_policy"]
        for field in required:
            if field in config:
                self.results.append((f"Config.{field}", True, "Present"))
            else:
                self.results.append((f"Config.{field}", False, "Missing"))

        return all(field in config for field in required)

    def test_file_integrity(self) -> bool:
        """Test file integrity checking."""
        print(f"\n{Colors.BOLD}Testing File Integrity Check...{Colors.RESET}")

        success, output = self.run_command(
            ["python3", "scripts/check_integrity.py", "--verbose"],
            "File integrity check"
        )

        # In warn mode, it should always return 0
        if "WARNING" in output or "PASS" in output:
            self.results.append(("Integrity check", True, "Executed"))
            return True
        else:
            self.results.append(("Integrity check", False, "Failed"))
            return False

    def test_schema_validation(self) -> bool:
        """Test JSON schema validation."""
        print(f"\n{Colors.BOLD}Testing Schema Validation...{Colors.RESET}")

        # Check if jsonschema is available
        try:
            import jsonschema
            has_jsonschema = True
        except ImportError:
            has_jsonschema = False
            self.results.append(("jsonschema module", False, "Not installed"))

        # Check schema file
        schema_path = Path("docs/eval_history.schema.json")
        if not schema_path.exists():
            self.results.append(("Schema file", False, "Not found"))
            return False

        # Validate schema structure
        try:
            with open(schema_path, "r") as f:
                schema = json.load(f)

            if "$schema" in schema and "properties" in schema:
                self.results.append(("Schema structure", True, "Valid"))
            else:
                self.results.append(("Schema structure", False, "Invalid"))
                return False
        except Exception as e:
            self.results.append(("Schema structure", False, str(e)))
            return False

        # Run validation command
        success, output = self.run_command(
            ["python3", "scripts/validate_eval_history.py", "--dir", "reports/eval_history"],
            "History validation"
        )

        self.results.append(("History validation", success, "Completed" if success else "Failed"))
        return True

    def test_evaluation_pipeline(self) -> bool:
        """Test the evaluation pipeline (quick mode skips actual evaluation)."""
        print(f"\n{Colors.BOLD}Testing Evaluation Pipeline...{Colors.RESET}")

        if self.quick:
            print("  Skipping actual evaluation (quick mode)")
            self.results.append(("Evaluation pipeline", True, "Skipped (quick mode)"))
            return True

        # Run combined evaluation
        success, output = self.run_command(
            ["python3", "scripts/evaluate_vision_ocr_combined.py"],
            "Combined evaluation"
        )

        if success and "Combined:" in output:
            # Extract score
            for line in output.split('\n'):
                if "Combined:" in line:
                    score = line.split(":")[-1].strip()
                    self.results.append(("Combined evaluation", True, f"Score: {score}"))
                    break
        else:
            self.results.append(("Combined evaluation", False, "Failed"))

        return success

    def test_report_generation(self) -> bool:
        """Test report generation."""
        print(f"\n{Colors.BOLD}Testing Report Generation...{Colors.RESET}")

        # Test basic report
        success1, _ = self.run_command(
            ["python3", "scripts/generate_eval_report.py"],
            "Basic report generation"
        )
        self.results.append(("Basic report", success1, "Generated" if success1 else "Failed"))

        # Test v2 report
        success2, _ = self.run_command(
            ["python3", "scripts/generate_eval_report_v2.py", "--use-cdn"],
            "Enhanced report generation"
        )
        self.results.append(("Enhanced report", success2, "Generated" if success2 else "Failed"))

        return success1 or success2

    def test_retention_policy(self) -> bool:
        """Test data retention policy."""
        print(f"\n{Colors.BOLD}Testing Retention Policy...{Colors.RESET}")

        success, output = self.run_command(
            ["python3", "scripts/manage_eval_retention.py", "--dry-run"],
            "Retention policy check"
        )

        if success and ("files found" in output.lower() or "found" in output.lower()):
            self.results.append(("Retention policy", True, "Checked"))
        else:
            self.results.append(("Retention policy", False, "Failed"))

        return success

    def test_version_monitoring(self) -> bool:
        """Test version monitoring."""
        print(f"\n{Colors.BOLD}Testing Version Monitoring...{Colors.RESET}")

        success, output = self.run_command(
            ["python3", "scripts/check_chartjs_updates.py"],
            "Version check"
        )

        if "Current version:" in output or "disabled in config" in output:
            self.results.append(("Version monitoring", True, "Functional"))
            return True
        else:
            self.results.append(("Version monitoring", False, "Failed"))
            return False

    def test_makefile_targets(self) -> bool:
        """Test key Makefile targets."""
        print(f"\n{Colors.BOLD}Testing Makefile Targets...{Colors.RESET}")

        targets = [
            "eval-validate",
            "integrity-check",
            "eval-retention",
            "health-check"
        ]

        all_success = True
        for target in targets:
            success, _ = self.run_command(
                ["make", target],
                f"Makefile target: {target}"
            )
            self.results.append((f"make {target}", success, "OK" if success else "Failed"))
            all_success = all_success and success

        return all_success

    def generate_report(self) -> None:
        """Generate test report."""
        elapsed = time.time() - self.start_time

        print(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}EVALUATION SYSTEM TEST REPORT{Colors.RESET}")
        print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")

        # Count results
        passed = sum(1 for _, success, _ in self.results if success)
        failed = sum(1 for _, success, _ in self.results if not success)
        total = len(self.results)

        # Summary
        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"  Total Tests: {total}")
        print(f"  {Colors.GREEN}Passed: {passed}{Colors.RESET}")
        print(f"  {Colors.RED}Failed: {failed}{Colors.RESET}")
        print(f"  Time: {elapsed:.2f}s")

        # Pass rate
        pass_rate = (passed / total * 100) if total > 0 else 0
        if pass_rate >= 90:
            status_color = Colors.GREEN
            status = "EXCELLENT"
        elif pass_rate >= 70:
            status_color = Colors.YELLOW
            status = "GOOD"
        else:
            status_color = Colors.RED
            status = "NEEDS ATTENTION"

        print(f"  Pass Rate: {status_color}{pass_rate:.1f}%{Colors.RESET}")
        print(f"  Status: {status_color}{status}{Colors.RESET}")

        # Detailed results
        print(f"\n{Colors.BOLD}Detailed Results:{Colors.RESET}")
        print(f"{'Test':<30} {'Status':<10} {'Details'}")
        print("-" * 60)

        for test_name, success, details in self.results:
            status = f"{Colors.GREEN}✓ PASS{Colors.RESET}" if success else f"{Colors.RED}✗ FAIL{Colors.RESET}"
            print(f"{test_name:<30} {status:<20} {details}")

        # Recommendations
        if failed > 0:
            print(f"\n{Colors.BOLD}Recommendations:{Colors.RESET}")
            for test_name, success, details in self.results:
                if not success:
                    if "jsonschema" in test_name:
                        print(f"  - Install jsonschema: pip install jsonschema==4.21.1")
                    elif "Config" in test_name:
                        print(f"  - Check config/eval_frontend.json")
                    elif "Schema" in test_name:
                        print(f"  - Verify docs/eval_history.schema.json")
                    else:
                        print(f"  - Debug {test_name}: {details}")

        print(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")

    def run_all_tests(self) -> int:
        """Run all tests and return exit code."""
        print(f"{Colors.BOLD}Starting Evaluation System Tests...{Colors.RESET}")
        print(f"Mode: {'Quick' if self.quick else 'Full'}")

        # Run test suites
        self.test_config_integrity()
        self.test_file_integrity()
        self.test_schema_validation()

        if not self.quick:
            self.test_evaluation_pipeline()
            self.test_report_generation()

        self.test_retention_policy()
        self.test_version_monitoring()
        self.test_makefile_targets()

        # Generate report
        self.generate_report()

        # Return exit code
        failed_count = sum(1 for _, success, _ in self.results if not success)
        return 0 if failed_count == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Test evaluation system components")
    parser.add_argument("--quick", action="store_true",
                        help="Skip time-consuming tests")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()

    runner = TestRunner(verbose=args.verbose, quick=args.quick)
    return runner.run_all_tests()


if __name__ == "__main__":
    sys.exit(main())