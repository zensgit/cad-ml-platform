#!/usr/bin/env python3
"""
Full integration test for the evaluation system.

Runs a complete workflow from evaluation to reporting to validation.

Usage:
    python3 scripts/run_full_integration_test.py [--cleanup]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


class Colors:
    """Terminal colors."""
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


class IntegrationTest:
    """Run full integration test workflow."""

    def __init__(self, cleanup: bool = False):
        self.cleanup = cleanup
        self.test_dir = Path("test_integration_output")
        self.results = []
        self.start_time = time.time()

    def run_command(self, cmd: List[str], description: str) -> Tuple[bool, str]:
        """Run a command and capture output."""
        print(f"  {Colors.BLUE}→ {description}{Colors.RESET}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            success = result.returncode == 0
            output = result.stdout + result.stderr
            status = f"{Colors.GREEN}✓{Colors.RESET}" if success else f"{Colors.RED}✗{Colors.RESET}"
            print(f"    {status} {description}")
            return success, output
        except Exception as e:
            print(f"    {Colors.RED}✗ {description}: {str(e)}{Colors.RESET}")
            return False, str(e)

    def save_output(self, filename: str, content: str) -> None:
        """Save output to test directory."""
        self.test_dir.mkdir(exist_ok=True)
        with open(self.test_dir / filename, "w") as f:
            f.write(content)

    def run_workflow(self) -> bool:
        """Run complete evaluation workflow."""
        print(f"\n{Colors.BOLD}INTEGRATION TEST WORKFLOW{Colors.RESET}")
        print("=" * 60)

        all_success = True

        # Step 1: Configuration Integrity Check
        print(f"\n{Colors.BOLD}Step 1: Configuration Integrity{Colors.RESET}")
        success, output = self.run_command(
            ["python3", "scripts/check_integrity.py"],
            "Check file integrity"
        )
        self.save_output("integrity_check.txt", output)
        self.results.append(("Integrity Check", success))
        all_success = all_success and success

        # Step 2: Run Combined Evaluation
        print(f"\n{Colors.BOLD}Step 2: Run Evaluation{Colors.RESET}")
        success, output = self.run_command(
            ["python3", "scripts/evaluate_vision_ocr_combined.py"],
            "Vision+OCR combined evaluation"
        )
        self.save_output("evaluation.txt", output)

        # Extract score
        score = None
        if "Combined:" in output:
            for line in output.split('\n'):
                if "Combined:" in line:
                    score = line.split(":")[-1].strip()
                    break

        self.results.append(("Combined Evaluation", success, score))
        all_success = all_success and success

        # Step 3: Validate History JSON
        print(f"\n{Colors.BOLD}Step 3: Validate Output{Colors.RESET}")
        success, output = self.run_command(
            ["python3", "scripts/validate_eval_history.py", "--dir", "reports/eval_history"],
            "Validate history JSON"
        )
        self.save_output("validation.txt", output)
        self.results.append(("JSON Validation", success))
        all_success = all_success and success

        # Step 4: Generate Reports
        print(f"\n{Colors.BOLD}Step 4: Generate Reports{Colors.RESET}")

        # Basic report
        success1, output1 = self.run_command(
            ["python3", "scripts/generate_eval_report.py"],
            "Generate static report"
        )
        self.save_output("report_basic.txt", output1)
        self.results.append(("Basic Report", success1))

        # Enhanced report
        success2, output2 = self.run_command(
            ["python3", "scripts/generate_eval_report_v2.py", "--use-cdn"],
            "Generate interactive report"
        )
        self.save_output("report_enhanced.txt", output2)
        self.results.append(("Enhanced Report", success2))

        all_success = all_success and (success1 or success2)

        # Step 5: Test Retention Policy
        print(f"\n{Colors.BOLD}Step 5: Retention Policy{Colors.RESET}")
        success, output = self.run_command(
            ["python3", "scripts/manage_eval_retention.py", "--dry-run"],
            "Check retention policy"
        )
        self.save_output("retention.txt", output)

        # Parse retention stats
        files_found = "Unknown"
        if "Found" in output:
            for line in output.split('\n'):
                if "Found" in line and "files" in line:
                    files_found = line.strip()
                    break

        self.results.append(("Retention Policy", success, files_found))
        all_success = all_success and success

        # Step 6: Version Monitoring
        print(f"\n{Colors.BOLD}Step 6: Version Monitoring{Colors.RESET}")
        success, output = self.run_command(
            ["python3", "scripts/check_chartjs_updates.py"],
            "Check dependency versions"
        )
        self.save_output("version_check.txt", output)
        self.results.append(("Version Check", success))
        all_success = all_success and success

        # Step 7: Health Check
        print(f"\n{Colors.BOLD}Step 7: System Health Check{Colors.RESET}")
        success, output = self.run_command(
            ["make", "health-check"],
            "Run health check"
        )
        self.save_output("health_check.txt", output)
        self.results.append(("Health Check", success))
        all_success = all_success and success

        return all_success

    def generate_report(self) -> None:
        """Generate integration test report."""
        elapsed = time.time() - self.start_time

        print(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}INTEGRATION TEST REPORT{Colors.RESET}")
        print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")

        # Summary statistics
        passed = sum(1 for r in self.results if r[1])
        failed = sum(1 for r in self.results if not r[1])
        total = len(self.results)
        pass_rate = (passed / total * 100) if total > 0 else 0

        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"  Timestamp: {datetime.now(timezone.utc).isoformat()}")
        print(f"  Duration: {elapsed:.2f}s")
        print(f"  Total Steps: {total}")
        print(f"  {Colors.GREEN}Passed: {passed}{Colors.RESET}")
        print(f"  {Colors.RED}Failed: {failed}{Colors.RESET}")
        print(f"  Pass Rate: {pass_rate:.1f}%")

        # Overall status
        if pass_rate == 100:
            status = f"{Colors.GREEN}✓ ALL TESTS PASSED{Colors.RESET}"
        elif pass_rate >= 80:
            status = f"{Colors.YELLOW}⚠ MOSTLY PASSING{Colors.RESET}"
        else:
            status = f"{Colors.RED}✗ NEEDS ATTENTION{Colors.RESET}"

        print(f"  Status: {status}")

        # Detailed results
        print(f"\n{Colors.BOLD}Detailed Results:{Colors.RESET}")
        print(f"{'Step':<25} {'Status':<10} {'Details'}")
        print("-" * 60)

        for result in self.results:
            name = result[0]
            success = result[1]
            details = result[2] if len(result) > 2 else ""

            status = f"{Colors.GREEN}PASS{Colors.RESET}" if success else f"{Colors.RED}FAIL{Colors.RESET}"
            print(f"{name:<25} {status:<20} {details}")

        # Output location
        print(f"\n{Colors.BOLD}Output Files:{Colors.RESET}")
        print(f"  Test outputs saved to: {self.test_dir}/")

        # Recommendations
        if failed > 0:
            print(f"\n{Colors.BOLD}Recommendations:{Colors.RESET}")
            for result in self.results:
                if not result[1]:
                    name = result[0]
                    if "Integrity" in name:
                        print(f"  - Review integrity check configuration")
                    elif "Evaluation" in name:
                        print(f"  - Check evaluation script dependencies")
                    elif "Validation" in name:
                        print(f"  - Install jsonschema: pip install jsonschema==4.21.1")
                    elif "Report" in name:
                        print(f"  - Check report generation dependencies")
                    else:
                        print(f"  - Debug {name} failure")

        print(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")

        # Save JSON report
        report_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration": elapsed,
            "pass_rate": pass_rate,
            "results": [
                {
                    "step": r[0],
                    "passed": r[1],
                    "details": r[2] if len(r) > 2 else None
                }
                for r in self.results
            ]
        }

        report_path = self.test_dir / "integration_report.json"
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"\nJSON report saved to: {report_path}")

    def cleanup_outputs(self) -> None:
        """Clean up test outputs."""
        if self.cleanup and self.test_dir.exists():
            print(f"\n{Colors.YELLOW}Cleaning up test outputs...{Colors.RESET}")
            import shutil
            shutil.rmtree(self.test_dir)
            print(f"  {Colors.GREEN}✓ Removed {self.test_dir}/{Colors.RESET}")

    def run(self) -> int:
        """Run complete integration test."""
        print(f"{Colors.BOLD}Starting Full Integration Test...{Colors.RESET}")
        print(f"Output directory: {self.test_dir}/")

        # Run workflow
        success = self.run_workflow()

        # Generate report
        self.generate_report()

        # Cleanup if requested
        if self.cleanup:
            self.cleanup_outputs()

        return 0 if success else 1


def main():
    parser = argparse.ArgumentParser(
        description="Run full integration test for evaluation system"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up test outputs after completion"
    )

    args = parser.parse_args()

    tester = IntegrationTest(cleanup=args.cleanup)
    return tester.run()


if __name__ == "__main__":
    sys.exit(main())