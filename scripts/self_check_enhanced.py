#!/usr/bin/env python3
"""Enhanced self-check script for CAD ML Platform.

Comprehensive health check covering:
1. Service health and configuration
2. Provider availability
3. Metrics consistency
4. Error code mappings
5. File encoding validation
6. Security status
7. Performance baselines
"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class SelfCheck:
    """Comprehensive self-check utility."""

    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = []
        self.errors = []

    def print_header(self, title: str) -> None:
        """Print section header."""
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print('=' * 60)

    def print_status(self, check: str, passed: bool, details: str = "") -> None:
        """Print check status."""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} | {check}")
        if details:
            print(f"       {details}")

        if passed:
            self.checks_passed += 1
        else:
            self.checks_failed += 1
            self.errors.append(f"{check}: {details}")

    def print_warning(self, message: str) -> None:
        """Print warning message."""
        print(f"⚠️  WARN | {message}")
        self.warnings.append(message)

    def run_command(self, cmd: List[str], check: bool = True) -> Tuple[bool, str]:
        """Run command and return success status and output."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=check,
                cwd=project_root
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, e.stderr
        except Exception as e:
            return False, str(e)

    async def check_service_health(self) -> None:
        """Check if service is running and healthy."""
        self.print_header("Service Health Check")

        # Check if service is running
        import httpx

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Check health endpoint
                response = await client.get("http://localhost:8000/health")
                if response.status_code == 200:
                    data = response.json()
                    self.print_status(
                        "Service health endpoint",
                        data.get("status") == "healthy",
                        f"Status: {data.get('status')}"
                    )

                    # Check services
                    services = data.get("services", {})
                    for service, status in services.items():
                        is_up = status in ["up", "disabled"]
                        self.print_status(
                            f"Service: {service}",
                            is_up,
                            f"Status: {status}"
                        )

                    # Check configuration visibility
                    config = data.get("config", {})
                    self.print_status(
                        "Configuration visibility",
                        bool(config),
                        f"{len(config)} configuration sections available"
                    )
                else:
                    self.print_status(
                        "Service health endpoint",
                        False,
                        f"HTTP {response.status_code}"
                    )

                # Check readiness
                ready_response = await client.get("http://localhost:8000/ready")
                self.print_status(
                    "Service readiness",
                    ready_response.status_code == 200,
                    f"HTTP {ready_response.status_code}"
                )

                # Check metrics endpoint
                metrics_response = await client.get("http://localhost:8000/metrics")
                self.print_status(
                    "Metrics endpoint",
                    metrics_response.status_code == 200,
                    f"HTTP {metrics_response.status_code}"
                )

        except httpx.ConnectError:
            self.print_warning("Service not running - start with 'make serve'")
        except Exception as e:
            self.print_status("Service health check", False, str(e))

    def check_environment(self) -> None:
        """Check environment and dependencies."""
        self.print_header("Environment Check")

        # Python version
        import sys
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self.print_status(
            "Python version",
            sys.version_info >= (3, 11),
            f"Version: {python_version}"
        )

        # Check virtual environment
        in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        if not in_venv:
            self.print_warning("Not running in virtual environment")

        # Check required packages
        required_packages = [
            "fastapi",
            "uvicorn",
            "pydantic",
            "httpx",
            "prometheus-client",
            "pytest"
        ]

        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                self.print_status(f"Package: {package}", True, "Installed")
            except ImportError:
                self.print_status(f"Package: {package}", False, "Not installed")

    def check_error_codes(self) -> None:
        """Check ErrorCode consistency."""
        self.print_header("Error Code Consistency")

        try:
            from src.core.errors import ErrorCode

            # Check all ErrorCode values are unique
            values = [code.value for code in ErrorCode]
            unique_values = set(values)
            self.print_status(
                "ErrorCode uniqueness",
                len(values) == len(unique_values),
                f"{len(values)} codes, {len(unique_values)} unique"
            )

            # Check naming convention
            invalid_names = [
                code.value for code in ErrorCode
                if not (code.value.isupper() and ("_" in code.value or code.value.isalpha()))
            ]
            self.print_status(
                "ErrorCode naming convention",
                len(invalid_names) == 0,
                f"{len(invalid_names)} invalid names" if invalid_names else "All valid"
            )

            # Check OCR exceptions compatibility
            from src.core.ocr.exceptions import OCR_ERRORS

            valid_codes = {code.value for code in ErrorCode}
            invalid_mappings = [
                f"{key}={value}" for key, value in OCR_ERRORS.items()
                if value not in valid_codes
            ]
            self.print_status(
                "OCR_ERRORS compatibility",
                len(invalid_mappings) == 0,
                f"{len(invalid_mappings)} invalid mappings" if invalid_mappings else "All valid"
            )

        except Exception as e:
            self.print_status("Error code check", False, str(e))

    def check_file_encoding(self) -> None:
        """Check Python files for UTF-8 encoding."""
        self.print_header("File Encoding Check")

        non_utf8_files = []

        # Check all Python files
        for py_file in project_root.rglob("*.py"):
            # Skip virtual environment
            if ".venv" in str(py_file) or "venv" in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    f.read()
            except UnicodeDecodeError:
                non_utf8_files.append(str(py_file.relative_to(project_root)))

        self.print_status(
            "UTF-8 encoding",
            len(non_utf8_files) == 0,
            f"{len(non_utf8_files)} files with encoding issues" if non_utf8_files else "All files UTF-8"
        )

        if non_utf8_files:
            for file in non_utf8_files[:5]:  # Show first 5
                print(f"       - {file}")
            if len(non_utf8_files) > 5:
                print(f"       ... and {len(non_utf8_files) - 5} more")

    def check_tests(self) -> None:
        """Run basic tests."""
        self.print_header("Test Suite Check")

        # Run fast tests
        success, output = self.run_command(
            ["python3", "-m", "pytest", "tests/", "-q", "--tb=no", "-x"]
        )

        if success:
            # Parse pytest output
            lines = output.strip().split('\n')
            summary = lines[-1] if lines else ""
            self.print_status("Test suite", True, summary)
        else:
            self.print_status("Test suite", False, "Tests failed")

    def check_security(self) -> None:
        """Run security checks."""
        self.print_header("Security Check")

        # Check for exposed secrets
        sensitive_patterns = [
            "password",
            "secret",
            "token",
            "api_key",
            "private_key"
        ]

        exposed_files = []
        for pattern in sensitive_patterns:
            cmd = ["grep", "-r", "-i", f"{pattern}\\s*=\\s*['\"]", ".",
                   "--include=*.py", "--include=*.json", "--include=*.yaml",
                   "--exclude-dir=.venv", "--exclude-dir=tests"]
            success, output = self.run_command(cmd, check=False)
            if output.strip():
                exposed_files.extend(output.strip().split('\n'))

        self.print_status(
            "No exposed secrets",
            len(exposed_files) == 0,
            f"Found {len(exposed_files)} potential exposures" if exposed_files else "No secrets found"
        )

        # Check file permissions
        world_writable = []
        for path in project_root.rglob("*"):
            if path.is_file() and not ".venv" in str(path):
                if os.stat(path).st_mode & 0o002:
                    world_writable.append(str(path.relative_to(project_root)))

        self.print_status(
            "File permissions",
            len(world_writable) == 0,
            f"{len(world_writable)} world-writable files" if world_writable else "All files secure"
        )

    def check_documentation(self) -> None:
        """Check documentation completeness."""
        self.print_header("Documentation Check")

        required_docs = [
            "README.md",
            "docs/OPERATIONAL_RUNBOOK.md",
            "docs/KEY_HIGHLIGHTS.md",
            "docs/CI_FAILURE_ROUTING.md",
            "docs/HEALTH_ENDPOINT_CONFIG.md",
            "CHANGELOG.md"
        ]

        for doc in required_docs:
            doc_path = project_root / doc
            exists = doc_path.exists()
            if exists:
                size = doc_path.stat().st_size
                self.print_status(
                    f"Document: {doc}",
                    size > 100,
                    f"Size: {size:,} bytes"
                )
            else:
                self.print_status(f"Document: {doc}", False, "Not found")

    def check_configuration(self) -> None:
        """Check configuration files."""
        self.print_header("Configuration Check")

        config_files = [
            "pyproject.toml",
            "Makefile",
            ".github/workflows/ci.yml",
            "config/eval_frontend.json"
        ]

        for config in config_files:
            config_path = project_root / config
            if config_path.exists():
                try:
                    if config.endswith('.json'):
                        with open(config_path) as f:
                            json.load(f)
                        self.print_status(f"Config: {config}", True, "Valid JSON")
                    else:
                        self.print_status(f"Config: {config}", True, "Exists")
                except json.JSONDecodeError:
                    self.print_status(f"Config: {config}", False, "Invalid JSON")
            else:
                self.print_status(f"Config: {config}", False, "Not found")

    def check_makefile_targets(self) -> None:
        """Check key Makefile targets."""
        self.print_header("Makefile Targets Check")

        important_targets = [
            "install",
            "test",
            "serve",
            "dev",
            "lint",
            "security-audit",
            "eval-phase6",
            "badges"
        ]

        # Get available targets
        success, output = self.run_command(
            ["make", "-qp"],
            check=False
        )

        available_targets = set()
        if success:
            for line in output.split('\n'):
                if line and not line.startswith('#') and ':' in line:
                    target = line.split(':')[0].strip()
                    if not target.startswith('.'):
                        available_targets.add(target)

        for target in important_targets:
            self.print_status(
                f"Target: make {target}",
                target in available_targets,
                "Available" if target in available_targets else "Not found"
            )

    async def run_all_checks(self) -> int:
        """Run all self-checks."""
        print("\n" + "=" * 60)
        print("  CAD ML Platform - Enhanced Self Check")
        print("=" * 60)

        # Run checks
        await self.check_service_health()
        self.check_environment()
        self.check_error_codes()
        self.check_file_encoding()
        self.check_tests()
        self.check_security()
        self.check_documentation()
        self.check_configuration()
        self.check_makefile_targets()

        # Print summary
        self.print_header("Summary")

        total_checks = self.checks_passed + self.checks_failed
        success_rate = (self.checks_passed / total_checks * 100) if total_checks > 0 else 0

        print(f"\nTotal Checks: {total_checks}")
        print(f"Passed: {self.checks_passed} ({success_rate:.1f}%)")
        print(f"Failed: {self.checks_failed}")
        print(f"Warnings: {len(self.warnings)}")

        if self.errors:
            print("\n❌ Failed Checks:")
            for error in self.errors[:10]:  # Show first 10
                print(f"  - {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more")

        if self.warnings:
            print("\n⚠️  Warnings:")
            for warning in self.warnings[:5]:  # Show first 5
                print(f"  - {warning}")
            if len(self.warnings) > 5:
                print(f"  ... and {len(self.warnings) - 5} more")

        # Overall status
        print("\n" + "=" * 60)
        if self.checks_failed == 0:
            print("  🎉 All checks passed! System is healthy.")
        elif self.checks_failed <= 3:
            print("  ⚠️  Minor issues detected. Review failed checks.")
        else:
            print("  ❌ Multiple issues detected. Action required!")
        print("=" * 60)

        # Return exit code
        return 0 if self.checks_failed == 0 else 1


async def main():
    """Main entry point."""
    checker = SelfCheck()
    exit_code = await checker.run_all_checks()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())