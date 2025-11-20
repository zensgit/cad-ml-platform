#!/usr/bin/env python3
"""
Security audit for dependencies and code.

Integrates security scanning into the evaluation workflow.

Usage:
    python3 scripts/security_audit.py [--severity critical|high|medium|low]
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class SecurityAuditor:
    """Perform security audits on dependencies and code."""

    def __init__(self, report_dir: str = "reports/security"):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.vulnerabilities = []
        self.audit_results = {}

    def check_python_dependencies(self) -> Tuple[bool, List[Dict]]:
        """Check Python dependencies for known vulnerabilities."""
        print("Checking Python dependencies...")

        vulnerabilities = []

        # Method 1: pip-audit (if available)
        try:
            result = subprocess.run(
                ["pip-audit", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                audit_data = json.loads(result.stdout)
                for vuln in audit_data.get("vulnerabilities", []):
                    vulnerabilities.append({
                        "type": "python_dependency",
                        "package": vuln["name"],
                        "installed_version": vuln["installed_version"],
                        "vulnerable_versions": vuln["vulnerable_versions"],
                        "severity": self._map_severity(vuln.get("severity", "unknown")),
                        "description": vuln.get("description", ""),
                        "fix_version": vuln.get("fixed_version"),
                        "cve": vuln.get("id")
                    })
                print(f"  pip-audit: Found {len(vulnerabilities)} vulnerabilities")
            else:
                print("  pip-audit not available or failed")

        except FileNotFoundError:
            print("  pip-audit not installed, trying safety...")

            # Method 2: safety check (fallback)
            try:
                result = subprocess.run(
                    ["safety", "check", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if result.stdout:
                    safety_data = json.loads(result.stdout)
                    for vuln in safety_data:
                        vulnerabilities.append({
                            "type": "python_dependency",
                            "package": vuln[0],
                            "installed_version": vuln[2],
                            "vulnerable_versions": vuln[1],
                            "severity": "high",  # Safety doesn't provide severity
                            "description": vuln[3],
                            "fix_version": vuln[5] if len(vuln) > 5 else None,
                            "cve": vuln[4] if len(vuln) > 4 else None
                        })
                    print(f"  safety: Found {len(vulnerabilities)} vulnerabilities")

            except FileNotFoundError:
                print("  Neither pip-audit nor safety installed")

        except Exception as e:
            print(f"  Error checking dependencies: {e}")

        self.vulnerabilities.extend(vulnerabilities)
        return len(vulnerabilities) == 0, vulnerabilities

    def check_javascript_dependencies(self) -> Tuple[bool, List[Dict]]:
        """Check JavaScript dependencies using npm audit."""
        print("Checking JavaScript dependencies...")

        if not Path("package.json").exists():
            print("  No package.json found, skipping")
            return True, []

        vulnerabilities = []

        try:
            result = subprocess.run(
                ["npm", "audit", "--json"],
                capture_output=True,
                text=True,
                timeout=60
            )

            audit_data = json.loads(result.stdout)

            if "vulnerabilities" in audit_data:
                for pkg_name, vuln_info in audit_data["vulnerabilities"].items():
                    vulnerabilities.append({
                        "type": "javascript_dependency",
                        "package": pkg_name,
                        "severity": vuln_info.get("severity", "unknown"),
                        "vulnerable_versions": vuln_info.get("range", ""),
                        "description": vuln_info.get("title", ""),
                        "fix_available": vuln_info.get("fixAvailable", False),
                        "cve": vuln_info.get("cves", [])
                    })

            print(f"  npm audit: Found {len(vulnerabilities)} vulnerabilities")

        except FileNotFoundError:
            print("  npm not found, skipping")
        except Exception as e:
            print(f"  Error checking npm dependencies: {e}")

        self.vulnerabilities.extend(vulnerabilities)
        return len(vulnerabilities) == 0, vulnerabilities

    def check_docker_images(self) -> Tuple[bool, List[Dict]]:
        """Check Docker images for vulnerabilities using trivy."""
        print("Checking Docker images...")

        dockerfile_path = Path("Dockerfile")
        if not dockerfile_path.exists():
            print("  No Dockerfile found, skipping")
            return True, []

        vulnerabilities = []

        try:
            # Use trivy if available
            result = subprocess.run(
                ["trivy", "config", ".", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0 and result.stdout:
                trivy_data = json.loads(result.stdout)
                for result_item in trivy_data.get("Results", []):
                    for vuln in result_item.get("Vulnerabilities", []):
                        vulnerabilities.append({
                            "type": "docker",
                            "package": vuln.get("PkgName", ""),
                            "installed_version": vuln.get("InstalledVersion", ""),
                            "severity": vuln.get("Severity", "").lower(),
                            "description": vuln.get("Description", ""),
                            "fix_version": vuln.get("FixedVersion", ""),
                            "cve": vuln.get("VulnerabilityID", "")
                        })

                print(f"  trivy: Found {len(vulnerabilities)} vulnerabilities")
            else:
                print("  trivy scan completed (no vulnerabilities or not available)")

        except FileNotFoundError:
            print("  trivy not installed, skipping Docker scan")
        except Exception as e:
            print(f"  Error scanning Docker: {e}")

        self.vulnerabilities.extend(vulnerabilities)
        return len(vulnerabilities) == 0, vulnerabilities

    def check_secrets(self) -> Tuple[bool, List[Dict]]:
        """Check for exposed secrets in code."""
        print("Checking for exposed secrets...")

        secrets_found = []

        # Simple patterns to check
        patterns = [
            ("AWS Key", r"AKIA[0-9A-Z]{16}"),
            ("API Key", r"api[_-]?key['\"]?\s*[:=]\s*['\"]?[A-Za-z0-9]{32,}"),
            ("Private Key", r"-----BEGIN (RSA|DSA|EC) PRIVATE KEY-----"),
            ("GitHub Token", r"gh[ps]_[A-Za-z0-9]{36}"),
            ("Generic Secret", r"(password|secret|token)['\"]?\s*[:=]\s*['\"]?[A-Za-z0-9]{16,}")
        ]

        try:
            # Use git-secrets if available
            result = subprocess.run(
                ["git", "secrets", "--scan"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0 and result.stderr:
                # git-secrets found something
                secrets_found.append({
                    "type": "exposed_secret",
                    "tool": "git-secrets",
                    "findings": result.stderr,
                    "severity": "critical"
                })
                print(f"  git-secrets: Found potential secrets")

        except:
            print("  git-secrets not available")

        # Try gitleaks as alternative
        try:
            result = subprocess.run(
                ["gitleaks", "detect", "--report-format", "json", "--report-path", "/tmp/gitleaks.json"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if Path("/tmp/gitleaks.json").exists():
                with open("/tmp/gitleaks.json", "r") as f:
                    leaks = json.load(f)
                    if leaks:
                        for leak in leaks:
                            secrets_found.append({
                                "type": "exposed_secret",
                                "file": leak.get("File", ""),
                                "line": leak.get("LineNumber", 0),
                                "rule": leak.get("RuleID", ""),
                                "severity": "critical",
                                "description": leak.get("Description", "")
                            })
                        print(f"  gitleaks: Found {len(leaks)} potential secrets")

        except:
            print("  gitleaks not available")

        self.vulnerabilities.extend(secrets_found)
        return len(secrets_found) == 0, secrets_found

    def check_code_quality(self) -> Tuple[bool, List[Dict]]:
        """Check code quality and security issues."""
        print("Checking code quality and security...")

        issues = []

        # Use bandit for Python security
        try:
            result = subprocess.run(
                ["bandit", "-r", "src/", "-f", "json"],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.stdout:
                bandit_data = json.loads(result.stdout)
                for issue in bandit_data.get("results", []):
                    issues.append({
                        "type": "code_security",
                        "file": issue.get("filename", ""),
                        "line": issue.get("line_number", 0),
                        "severity": issue.get("issue_severity", "").lower(),
                        "confidence": issue.get("issue_confidence", "").lower(),
                        "description": issue.get("issue_text", ""),
                        "cwe": issue.get("issue_cwe", {})
                    })
                print(f"  bandit: Found {len(issues)} security issues")

        except FileNotFoundError:
            print("  bandit not installed")
        except Exception as e:
            print(f"  Error running bandit: {e}")

        self.vulnerabilities.extend(issues)
        return len(issues) == 0, issues

    def _map_severity(self, severity: str) -> str:
        """Map severity levels to standard format."""
        severity_lower = severity.lower()
        if severity_lower in ["critical", "high", "medium", "low"]:
            return severity_lower
        elif severity_lower in ["error", "major"]:
            return "high"
        elif severity_lower in ["warning", "minor"]:
            return "medium"
        else:
            return "low"

    def generate_report(self, min_severity: str = "low") -> Dict:
        """Generate security audit report."""
        severity_levels = ["critical", "high", "medium", "low"]
        min_severity_index = severity_levels.index(min_severity)

        # Filter vulnerabilities by severity
        filtered_vulns = [
            v for v in self.vulnerabilities
            if severity_levels.index(self._map_severity(v.get("severity", "low"))) <= min_severity_index
        ]

        # Count by severity
        severity_counts = {
            "critical": sum(1 for v in filtered_vulns if v.get("severity") == "critical"),
            "high": sum(1 for v in filtered_vulns if v.get("severity") == "high"),
            "medium": sum(1 for v in filtered_vulns if v.get("severity") == "medium"),
            "low": sum(1 for v in filtered_vulns if v.get("severity") == "low")
        }

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_vulnerabilities": len(filtered_vulns),
                "by_severity": severity_counts,
                "by_type": {}
            },
            "vulnerabilities": filtered_vulns,
            "recommendations": [],
            "status": "pass" if len(filtered_vulns) == 0 else "fail"
        }

        # Count by type
        for vuln in filtered_vulns:
            vuln_type = vuln.get("type", "unknown")
            report["summary"]["by_type"][vuln_type] = report["summary"]["by_type"].get(vuln_type, 0) + 1

        # Generate recommendations
        if severity_counts["critical"] > 0:
            report["recommendations"].append("URGENT: Fix critical vulnerabilities immediately")
        if severity_counts["high"] > 0:
            report["recommendations"].append("HIGH PRIORITY: Address high severity issues")
        if any(v.get("type") == "exposed_secret" for v in filtered_vulns):
            report["recommendations"].append("CRITICAL: Rotate exposed secrets and remove from code")

        return report

    def save_report(self, report: Dict, filename: Optional[str] = None) -> str:
        """Save security audit report."""
        if not filename:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"security_audit_{timestamp}.json"

        report_path = self.report_dir / filename
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return str(report_path)

    def run_full_audit(self, min_severity: str = "low") -> Tuple[bool, Dict]:
        """Run complete security audit."""
        print("=" * 60)
        print("Security Audit Starting...")
        print("=" * 60)

        # Run all checks
        self.check_python_dependencies()
        self.check_javascript_dependencies()
        self.check_docker_images()
        self.check_secrets()
        self.check_code_quality()

        # Generate report
        report = self.generate_report(min_severity)

        # Save report
        report_file = self.save_report(report)

        print("\n" + "=" * 60)
        print("Security Audit Complete")
        print("=" * 60)
        print(f"Total vulnerabilities: {report['summary']['total_vulnerabilities']}")
        print(f"  Critical: {report['summary']['by_severity']['critical']}")
        print(f"  High:     {report['summary']['by_severity']['high']}")
        print(f"  Medium:   {report['summary']['by_severity']['medium']}")
        print(f"  Low:      {report['summary']['by_severity']['low']}")
        print(f"\nReport saved to: {report_file}")

        # Return pass/fail based on critical/high vulnerabilities
        critical_high = (report['summary']['by_severity']['critical'] +
                        report['summary']['by_severity']['high'])

        return critical_high == 0, report


def main():
    parser = argparse.ArgumentParser(description="Security audit for dependencies and code")
    parser.add_argument("--severity", choices=["critical", "high", "medium", "low"],
                        default="low", help="Minimum severity to report")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON to stdout")
    parser.add_argument("--fail-on-high", action="store_true",
                        help="Exit with error if high/critical vulnerabilities found")

    args = parser.parse_args()

    auditor = SecurityAuditor()
    passed, report = auditor.run_full_audit(min_severity=args.severity)

    if args.json:
        print(json.dumps(report, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to: {args.output}")

    # Granular exit codes for different issue types
    # Exit codes:
    # 0 - No issues found
    # 1 - General failure / mixed issues
    # 2 - Critical vulnerabilities found
    # 3 - Exposed secrets detected
    # 4 - High severity dependencies
    # 5 - Docker/container issues
    # 6 - Code security issues

    exit_code = 0

    by_type = report.get("summary", {}).get("by_type", {})
    by_severity = report.get("summary", {}).get("by_severity", {})

    # Priority-based exit code assignment
    if by_type.get("exposed_secret", 0) > 0:
        print("\n🔐 CRITICAL: Exposed secrets detected")
        exit_code = 3
    elif by_severity.get("critical", 0) > 0:
        print("\n❌ CRITICAL: Critical vulnerabilities found")
        exit_code = 2
    elif by_severity.get("high", 0) > 0 and args.fail_on_high:
        print("\n⚠️  HIGH: High severity issues found")
        exit_code = 4
    elif by_type.get("docker", 0) > 0:
        print("\n🐳 Docker/container vulnerabilities found")
        exit_code = 5
    elif by_type.get("code_security", 0) > 0:
        print("\n🔍 Code security issues found")
        exit_code = 6
    elif not passed:
        print("\n⚠️  Security audit found issues")
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())