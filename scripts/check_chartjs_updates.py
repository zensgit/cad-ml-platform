#!/usr/bin/env python3
"""
Chart.js version monitoring script.

Checks for updates without automatic upgrade.
Designed for scheduled runs (e.g., weekly cron).

Usage:
    python3 scripts/check_chartjs_updates.py [--create-issue]
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
from datetime import datetime
from typing import Dict, Optional, Tuple


def load_config(config_path: str = "config/eval_frontend.json") -> Dict:
    """Load configuration."""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        sys.exit(1)


def get_npm_package_info(package_name: str, timeout: int = 2) -> Optional[Dict]:
    """Query npm registry for package information."""
    url = f"https://registry.npmjs.org/{package_name}"

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"WARNING: Network error querying npm: {e}")
        print("SKIPPED (no network or timeout)")
        return None
    except Exception as e:
        print(f"ERROR: Failed to query npm: {e}")
        return None


def parse_version(version_str: str) -> Tuple[int, int, int]:
    """Parse semantic version string."""
    try:
        parts = version_str.split(".")
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2].split("-")[0]) if len(parts) > 2 else 0
        return major, minor, patch
    except Exception:
        return 0, 0, 0


def classify_version_diff(current: str, latest: str) -> str:
    """Classify the difference between versions."""
    curr_major, curr_minor, curr_patch = parse_version(current)
    late_major, late_minor, late_patch = parse_version(latest)

    if curr_major < late_major:
        return "MAJOR"
    elif curr_minor < late_minor:
        return "MINOR"
    elif curr_patch < late_patch:
        return "PATCH"
    else:
        return "NONE"


def check_for_updates(config: Dict) -> Optional[Dict]:
    """Check for Chart.js updates."""
    monitoring = config.get("version_monitoring", {})

    if not monitoring.get("enabled", False):
        print("Version monitoring is disabled in config")
        return None

    chartjs = config.get("chartjs", {})
    current_version = chartjs.get("version", "unknown")
    timeout = monitoring.get("timeout_seconds", 2)

    print(f"Current Chart.js version: {current_version}")
    print("Checking for updates...")

    # Query npm registry
    package_info = get_npm_package_info("chart.js", timeout)
    if not package_info:
        return None

    latest_version = package_info.get("dist-tags", {}).get("latest", "unknown")
    versions_list = list(package_info.get("versions", {}).keys())

    # Check if current version is suppressed
    suppressed = monitoring.get("suppressed_versions", [])
    if latest_version in suppressed:
        print(f"Latest version {latest_version} is in suppression list")
        return None

    # Classify difference
    diff_class = classify_version_diff(current_version, latest_version)

    result = {
        "current": current_version,
        "latest": latest_version,
        "diff_class": diff_class,
        "total_versions": len(versions_list),
        "npm_url": f"https://www.npmjs.com/package/chart.js/v/{latest_version}"
    }

    # Add security advisory check (simplified)
    if diff_class in ["MAJOR", "MINOR", "PATCH"]:
        result["action"] = "Update available"
        result["recommendation"] = f"Review changelog before updating to {latest_version}"

    return result


def create_github_issue(result: Dict, config: Dict) -> bool:
    """Create GitHub issue for major updates (if configured)."""
    monitoring = config.get("version_monitoring", {})

    if not monitoring.get("create_issue_on_major", False):
        return False

    if result.get("diff_class") != "MAJOR":
        return False

    print("Would create GitHub issue for MAJOR version update")
    print("(Not implemented - requires GITHUB_TOKEN)")
    return False


def main():
    parser = argparse.ArgumentParser(description="Check for Chart.js updates")
    parser.add_argument("--create-issue", action="store_true",
                        help="Create GitHub issue for major updates")
    parser.add_argument("--config", default="config/eval_frontend.json",
                        help="Config file path")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Check for updates
    result = check_for_updates(config)

    if not result:
        print("No update check performed")
        return 0

    # Report results
    print("=" * 60)
    print("Version Check Results")
    print("=" * 60)
    print(f"Current version: {result['current']}")
    print(f"Latest version:  {result['latest']}")
    print(f"Difference:      {result['diff_class']}")

    if result['diff_class'] != "NONE":
        print(f"Action:          {result.get('action', 'None')}")
        print(f"Recommendation:  {result.get('recommendation', 'None')}")
        print(f"NPM URL:         {result.get('npm_url', 'N/A')}")

        # Create issue if requested
        if args.create_issue:
            created = create_github_issue(result, config)
            if created:
                print("GitHub issue created")
    else:
        print("Status:          Up to date")

    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
