#!/usr/bin/env python3
"""
Validate that all expected alert names are present in alerting rules.

This script checks that new alerts (DriftBaselineStale, CacheHitRateLow, FaissIndexStale)
are properly defined in the Prometheus alerting rules configuration.

Exit Codes:
    0: All expected alerts found
    1: Missing or invalid alerts
    2: File not found or parse error
"""

import sys
import yaml
from pathlib import Path
from typing import Dict, List, Set

# ANSI color codes
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BOLD = '\033[1m'
RESET = '\033[0m'

# Expected new alerts from recent enhancements
EXPECTED_ALERTS = {
    "DriftBaselineStale": {
        "description": "Drift baseline age exceeds maximum threshold",
        "severity": "warning",
        "component": "drift"
    },
    "CacheHitRateLow": {
        "description": "Analysis cache hit rate below 30%",
        "severity": "warning",
        "component": "cache"
    },
    "FaissIndexStale": {
        "description": "Faiss index not refreshed for over 1 hour",
        "severity": "warning",
        "component": "cad_analysis"
    }
}

# All existing alerts that should be present
CORE_ALERTS = [
    "HighErrorRate",
    "ProviderDown",
    "ProviderTimeout",
    "ModelLoadError",
    "ResourceExhaustion",
    "SLOViolation",
    "ErrorBudgetCritical",
    "HighLatency",
    "InputRejectionSpike",
]


def load_alert_rules(rules_file: str) -> Dict:
    """Load and parse alert rules YAML file."""
    rules_path = Path(rules_file)

    if not rules_path.exists():
        print(f"{RED}Error: Rules file not found: {rules_file}{RESET}")
        sys.exit(2)

    try:
        with open(rules_path, 'r') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"{RED}Error parsing YAML: {e}{RESET}")
        sys.exit(2)


def extract_alert_names(rules_data: Dict) -> Set[str]:
    """Extract all alert names from rules."""
    alert_names = set()

    for group in rules_data.get('groups', []):
        for rule in group.get('rules', []):
            alert_name = rule.get('alert')
            if alert_name:
                alert_names.add(alert_name)

    return alert_names


def validate_alert_details(rules_data: Dict, alert_name: str, expected_details: Dict) -> List[str]:
    """Validate alert has expected severity and component labels."""
    issues = []

    for group in rules_data.get('groups', []):
        for rule in group.get('rules', []):
            if rule.get('alert') == alert_name:
                labels = rule.get('labels', {})

                # Check severity
                if 'severity' in expected_details:
                    if labels.get('severity') != expected_details['severity']:
                        issues.append(
                            f"  ⚠ Expected severity '{expected_details['severity']}', "
                            f"got '{labels.get('severity')}'"
                        )

                # Check component
                if 'component' in expected_details:
                    if labels.get('component') != expected_details['component']:
                        issues.append(
                            f"  ⚠ Expected component '{expected_details['component']}', "
                            f"got '{labels.get('component')}'"
                        )

                # Check runbook_url exists
                annotations = rule.get('annotations', {})
                if 'runbook_url' not in annotations:
                    issues.append(f"  ⚠ Missing runbook_url annotation")

                return issues

    return []


def main():
    """Main validation logic."""
    rules_file = "config/prometheus/alerting_rules.yml"

    print(f"{BOLD}Prometheus Alert Names Validation{RESET}")
    print("=" * 60)

    # Load rules
    rules_data = load_alert_rules(rules_file)
    alert_names = extract_alert_names(rules_data)

    print(f"\nFound {len(alert_names)} total alerts in {rules_file}")

    # Check for new alerts
    print(f"\n{BOLD}Checking New Alerts:{RESET}")
    all_found = True
    detail_issues = []

    for alert_name, expected_details in EXPECTED_ALERTS.items():
        if alert_name in alert_names:
            print(f"{GREEN}✓{RESET} {alert_name} - {expected_details['description']}")

            # Validate details
            issues = validate_alert_details(rules_data, alert_name, expected_details)
            if issues:
                detail_issues.extend([f"{alert_name}:"] + issues)
        else:
            print(f"{RED}✗{RESET} {alert_name} - MISSING")
            all_found = False

    # Check core alerts still present
    print(f"\n{BOLD}Checking Core Alerts:{RESET}")
    missing_core = []
    for alert_name in CORE_ALERTS:
        if alert_name not in alert_names:
            missing_core.append(alert_name)
            print(f"{RED}✗{RESET} {alert_name} - MISSING")

    if not missing_core:
        print(f"{GREEN}✓{RESET} All {len(CORE_ALERTS)} core alerts present")

    # Report detail issues
    if detail_issues:
        print(f"\n{YELLOW}{BOLD}Alert Detail Issues:{RESET}")
        for issue in detail_issues:
            print(f"{YELLOW}{issue}{RESET}")

    # Summary
    print("\n" + "=" * 60)
    if all_found and not missing_core and not detail_issues:
        print(f"{GREEN}{BOLD}✓ Validation PASSED{RESET}")
        print(f"  All {len(EXPECTED_ALERTS)} new alerts present")
        print(f"  All {len(CORE_ALERTS)} core alerts present")
        print(f"  No configuration issues detected")
        sys.exit(0)
    else:
        print(f"{RED}{BOLD}✗ Validation FAILED{RESET}")
        if not all_found:
            print(f"  Missing new alerts")
        if missing_core:
            print(f"  Missing {len(missing_core)} core alerts")
        if detail_issues:
            print(f"  {len(detail_issues)} configuration issues")
        sys.exit(1)


if __name__ == '__main__':
    main()
