#!/usr/bin/env python3
"""
Federated Learning Log Auditor.

Scans federated learning logs for potential PII or sensitive data leaks.
Checks for:
- IP Addresses
- Email addresses
- Unmasked User IDs
"""

import re
import sys
import argparse
from pathlib import Path

# Regex patterns for PII
PATTERNS = {
    "IP_ADDRESS": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    # Example: UserID should be hashed, so look for simple integer IDs or names
    "POTENTIAL_USER_ID": r"user_id=\d+", 
}

def scan_file(file_path: Path) -> bool:
    print(f"Scanning {file_path}...")
    issues_found = False
    
    try:
        with file_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                for label, pattern in PATTERNS.items():
                    if re.search(pattern, line):
                        print(f"  [LINE {line_num}] ⚠️  Found {label}: {line.strip()}")
                        issues_found = True
    except Exception as e:
        print(f"  Error reading file: {e}")
        return False

    return issues_found

def main():
    parser = argparse.ArgumentParser(description="Audit Federated Learning logs for PII")
    parser.add_argument("--log-dir", default="logs", help="Directory containing logs")
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Log directory {log_dir} does not exist.")
        return

    total_issues = 0
    for log_file in log_dir.glob("*.log"):
        if scan_file(log_file):
            total_issues += 1
            
    if total_issues > 0:
        print(f"\n❌ Audit Failed: Found potential PII in {total_issues} files.")
        sys.exit(1)
    else:
        print("\n✅ Audit Passed: No PII detected in logs.")

if __name__ == "__main__":
    main()
