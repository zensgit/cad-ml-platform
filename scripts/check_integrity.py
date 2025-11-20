#!/usr/bin/env python3
"""
Lightweight integrity checker using centralized config.

Validates SHA-384 hash and file size for critical dependencies.
Reads configuration from config/eval_frontend.json.

Usage:
    python3 scripts/check_integrity.py [--strict] [--file PATH] [--sha384 HASH]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple


def load_config(config_path: str = "config/eval_frontend.json") -> Dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"WARNING: Config file not found: {config_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in config file: {e}")
        sys.exit(1)


def calculate_sha384(file_path: Path) -> Tuple[Optional[str], Optional[int]]:
    """Calculate SHA-384 hash and file size."""
    if not file_path.exists():
        return None, None

    try:
        sha384 = hashlib.sha384()
        size = 0

        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha384.update(chunk)
                size += len(chunk)

        # Convert to base64 (SRI format)
        import base64
        hash_b64 = base64.b64encode(sha384.digest()).decode("ascii")

        return hash_b64, size

    except Exception as e:
        print(f"ERROR: Failed to calculate hash: {e}")
        return None, None


def check_integrity(
    file_path: str,
    expected_sha384: str,
    expected_size: Optional[int] = None,
    warn_only: bool = False
) -> Tuple[bool, str]:
    """Check file integrity against expected values."""
    path = Path(file_path)

    # Check existence
    if not path.exists():
        return False, f"File not found: {file_path}"

    # Calculate actual values
    actual_sha384, actual_size = calculate_sha384(path)

    if actual_sha384 is None:
        return False, "Failed to calculate hash"

    # Check hash
    if actual_sha384 != expected_sha384:
        msg = (f"Hash mismatch:\n"
               f"  Expected: {expected_sha384[:20]}...\n"
               f"  Actual:   {actual_sha384[:20]}...")
        return False, msg

    # Check size if provided
    if expected_size and actual_size != expected_size:
        msg = f"Size mismatch: expected {expected_size} bytes, got {actual_size} bytes"
        return False, msg

    return True, f"OK (hash match, {actual_size} bytes)"


def main():
    parser = argparse.ArgumentParser(description="Check file integrity")
    parser.add_argument("--file", help="File path to check (overrides config)")
    parser.add_argument("--sha384", help="Expected SHA-384 hash (overrides config)")
    parser.add_argument("--size", type=int, help="Expected file size in bytes")
    parser.add_argument("--strict", action="store_true",
                        help="Exit with code 1 on failure (default: warn only)")
    parser.add_argument("--config", default="config/eval_frontend.json",
                        help="Config file path")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Determine what to check
    if args.file and args.sha384:
        # Command-line override
        file_path = args.file
        expected_sha384 = args.sha384
        expected_size = args.size
        warn_only = not args.strict
    elif "chartjs" in config:
        # Use config values
        chartjs = config["chartjs"]
        file_path = chartjs.get("local_path")
        expected_sha384 = chartjs.get("sha384")
        expected_size = chartjs.get("expected_size_bytes")
        warn_only = chartjs.get("warn_only", True) if not args.strict else False

        if not chartjs.get("integrity_check_enabled", True):
            print("INFO: Integrity check disabled in config")
            return 0
    else:
        print("ERROR: No file specified and no config available")
        print("Usage: check_integrity.py --file PATH --sha384 HASH")
        return 1

    # Validate inputs
    if not file_path or not expected_sha384:
        print("ERROR: Missing required parameters (file and sha384)")
        return 1

    # Print header
    if args.verbose:
        print("=" * 60)
        print("File Integrity Check")
        print("=" * 60)
        print(f"File: {file_path}")
        print(f"Expected SHA-384: {expected_sha384[:20]}...")
        if expected_size:
            print(f"Expected size: {expected_size} bytes")
        print(f"Mode: {'STRICT' if not warn_only else 'WARN'}")
        print("-" * 60)

    # Check integrity
    is_valid, message = check_integrity(
        file_path,
        expected_sha384,
        expected_size,
        warn_only
    )

    # Report results
    if is_valid:
        print(f"✅ PASS: {message}")
        return 0
    else:
        if warn_only:
            print(f"⚠️  WARNING: {message}")
            print("   (Running in warn-only mode, not failing)")
            return 0
        else:
            print(f"❌ FAIL: {message}")
            return 1


if __name__ == "__main__":
    sys.exit(main())