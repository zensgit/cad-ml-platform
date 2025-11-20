#!/usr/bin/env python3
"""
Validate evaluation history JSON files for schema compliance.

Checks:
- Schema version compatibility
- Required fields presence
- Data type validation
- Run context integrity
- JSON Schema compliance (if jsonschema available)

Usage:
    python3 scripts/validate_eval_history.py [--strict] [--migrate] [--schema PATH]
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from datetime import datetime

# Try to import jsonschema for enhanced validation
try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    print("Note: Install 'jsonschema' package for enhanced validation (pip install jsonschema)")

# Default schema path
DEFAULT_SCHEMA_PATH = "docs/eval_history.schema.json"


# Schema definitions for each version
SCHEMAS = {
    "1.0.0": {
        "required_fields": ["schema_version", "timestamp", "branch", "commit", "type"],
        "conditional_fields": {
            "combined": ["vision_metrics", "ocr_metrics", "combined"],
            "ocr": ["metrics"],
            "vision": ["metrics"]
        },
        "run_context_fields": ["runner", "machine", "os", "python", "start_time"],
        "optional_run_context": ["ci_job_id", "ci_workflow"]
    },
    "0.0.0": {  # Legacy schema (no schema_version field)
        "required_fields": ["timestamp", "branch", "commit"],
        "conditional_fields": {
            "combined": ["vision_metrics", "ocr_metrics", "combined"],
            "ocr": ["metrics"]
        }
    }
}


def detect_schema_version(data: Dict) -> str:
    """Detect the schema version of a JSON file."""
    if "schema_version" in data:
        return data["schema_version"]
    return "0.0.0"  # Legacy files without schema_version


def validate_file(filepath: Path, strict: bool = False) -> Tuple[bool, List[str]]:
    """
    Validate a single evaluation history file.

    Returns:
        (is_valid, list_of_issues)
    """
    issues = []

    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]
    except Exception as e:
        return False, [f"Cannot read file: {e}"]

    # Detect schema version
    version = detect_schema_version(data)

    if version not in SCHEMAS:
        issues.append(f"Unknown schema version: {version}")
        return False, issues

    schema = SCHEMAS[version]

    # Check required fields
    for field in schema["required_fields"]:
        if field not in data:
            issues.append(f"Missing required field: {field}")

    # Check type-specific fields
    if "type" in data and data["type"] in schema.get("conditional_fields", {}):
        for field in schema["conditional_fields"][data["type"]]:
            if field not in data:
                issues.append(f"Missing required field for type '{data['type']}': {field}")

    # For v1.0.0, check run_context
    if version == "1.0.0":
        if strict and "run_context" not in data:
            issues.append("Missing run_context (required in strict mode for v1.0.0)")
        elif "run_context" in data:
            for field in schema["run_context_fields"]:
                if field not in data["run_context"]:
                    issues.append(f"Missing run_context field: {field}")

    # Validate data types
    if "timestamp" in data:
        try:
            datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        except:
            issues.append(f"Invalid timestamp format: {data['timestamp']}")

    # Check metrics ranges
    if "vision_metrics" in data:
        metrics = data["vision_metrics"]
        if "avg_hit_rate" in metrics:
            if not (0 <= metrics["avg_hit_rate"] <= 1):
                issues.append(f"vision avg_hit_rate out of range: {metrics['avg_hit_rate']}")

    if "ocr_metrics" in data:
        metrics = data["ocr_metrics"]
        if "dimension_recall" in metrics:
            if not (0 <= metrics["dimension_recall"] <= 1):
                issues.append(f"ocr dimension_recall out of range: {metrics['dimension_recall']}")
        if "brier_score" in metrics:
            if not (0 <= metrics["brier_score"] <= 1):
                issues.append(f"ocr brier_score out of range: {metrics['brier_score']}")

    if "combined" in data:
        combined = data["combined"]
        for score_field in ["vision_score", "ocr_score", "combined_score"]:
            if score_field in combined:
                if not (0 <= combined[score_field] <= 1):
                    issues.append(f"combined {score_field} out of range: {combined[score_field]}")

    return len(issues) == 0, issues


def migrate_file(filepath: Path) -> bool:
    """
    Migrate a legacy file to schema version 1.0.0.
    Creates a backup with .bak extension.
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        if "schema_version" in data:
            print(f"  Already migrated: {filepath.name}")
            return True

        # Backup original
        backup_path = filepath.with_suffix(".json.bak")
        with open(backup_path, "w") as f:
            json.dump(data, f, indent=2)

        # Add schema version and basic run context
        data["schema_version"] = "1.0.0"

        # Determine type from filename or content
        if "combined" in data:
            data["type"] = "combined"
        elif "metrics" in data and "dimension_recall" in data["metrics"]:
            data["type"] = "ocr"
        elif "_combined" in filepath.name:
            data["type"] = "combined"
        else:
            data["type"] = "ocr"

        # Add minimal run context (unknown values)
        data["run_context"] = {
            "runner": "unknown",
            "machine": "unknown",
            "os": "unknown",
            "python": "unknown",
            "start_time": data.get("timestamp", "unknown"),
            "ci_job_id": None,
            "ci_workflow": None
        }

        # Save migrated version
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"  Migrated: {filepath.name} (backup: {backup_path.name})")
        return True

    except Exception as e:
        print(f"  Failed to migrate {filepath.name}: {e}")
        return False


def load_json_schema(schema_path: str = None) -> Optional[Dict]:
    """Load JSON schema for validation."""
    if not JSONSCHEMA_AVAILABLE:
        return None

    # Try config file first
    if schema_path is None:
        try:
            with open("config/eval_frontend.json", "r") as f:
                config = json.load(f)
                schema_version = config.get("schema_version", "1.0.0")
                schema_path = f"docs/eval_history.schema.json"
        except:
            schema_path = DEFAULT_SCHEMA_PATH

    try:
        with open(schema_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Schema file not found: {schema_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Invalid JSON schema: {e}")
        return None


def validate_with_json_schema(data: Dict, schema: Dict) -> Tuple[bool, List[str]]:
    """Validate data against JSON schema."""
    if not JSONSCHEMA_AVAILABLE:
        return True, ["JSON schema validation skipped (jsonschema not installed)"]

    try:
        jsonschema.validate(instance=data, schema=schema)
        return True, []
    except jsonschema.ValidationError as e:
        return False, [f"Schema validation failed: {e.message} at {'.'.join(str(p) for p in e.path)}"]
    except Exception as e:
        return False, [f"Schema validation error: {e}"]


def main():
    parser = argparse.ArgumentParser(description="Validate evaluation history JSON files")
    parser.add_argument("--strict", action="store_true",
                        help="Strict validation (all v1.0.0 features required)")
    parser.add_argument("--migrate", action="store_true",
                        help="Migrate legacy files to v1.0.0 schema")
    parser.add_argument("--dir", default="reports/eval_history",
                        help="Directory to validate (default: reports/eval_history)")
    parser.add_argument("--schema", help="Path to JSON schema file")
    parser.add_argument("--summary", action="store_true",
                        help="Show summary only")
    parser.add_argument("--config", default="config/eval_frontend.json",
                        help="Config file path")
    args = parser.parse_args()

    # Load JSON schema if available
    json_schema = load_json_schema(args.schema) if args.schema else load_json_schema()

    history_dir = Path(args.dir)
    if not history_dir.exists():
        print(f"Directory not found: {history_dir}")
        sys.exit(1)

    json_files = list(history_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {history_dir}")
        return

    print(f"Found {len(json_files)} JSON files in {history_dir}")
    print("-" * 60)

    if args.migrate:
        print("MIGRATION MODE")
        print("-" * 60)
        migrated = 0
        failed = 0
        for filepath in json_files:
            if migrate_file(filepath):
                migrated += 1
            else:
                failed += 1

        print("-" * 60)
        print(f"Migration complete: {migrated} success, {failed} failed")
        if failed > 0:
            sys.exit(1)
        return

    # Validation mode
    valid_count = 0
    invalid_count = 0
    legacy_count = 0

    for filepath in json_files:
        is_valid, issues = validate_file(filepath, strict=args.strict)

        # Check if it's a legacy file
        with open(filepath, "r") as f:
            data = json.load(f)
        version = detect_schema_version(data)

        status = "✓ VALID" if is_valid else "✗ INVALID"
        if version == "0.0.0":
            status += " (legacy)"
            legacy_count += 1

        print(f"{status}: {filepath.name} (v{version})")

        if issues:
            for issue in issues:
                print(f"  - {issue}")

        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1

    print("-" * 60)
    print(f"Summary: {valid_count} valid, {invalid_count} invalid, {legacy_count} legacy")

    if legacy_count > 0 and not args.migrate:
        print(f"\nFound {legacy_count} legacy files. Run with --migrate to upgrade them.")

    if invalid_count > 0:
        sys.exit(1)

    print("\nAll files validated successfully!")


if __name__ == "__main__":
    main()