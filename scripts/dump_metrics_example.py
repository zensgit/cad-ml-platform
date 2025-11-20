#!/usr/bin/env python3
"""
Dump all Prometheus metrics with example values.

Usage:
    python scripts/dump_metrics_example.py
    python scripts/dump_metrics_example.py --format json
"""

import argparse
import json
from typing import Dict, List


def get_all_metrics() -> Dict[str, Dict[str, str]]:
    """
    Return all Prometheus metrics with metadata.

    Returns:
        Dict mapping metric name to {type, help, example_value}
    """
    return {
        # === OCR Request Metrics ===
        "ocr_requests_total": {
            "type": "Counter",
            "help": "Total OCR requests processed",
            "labels": ["provider", "status"],
            "example_labels": 'provider="deepseek", status="success"',
            "example_value": "1234",
        },
        "ocr_request_duration_seconds": {
            "type": "Histogram",
            "help": "OCR request duration in seconds",
            "labels": ["provider"],
            "buckets": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            "example_labels": 'provider="paddle"',
            "example_value": "0.523",
        },
        "ocr_request_size_bytes": {
            "type": "Histogram",
            "help": "OCR request image size in bytes",
            "labels": ["provider"],
            "buckets": [1024, 10240, 102400, 1048576, 10485760],
            "example_labels": 'provider="deepseek"',
            "example_value": "524288",
        },

        # === Fallback Strategy Metrics ===
        "ocr_fallback_level": {
            "type": "Counter",
            "help": "Fallback level usage count",
            "labels": ["level", "provider"],
            "example_labels": 'level="json_strict", provider="deepseek"',
            "example_value": "850",
        },
        "ocr_parsing_failures_total": {
            "type": "Counter",
            "help": "Total parsing failures by level",
            "labels": ["level", "provider"],
            "example_labels": 'level="markdown_fence", provider="deepseek"',
            "example_value": "12",
        },

        # === Cache Metrics ===
        "ocr_cache_operations_total": {
            "type": "Counter",
            "help": "Cache operations count",
            "labels": ["operation", "status"],
            "example_labels": 'operation="get", status="hit"',
            "example_value": "3456",
        },
        "ocr_cache_size_bytes": {
            "type": "Gauge",
            "help": "Current cache size in bytes",
            "labels": [],
            "example_labels": "",
            "example_value": "104857600",
        },
        "ocr_cache_entries": {
            "type": "Gauge",
            "help": "Current number of cache entries",
            "labels": [],
            "example_labels": "",
            "example_value": "1250",
        },

        # === Error Metrics ===
        "ocr_errors_total": {
            "type": "Counter",
            "help": "Total errors by type",
            "labels": ["error_type", "provider"],
            "example_labels": 'error_type="timeout", provider="deepseek"',
            "example_value": "5",
        },
        "ocr_retry_attempts_total": {
            "type": "Counter",
            "help": "Total retry attempts",
            "labels": ["provider", "final_status"],
            "example_labels": 'provider="deepseek", final_status="success"',
            "example_value": "23",
        },

        # === Dimension Extraction Metrics ===
        "ocr_dimensions_extracted_total": {
            "type": "Counter",
            "help": "Total dimensions extracted",
            "labels": ["dimension_type", "provider"],
            "example_labels": 'dimension_type="diameter", provider="deepseek"',
            "example_value": "4567",
        },
        "ocr_symbols_extracted_total": {
            "type": "Counter",
            "help": "Total symbols extracted",
            "labels": ["symbol_type", "provider"],
            "example_labels": 'symbol_type="surface_roughness", provider="paddle"',
            "example_value": "890",
        },

        # === Confidence Metrics ===
        "ocr_confidence_score": {
            "type": "Histogram",
            "help": "OCR confidence scores distribution",
            "labels": ["provider"],
            "buckets": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
            "example_labels": 'provider="deepseek"',
            "example_value": "0.87",
        },
        "ocr_brier_score": {
            "type": "Histogram",
            "help": "Brier score for confidence calibration",
            "labels": ["provider"],
            "buckets": [0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
            "example_labels": 'provider="deepseek"',
            "example_value": "0.12",
        },

        # === Provider Routing Metrics ===
        "ocr_provider_selection_total": {
            "type": "Counter",
            "help": "Provider selection count",
            "labels": ["strategy", "selected_provider"],
            "example_labels": 'strategy="auto", selected_provider="deepseek"',
            "example_value": "678",
        },
        "ocr_provider_switch_total": {
            "type": "Counter",
            "help": "Provider switches due to failure",
            "labels": ["from_provider", "to_provider", "reason"],
            "example_labels": 'from_provider="deepseek", to_provider="paddle", reason="timeout"',
            "example_value": "8",
        },

        # === Evaluation Metrics (Golden Dataset) ===
        "ocr_evaluation_dimension_recall": {
            "type": "Gauge",
            "help": "Dimension recall on golden dataset",
            "labels": ["provider", "dataset_version"],
            "example_labels": 'provider="deepseek", dataset_version="v1.0"',
            "example_value": "0.92",
        },
        "ocr_evaluation_edge_f1": {
            "type": "Gauge",
            "help": "Edge F1 score on golden dataset",
            "labels": ["provider", "dataset_version"],
            "example_labels": 'provider="paddle", dataset_version="v1.0"',
            "example_value": "0.78",
        },

        # === System Resource Metrics ===
        "ocr_gpu_utilization_percent": {
            "type": "Gauge",
            "help": "GPU utilization percentage",
            "labels": ["gpu_id"],
            "example_labels": 'gpu_id="0"',
            "example_value": "65.5",
        },
        "ocr_gpu_memory_used_bytes": {
            "type": "Gauge",
            "help": "GPU memory used in bytes",
            "labels": ["gpu_id"],
            "example_labels": 'gpu_id="0"',
            "example_value": "5368709120",
        },
        "ocr_active_requests": {
            "type": "Gauge",
            "help": "Number of active OCR requests",
            "labels": ["provider"],
            "example_labels": 'provider="deepseek"',
            "example_value": "3",
        },
    }


def format_prometheus(metrics: Dict) -> List[str]:
    """Format metrics in Prometheus exposition format."""
    lines = []

    for name, meta in sorted(metrics.items()):
        # Help text
        lines.append(f"# HELP {name} {meta['help']}")

        # Type
        lines.append(f"# TYPE {name} {meta['type'].lower()}")

        # Example value with labels
        if meta.get("example_labels"):
            metric_line = f"{name}{{{meta['example_labels']}}} {meta['example_value']}"
        else:
            metric_line = f"{name} {meta['example_value']}"
        lines.append(metric_line)

        # Histogram buckets example
        if meta["type"] == "Histogram" and "buckets" in meta:
            lines.append(f"# Buckets: {meta['buckets']}")

        lines.append("")  # Blank line between metrics

    return lines


def format_json(metrics: Dict) -> str:
    """Format metrics as JSON."""
    return json.dumps(metrics, indent=2, ensure_ascii=False)


def format_table(metrics: Dict) -> List[str]:
    """Format metrics as a readable table."""
    lines = [
        "=" * 100,
        f"{'Metric Name':<40} {'Type':<12} {'Example Labels':<48}",
        "=" * 100,
    ]

    for name, meta in sorted(metrics.items()):
        example_labels = meta.get("example_labels", "N/A")
        lines.append(f"{name:<40} {meta['type']:<12} {example_labels:<48}")

    lines.append("=" * 100)
    lines.append(f"\nTotal Metrics: {len(metrics)}")

    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Dump Prometheus metrics for OCR module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--format",
        choices=["prometheus", "json", "table"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--filter",
        help="Filter metrics by name substring (case-insensitive)",
    )

    args = parser.parse_args()

    metrics = get_all_metrics()

    # Apply filter if specified
    if args.filter:
        filter_lower = args.filter.lower()
        metrics = {
            name: meta
            for name, meta in metrics.items()
            if filter_lower in name.lower()
        }

        if not metrics:
            print(f"No metrics found matching filter: {args.filter}")
            return

    # Format output
    if args.format == "prometheus":
        output = format_prometheus(metrics)
        print("\n".join(output))
    elif args.format == "json":
        output = format_json(metrics)
        print(output)
    else:  # table
        output = format_table(metrics)
        print("\n".join(output))


if __name__ == "__main__":
    main()
