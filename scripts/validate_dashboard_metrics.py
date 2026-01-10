#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

METRICS_FILE = Path("src/utils/analysis_metrics.py")
DASHBOARDS = [
    Path("config/grafana/dashboard_main.json"),
    Path("grafana/dashboards/observability.json"),
    Path("grafana/dashboards/observability_eta_suppression.json"),
]

def load_exported_metrics() -> set[str]:
    exported = set()
    text = METRICS_FILE.read_text(encoding="utf-8")
    # Collect metric names from __all__ and definitions
    for m in re.finditer(r'\b(__all__\s*=\s*\[.*?\])', text, re.S):
        block = m.group(1)
        exported.update(re.findall(r'"([a-zA-Z0-9_:]+)"', block))
    # Fallback: find explicit declarations
    exported.update(re.findall(r'Counter\(\s*"([a-zA-Z0-9_:]+)"', text))
    exported.update(re.findall(r'Gauge\(\s*"([a-zA-Z0-9_:]+)"', text))
    exported.update(re.findall(r'Histogram\(\s*"([a-zA-Z0-9_:]+)"', text))
    return exported

def extract_metric_names_from_expr(expr: str) -> set[str]:
    """Extract metric names from PromQL expressions.
    Handles metrics with label selectors, recording rules, and bare metrics.
    """
    # First, remove label selector blocks {} to avoid capturing label names
    expr_clean = re.sub(r"\{[^}]*\}", "", expr)
    # Remove aggregation clause labels: by (label1, label2) and without (...)
    expr_clean = re.sub(r"\b(by|without)\s*\([^)]*\)", "", expr_clean)
    # Remove grouping modifiers: on (label) and ignoring (label)
    expr_clean = re.sub(r"\b(on|ignoring)\s*\([^)]*\)", "", expr_clean)

    # Match metrics before '(' (function calls) or '[' (range vectors)
    with_suffix = set(re.findall(r"([a-zA-Z_:][a-zA-Z0-9_:]*)\s*(?=\(|\[)", expr_clean))
    # Match standalone metrics at word boundaries
    standalone = set(re.findall(r"(?<![a-zA-Z0-9_:])([a-zA-Z_:][a-zA-Z0-9_:]*)\b", expr_clean))
    candidates = with_suffix | standalone
    ignore = {
        # PromQL functions
        "rate", "sum", "increase", "histogram_quantile", "avg", "max", "min", "count",
        "irate", "avg_over_time", "sum_over_time", "stddev_over_time", "stdvar_over_time",
        "time", "scalar", "changes", "delta", "idelta", "topk", "bottomk", "absent",
        "clamp", "clamp_max", "clamp_min", "ceil", "floor", "round", "exp", "ln", "log2",
        "log10", "sqrt", "abs", "sgn", "sort", "sort_desc", "label_join", "label_replace",
        "vector", "group_left", "group_right", "on", "ignoring", "offset",
        "max_over_time", "min_over_time",
        # Common PromQL keywords
        "by", "without", "and", "or", "unless", "bool",
        "interval",
    }
    return {m for m in candidates if m not in ignore}

def collect_dashboard_metrics() -> set[str]:
    names: set[str] = set()
    for path in DASHBOARDS:
        if not path.exists():
            print(f"WARN: dashboard missing: {path}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        panels = data.get('panels', [])
        for p in panels:
            targets = p.get('targets', [])
            for t in targets:
                expr = t.get('expr') or t.get('target') or ''
                names.update(extract_metric_names_from_expr(expr))
    return names

def is_metric_exported(metric: str, exported: set[str]) -> bool:
    """Check if metric or its base name (for histograms) is exported."""
    if metric in exported:
        return True
    # For histogram variants, check if base metric is exported
    histogram_suffixes = ["_bucket", "_sum", "_count"]
    for suffix in histogram_suffixes:
        if metric.endswith(suffix):
            base = metric[:-len(suffix)]
            if base in exported:
                return True
    return False

def main() -> int:
    exported = load_exported_metrics()
    dash = collect_dashboard_metrics()
    missing = sorted(n for n in dash if n and not is_metric_exported(n, exported))
    if missing:
        print("Dashboard references metrics not exported:")
        for n in missing:
            print(f" - {n}")
        return 1
    print("Dashboard metrics all exported.")
    return 0

if __name__ == '__main__':
    sys.exit(main())
