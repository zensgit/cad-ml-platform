#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

METRICS_FILE = Path("src/utils/analysis_metrics.py")
DASHBOARDS = [
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
    # crude parse: tokenise by non-metric chars, filter out functions/keywords
    tokens = re.findall(r'[a-zA-Z_:][a-zA-Z0-9_:]*', expr)
    functions = {
        'sum','rate','histogram_quantile','increase','topk','max_over_time','min_over_time','changes'
    }
    return {t for t in tokens if t not in functions}

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

def main() -> int:
    exported = load_exported_metrics()
    dash = collect_dashboard_metrics()
    missing = sorted(n for n in dash if n and n not in exported)
    if missing:
        print("Dashboard references metrics not exported:")
        for n in missing:
            print(f" - {n}")
        return 1
    print("Dashboard metrics all exported.")
    return 0

if __name__ == '__main__':
    sys.exit(main())

