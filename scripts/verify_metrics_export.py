#!/usr/bin/env python3
"""
Verifies that required Prometheus metrics are exported via src/utils/analysis_metrics.py __all__.
Also parses Prometheus rules YAML and Grafana JSON to catch obvious typos in metric names.
"""
import json
import re
from pathlib import Path
import glob

REQUIRED = {
    # Feature extraction / performance
    "feature_extraction_latency_seconds",
    # Vector migration
    "vector_migrate_dimension_delta",
    # Degraded mode lifecycle
    "similarity_degraded_total",
    "faiss_recovery_attempts_total",
    "faiss_degraded_duration_seconds",
    "faiss_next_recovery_eta_seconds",
    "faiss_recovery_suppressed_total",
    "faiss_recovery_suppression_remaining_seconds",
    # Security / opcode audit
    "model_opcode_audit_total",
    "model_opcode_whitelist_violations_total",
}

def load_exports() -> set[str]:
    txt = Path("src/utils/analysis_metrics.py").read_text(encoding="utf-8")
    m = re.search(r"__all__\s*=\s*\[(.*?)\]", txt, re.S)
    if not m:
        raise SystemExit("__all__ not found in analysis_metrics.py")
    items = re.findall(r"'([^']+)'|\"([^\"]+)\"", m.group(1))
    names = {a or b for a, b in items}
    return names

def _extract_metrics(expr: str) -> set[str]:
    """Extract metric names by matching identifiers immediately before '{' or '('.
    This avoids capturing label keys/values inside braces.
    """
    candidates = set(re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*(?=\{|\()", expr))
    ignore = {
        # functions
        "rate", "sum", "increase", "histogram_quantile", "avg", "max", "min", "count",
        "irate", "avg_over_time", "sum_over_time", "stddev_over_time", "stdvar_over_time",
        "time", "scalar", "changes", "delta", "idelta",
    }
    return {m for m in candidates if m not in ignore}


def parse_yaml_metrics(path: Path) -> set[str]:
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if "expr:" in line:
            expr = line.split("expr:", 1)[1]
            names |= _extract_metrics(expr)
    return names

def parse_json_metrics(path: Path) -> set[str]:
    names: set[str] = set()
    data = json.loads(path.read_text(encoding="utf-8"))
    for panel in data.get("panels", []):
        for tgt in panel.get("targets", []):
            expr = tgt.get("expr", "")
            if expr:
                names |= _extract_metrics(expr)
    return names

def main() -> None:
    exports = load_exports()
    missing = REQUIRED - exports
    if missing:
        raise SystemExit(f"Missing required exports: {sorted(missing)}")

    # Identify extraneous metrics possibly orphaned (in exports but unused by rules/dashboard)
    referenced: set[str] = set()

    # Cross-check dashboards/rules against exports
    check_paths = []
    # Glob all rule files and dashboards for comprehensive verification
    for rp in glob.glob("prometheus/rules/*.yaml"):
        check_paths.append(("yaml", Path(rp)))
    for dp in glob.glob("grafana/dashboards/*.json"):
        check_paths.append(("json", Path(dp)))

    for kind, p in check_paths:
        if kind == "yaml":
            referenced |= parse_yaml_metrics(p)
        else:
            referenced |= parse_json_metrics(p)
    unknown = {m for m in referenced if m not in exports and not m.endswith("_bucket") and not m.endswith("_sum") and not m.endswith("_count")}
    unknown -= {"process_resident_memory_bytes", "by", "and", "or"}
    if unknown:
        raise SystemExit(f"Referenced metrics not exported: {sorted(unknown)}")

    unused = sorted(m for m in exports if m not in referenced and m in REQUIRED)
    if unused:
        raise SystemExit(f"Required metrics not referenced by rules/dashboard: {unused}")

    print("Metrics export verification passed.")

if __name__ == "__main__":
    main()
