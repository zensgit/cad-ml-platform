# Vision CAD Feature Grid Baseline Report Run 100 Design

## Overview
Generate a baseline-only markdown summary from the 100-sample grid baseline
results.

## Command
```
python3 - <<'PY'
import json
from pathlib import Path

payload = json.loads(
    Path('reports/vision_cad_feature_grid_baseline_20260106_100.json').read_text()
)
results = payload.get('results', [])
lines = ["# CAD Feature Benchmark Baseline Summary", ""]
metrics = ["total_lines", "total_circles", "total_arcs", "avg_ink_ratio", "avg_components"]
for idx, entry in enumerate(results, start=1):
    thresholds = entry.get('thresholds', {})
    summary = entry.get('summary', {})
    threshold_text = "-"
    if thresholds:
        threshold_text = ", ".join(
            f"{key}={thresholds[key]}" for key in sorted(thresholds.keys())
        )
    lines.append(f"## Combo {idx}")
    lines.append(f"**Thresholds**: {threshold_text}")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    for key in metrics:
        lines.append(f"| {key} | {summary.get(key, '-')} |")
    lines.append("")

Path('reports/vision_cad_feature_grid_baseline_report_20260106_100.md').write_text(
    "\n".join(lines).rstrip() + "\n"
)
PY
```

## Output
- `reports/vision_cad_feature_grid_baseline_report_20260106_100.md`
