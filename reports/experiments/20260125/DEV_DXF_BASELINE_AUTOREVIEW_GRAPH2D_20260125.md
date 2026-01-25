# DEV_DXF_BASELINE_AUTOREVIEW_GRAPH2D_20260125

## Goal
Re-run the automated DXF baseline review with Graph2D enabled to obtain real agreement/conflict rates (no manual image inspection).

## Scope
- Dataset: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf`
- Requested sample size: 300
- Available DXF files: 110 (sample size capped at 110)
- Runtime: `.venv-graph/bin/python` (Torch + ezdxf available)

## Work performed
1. Batch analyze DXF files using local TestClient (Graph2D enabled)
2. Auto-review results via filename + synonym matching
3. Summarize review outcomes + label coverage
4. Emit conflict list for follow-up
5. Hardened `review_soft_override_batch.py` output columns
6. Guarded `vision_2d.py` against missing DXF constants when Torch is unavailable

## Commands
```bash
TITLEBLOCK_OVERRIDE_ENABLED=false \
  .venv-graph/bin/python scripts/batch_analyze_dxf_local.py \
    --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
    --max-files 300 \
    --seed 20260125 \
    --output-dir "reports/experiments/20260125/dxf_batch_analysis_training_dxf_random110_graph2d_20260125"

.venv-graph/bin/python scripts/review_soft_override_batch.py \
  --input "reports/experiments/20260125/dxf_batch_analysis_training_dxf_random110_graph2d_20260125/batch_results.csv" \
  --output "reports/experiments/20260125/dxf_batch_analysis_training_dxf_random110_graph2d_20260125/soft_override_reviewed_20260125.csv" \
  --reviewer auto

.venv-graph/bin/python scripts/summarize_soft_override_review.py \
  --review-template "reports/experiments/20260125/dxf_batch_analysis_training_dxf_random110_graph2d_20260125/soft_override_reviewed_20260125.csv" \
  --summary-out "reports/experiments/20260125/dxf_batch_analysis_training_dxf_random110_graph2d_20260125/soft_override_review_summary_20260125.csv" \
  --correct-labels-out "reports/experiments/20260125/dxf_batch_analysis_training_dxf_random110_graph2d_20260125/soft_override_correct_label_counts_20260125.csv"

.venv-graph/bin/python - <<'PY'
import csv
from pathlib import Path

review_path = Path('reports/experiments/20260125/dxf_batch_analysis_training_dxf_random110_graph2d_20260125/soft_override_reviewed_20260125.csv')
conflict_path = Path('reports/experiments/20260125/dxf_batch_analysis_training_dxf_random110_graph2d_20260125/soft_override_conflicts_20260125.csv')

with review_path.open(newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

conflicts = [r for r in rows if (r.get('agree_with_graph2d') or '').strip().upper() in {'N', '?'}]

conflict_path.parent.mkdir(parents=True, exist_ok=True)
with conflict_path.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
    writer.writeheader()
    writer.writerows(conflicts)
PY

.venv-graph/bin/python - <<'PY'
import csv
from pathlib import Path

review_path = Path('reports/experiments/20260125/dxf_batch_analysis_training_dxf_random110_graph2d_20260125/soft_override_reviewed_20260125.csv')
summary_path = Path('reports/experiments/20260125/dxf_batch_analysis_training_dxf_random110_graph2d_20260125/filename_coverage_summary_20260125.csv')

with review_path.open(newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

matched = sum(1 for r in rows if (r.get('correct_label') or '').strip())
unknown = len(rows) - matched
coverage = matched / len(rows) if rows else 0

summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['total','matched_labels','unmatched','coverage'])
    writer.writeheader()
    writer.writerow({
        'total': len(rows),
        'matched_labels': matched,
        'unmatched': unknown,
        'coverage': f"{coverage:.4f}",
    })
PY
```

## Code changes
- `src/ml/vision_2d.py`
  - Define safe defaults for `DXF_NODE_DIM`/`DXF_EDGE_DIM` when Torch is missing.
- `scripts/review_soft_override_batch.py`
  - Ensure `reviewer` and `review_date` columns are included in CSV output.

## Outputs
- `reports/experiments/20260125/dxf_batch_analysis_training_dxf_random110_graph2d_20260125/batch_results.csv`
- `reports/experiments/20260125/dxf_batch_analysis_training_dxf_random110_graph2d_20260125/batch_low_confidence.csv`
- `reports/experiments/20260125/dxf_batch_analysis_training_dxf_random110_graph2d_20260125/summary.json`
- `reports/experiments/20260125/dxf_batch_analysis_training_dxf_random110_graph2d_20260125/label_distribution.csv`
- `reports/experiments/20260125/dxf_batch_analysis_training_dxf_random110_graph2d_20260125/soft_override_reviewed_20260125.csv`
- `reports/experiments/20260125/dxf_batch_analysis_training_dxf_random110_graph2d_20260125/soft_override_review_summary_20260125.csv`
- `reports/experiments/20260125/dxf_batch_analysis_training_dxf_random110_graph2d_20260125/soft_override_correct_label_counts_20260125.csv`
- `reports/experiments/20260125/dxf_batch_analysis_training_dxf_random110_graph2d_20260125/soft_override_conflicts_20260125.csv`
- `reports/experiments/20260125/dxf_batch_analysis_training_dxf_random110_graph2d_20260125/filename_coverage_summary_20260125.csv`

## Notes
- The dataset contains 110 DXFs, so the 300-sample target was capped.
- Graph2D is now active but agreement against filename-derived labels remains 0/110; conflicts list includes all rows.
