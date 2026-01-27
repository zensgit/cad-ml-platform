# DEV_GRAPH2D_RETRAIN_20260126

## Goal
Retrain Graph2D on filename-synonym labels aligned to the 144-label taxonomy, then evaluate and validate via automated batch review (no manual DXF inspection).

## Scope
- Dataset: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf`
- Available DXF files: 110
- Labeling: `FilenameClassifier` + `label_synonyms_template.json`

## Work performed
1. Build a training manifest using filename-based canonical labels.
2. Train a Graph2D GCN model (balanced sampling + focal loss + augmentation).
3. Evaluate the checkpoint on a stratified validation split.
4. Run a Graph2D-enabled batch analysis with the new checkpoint and auto-review agreement vs filename labels.

## Commands
```bash
# 1) Build manifest from filename + synonyms
.venv-graph/bin/python - <<'PY'
import csv
import json
from collections import Counter
from pathlib import Path
from src.ml.filename_classifier import get_filename_classifier

dxf_dir = Path("/Users/huazhou/Downloads/训练图纸/训练图纸_dxf")
output_dir = Path("reports/experiments/20260126/graph2d_retrain_manifest_20260126")
output_dir.mkdir(parents=True, exist_ok=True)

manifest_path = output_dir / "dxf_manifest_filename_synonyms_20260126.csv"
counts_path = output_dir / "dxf_manifest_label_counts_20260126.csv"
summary_path = output_dir / "dxf_manifest_summary_20260126.json"
unmatched_path = output_dir / "dxf_manifest_unmatched_20260126.txt"

files = sorted(dxf_dir.glob("*.dxf"))
classifier = get_filename_classifier()

rows = []
counts = Counter()
unmatched = []
for path in files:
    pred = classifier.predict(path.name)
    label = pred.get("label")
    if not label:
        unmatched.append(path.name)
        continue
    counts[label] += 1
    rows.append({
        "file_name": path.name,
        "label_cn": label,
        "label_confidence": f"{float(pred.get('confidence') or 0.0):.3f}",
        "match_type": pred.get("match_type"),
        "extracted_name": pred.get("extracted_name"),
    })

with manifest_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=[
        "file_name", "label_cn", "label_confidence", "match_type", "extracted_name"
    ])
    writer.writeheader()
    writer.writerows(rows)

with counts_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["label_cn", "count"])
    writer.writeheader()
    for label, count in counts.most_common():
        writer.writerow({"label_cn": label, "count": count})

summary = {
    "total_files": len(files),
    "matched": len(rows),
    "unmatched": len(unmatched),
    "coverage": round(len(rows) / len(files), 4) if files else 0.0,
    "unique_labels": len(counts),
}
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

unmatched_path.write_text("\n".join(unmatched), encoding="utf-8")
PY

# 2) Train Graph2D (CPU fallback due to MPS loss issue)
DISABLE_MODEL_SOURCE_CHECK=1 \
  .venv-graph/bin/python scripts/train_2d_graph.py \
    --manifest "reports/experiments/20260126/graph2d_retrain_manifest_20260126/dxf_manifest_filename_synonyms_20260126.csv" \
    --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
    --epochs 5 \
    --batch-size 4 \
    --hidden-dim 64 \
    --model gcn \
    --loss focal \
    --class-weighting sqrt \
    --sampler balanced \
    --augment \
    --augment-prob 0.6 \
    --augment-scale 0.05 \
    --device cpu \
    --output "models/graph2d_parts_filename_synonyms_20260126.pth"

# 3) Evaluate checkpoint
DISABLE_MODEL_SOURCE_CHECK=1 \
  .venv-graph/bin/python scripts/eval_2d_graph.py \
    --manifest "reports/experiments/20260126/graph2d_retrain_manifest_20260126/dxf_manifest_filename_synonyms_20260126.csv" \
    --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
    --checkpoint "models/graph2d_parts_filename_synonyms_20260126.pth" \
    --output-metrics "reports/experiments/20260126/graph2d_retrain_eval_metrics_20260126.csv" \
    --output-errors "reports/experiments/20260126/graph2d_retrain_eval_errors_20260126.csv" \
    --seed 20260126 \
    --val-split 0.2 \
    --split-strategy stratified

# 4) Batch analyze with new model + auto review
GRAPH2D_MODEL_PATH="models/graph2d_parts_filename_synonyms_20260126.pth" \
GRAPH2D_DRAWING_TYPE_LABELS="零件图,机械制图,装配图,练习零件图,原理图,模板" \
TITLEBLOCK_OVERRIDE_ENABLED=false \
DISABLE_MODEL_SOURCE_CHECK=1 \
  .venv-graph/bin/python scripts/batch_analyze_dxf_local.py \
    --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
    --max-files 300 \
    --seed 20260126 \
    --output-dir "reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_retrained_20260126"

.venv-graph/bin/python scripts/review_soft_override_batch.py \
  --input "reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_retrained_20260126/batch_results.csv" \
  --output "reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_retrained_20260126/soft_override_reviewed_20260126.csv" \
  --reviewer auto

.venv-graph/bin/python scripts/summarize_soft_override_review.py \
  --review-template "reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_retrained_20260126/soft_override_reviewed_20260126.csv" \
  --summary-out "reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_retrained_20260126/soft_override_review_summary_20260126.csv" \
  --correct-labels-out "reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_retrained_20260126/soft_override_correct_label_counts_20260126.csv"
```

## Outputs
### Manifest
- `reports/experiments/20260126/graph2d_retrain_manifest_20260126/dxf_manifest_filename_synonyms_20260126.csv`
- `reports/experiments/20260126/graph2d_retrain_manifest_20260126/dxf_manifest_label_counts_20260126.csv`
- `reports/experiments/20260126/graph2d_retrain_manifest_20260126/dxf_manifest_summary_20260126.json`
- `reports/experiments/20260126/graph2d_retrain_manifest_20260126/dxf_manifest_unmatched_20260126.txt`

### Training + Eval
- `models/graph2d_parts_filename_synonyms_20260126.pth`
- `reports/experiments/20260126/graph2d_retrain_eval_metrics_20260126.csv`
- `reports/experiments/20260126/graph2d_retrain_eval_errors_20260126.csv`

### Batch review (new model)
- `reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_retrained_20260126/batch_results.csv`
- `reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_retrained_20260126/batch_low_confidence.csv`
- `reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_retrained_20260126/summary.json`
- `reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_retrained_20260126/label_distribution.csv`
- `reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_retrained_20260126/soft_override_reviewed_20260126.csv`
- `reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_retrained_20260126/soft_override_review_summary_20260126.csv`
- `reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_retrained_20260126/soft_override_correct_label_counts_20260126.csv`
- `reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_retrained_20260126/soft_override_conflicts_20260126.csv`
- `reports/experiments/20260126/dxf_batch_analysis_training_dxf_random110_graph2d_retrained_20260126/filename_coverage_summary_20260126.csv`

## Notes
- MPS training failed due to `cross_entropy_loss` placeholder storage; training was rerun on CPU.
- Dataset coverage is high, but only 47 unique labels are present in this subset, limiting generalization.
