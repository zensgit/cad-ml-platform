#!/bin/bash
# B6.1c: Automated retraining pipeline (data flywheel).
#
# Checks the review queue for enough annotated samples, appends them
# to the training manifest, fine-tunes the model, evaluates, and
# optionally quantizes.
#
# Usage:
#   bash scripts/auto_retrain.sh
#   bash scripts/auto_retrain.sh --min-samples 100 --acc-gate 91.5
#
# Environment:
#   QUEUE_PATH          (default: data/review_queue/low_conf.csv)
#   BASE_MANIFEST       (default: data/graph_cache_v4_aug/cache_manifest_v4.csv)
#   BASE_CHECKPOINT     (default: models/graph2d_finetuned_24class_v4.pth)
#   OUTPUT_DIR          (default: models/)
#   MIN_REVIEWED        (default: 200)
#   ACC_GATE            (default: 91.5)

set -euo pipefail

QUEUE_PATH="${QUEUE_PATH:-data/review_queue/low_conf.csv}"
BASE_MANIFEST="${BASE_MANIFEST:-data/graph_cache_v4_aug/cache_manifest_v4.csv}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-models/graph2d_finetuned_24class_v4.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-models}"
MIN_REVIEWED="${1:-${MIN_REVIEWED:-200}}"
ACC_GATE="${2:-${ACC_GATE:-91.5}}"

VERSION=$(date +%Y%m%d_%H%M%S)
NEW_MANIFEST="data/manifests/manifest_${VERSION}.csv"
NEW_MODEL="${OUTPUT_DIR}/graph2d_v${VERSION}.pth"
NEW_INT8="${OUTPUT_DIR}/graph2d_v${VERSION}_int8.pth"

echo "============================================================"
echo "Auto-Retrain Pipeline (B6.1c)"
echo "============================================================"
echo "Queue:       ${QUEUE_PATH}"
echo "Base model:  ${BASE_CHECKPOINT}"
echo "Min samples: ${MIN_REVIEWED}"
echo "Acc gate:    ${ACC_GATE}%"
echo ""

# Step 1: Check reviewed sample count
REVIEWED=$(python3 -c "
from src.ml.low_conf_queue import LowConfidenceQueue
q = LowConfidenceQueue(queue_path='${QUEUE_PATH}')
print(len(q.reviewed_entries()))
" 2>/dev/null || echo "0")

echo "Step 1: Reviewed samples = ${REVIEWED}"
if [ "${REVIEWED}" -lt "${MIN_REVIEWED}" ]; then
    echo "  Not enough reviewed samples (${REVIEWED} < ${MIN_REVIEWED}). Exiting."
    exit 0
fi

# Step 2: Append reviewed samples to manifest
echo "Step 2: Appending reviewed samples to manifest..."
python3 scripts/append_reviewed_to_manifest.py \
    --queue "${QUEUE_PATH}" \
    --manifest "${BASE_MANIFEST}" \
    --output "${NEW_MANIFEST}" \
    --corrections-only

NEW_COUNT=$(wc -l < "${NEW_MANIFEST}")
echo "  New manifest: ${NEW_MANIFEST} (${NEW_COUNT} rows)"

# Step 3: Fine-tune
echo "Step 3: Fine-tuning from ${BASE_CHECKPOINT}..."
python3 scripts/finetune_graph2d_v2_augmented.py \
    --checkpoint "${BASE_CHECKPOINT}" \
    --manifest "${NEW_MANIFEST}" \
    --output "${NEW_MODEL}" \
    --epochs 40 --batch-size 32 \
    --encoder-lr 2e-5 --head-lr 2e-4 \
    --focal-gamma 2.0 --patience 12 --device cpu

# Step 4: Evaluate (gate check)
echo "Step 4: Evaluating..."
ACC=$(python3 -c "
import sys; sys.path.insert(0, '.')
from scripts.evaluate_graph2d_v2 import load_model
from scripts.finetune_graph2d_from_pretrained import CachedGraphDataset, collate_finetune
from torch.utils.data import DataLoader
import torch

model, label_map = load_model('${NEW_MODEL}')
model.eval()
dataset = CachedGraphDataset('data/graph_cache/cache_manifest.csv')
ds_inv = {v:k for k,v in dataset.label_map.items()}
inv = {v:k for k,v in label_map.items()}
loader = DataLoader(dataset, batch_size=64, collate_fn=collate_finetune)
correct = total = 0
with torch.no_grad():
    for b, l in loader:
        logits = model(b['x'], b['edge_index'], b.get('edge_attr'), b['batch'])
        preds = logits.argmax(dim=1)
        for p, la in zip(preds, l):
            if inv.get(p.item()) == ds_inv.get(la.item()): correct += 1
            total += 1
print(f'{correct/total*100:.1f}')
" 2>/dev/null)

echo "  New model accuracy: ${ACC}%"

# Gate check
PASS=$(python3 -c "print('yes' if float('${ACC}') >= float('${ACC_GATE}') else 'no')")
if [ "${PASS}" = "yes" ]; then
    echo "  PASS: ${ACC}% >= ${ACC_GATE}%"

    # Step 5: Quantize
    echo "Step 5: Quantizing..."
    python3 scripts/quantize_graph2d_model.py \
        --model "${NEW_MODEL}" \
        --output "${NEW_INT8}"

    echo ""
    echo "============================================================"
    echo "SUCCESS: Ready for deployment"
    echo "  Model:    ${NEW_MODEL}"
    echo "  INT8:     ${NEW_INT8}"
    echo "  Accuracy: ${ACC}%"
    echo ""
    echo "Deploy with:"
    echo "  export GRAPH2D_MODEL_PATH=${NEW_INT8}"
    echo "============================================================"
else
    echo "  FAIL: ${ACC}% < ${ACC_GATE}%"
    echo "  Model NOT deployed. Investigate training data quality."
    rm -f "${NEW_MODEL}"
    exit 1
fi
