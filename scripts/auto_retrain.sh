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

# Step 1b: Provenance check — count human-verified and eligible rows
HUMAN_VERIFIED=$(python3 -c "
import csv, sys
total = verified = eligible = 0
try:
    with open('${QUEUE_PATH}', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if not row.get('reviewed_label', '').strip():
                continue
            total += 1
            hv = str(row.get('human_verified', '')).strip().lower()
            if hv not in ('', 'false', '0', 'no'):
                verified += 1
            eft = str(row.get('eligible_for_training', '')).strip().lower()
            if eft not in ('', 'false', '0', 'no'):
                eligible += 1
    print(f'{total},{verified},{eligible}')
except Exception as e:
    print(f'0,0,0')
" 2>/dev/null || echo "0,0,0")

IFS=',' read -r HV_TOTAL HV_VERIFIED HV_ELIGIBLE <<< "${HUMAN_VERIFIED}"
echo "Step 1b: Reviewed: ${HV_TOTAL} total, ${HV_VERIFIED} human-verified, ${HV_ELIGIBLE} eligible"

if [ "${HV_VERIFIED}" -lt "${MIN_REVIEWED}" ]; then
    echo "  Not enough human-verified samples (${HV_VERIFIED} < ${MIN_REVIEWED}). Exiting."
    echo "  Note: Total reviewed is ${REVIEWED}, but provenance gate requires ${MIN_REVIEWED} human-verified."
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

# Step 2b: Ensure all manifest rows have cache_path (generate .pt for new DXF files)
echo "Step 2b: Generating graph cache for new samples..."
MISSING_CACHE=$(python3 -c "
import csv
missing = 0
with open('${NEW_MANIFEST}', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        if not row.get('cache_path', '').strip():
            missing += 1
print(missing)
" 2>/dev/null || echo "0")

if [ "${MISSING_CACHE}" -gt 0 ]; then
    echo "  ${MISSING_CACHE} rows missing cache_path — running preprocess..."
    python3 scripts/preprocess_dxf_to_graphs.py \
        --manifest "${NEW_MANIFEST}" \
        --output-dir data/graph_cache \
        --skip-existing

    # Backfill: merge generated cache_path into NEW_MANIFEST
    echo "  Backfilling cache_path into manifest..."
    python3 -c "
import csv, hashlib, os
from pathlib import Path

manifest = '${NEW_MANIFEST}'
cache_dir = 'data/graph_cache'

# Read preprocess output manifest (has file_path → cache_path mapping)
cache_map = {}
cache_manifest = Path(cache_dir) / 'cache_manifest.csv'
if cache_manifest.exists():
    with open(cache_manifest, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            fp = row.get('file_path', '').strip()
            cp = row.get('cache_path', '').strip()
            if fp and cp:
                cache_map[fp] = cp

# Also check by file_path hash (preprocess uses md5 of file_path as key)
def hash_path(fp):
    return os.path.join(cache_dir, hashlib.md5(fp.encode()).hexdigest() + '.pt')

# Read and update manifest
rows = []
filled = 0
with open(manifest, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        if not row.get('cache_path', '').strip():
            fp = row.get('file_path', '').strip()
            # Try cache_map first, then hash-based lookup
            cp = cache_map.get(fp, '')
            if not cp:
                candidate = hash_path(fp)
                if os.path.exists(candidate):
                    cp = candidate
            if cp:
                row['cache_path'] = cp
                filled += 1
        rows.append(row)

# Write back
with open(manifest, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f'  Backfilled {filled} cache_path entries')
remaining = sum(1 for r in rows if not r.get('cache_path', '').strip())
if remaining > 0:
    print(f'FATAL: {remaining} rows still missing cache_path after backfill')
    import sys; sys.exit(1)
" 2>/dev/null
    BACKFILL_RC=$?
    if [ "${BACKFILL_RC}" -ne 0 ]; then
        echo "  FATAL: backfill failed — some samples have no cache. Aborting."
        exit 1
    fi
    echo "  Cache generation + backfill complete."
else
    echo "  All rows have cache_path — skipping preprocess."
fi

# Step 3: Fine-tune
echo "Step 3: Fine-tuning from ${BASE_CHECKPOINT}..."
GOLDEN_VAL="${GOLDEN_VAL:-data/manifests/golden_val_set.csv}"

python3 scripts/finetune_graph2d_v2_augmented.py \
    --checkpoint "${BASE_CHECKPOINT}" \
    --manifest "${NEW_MANIFEST}" \
    --val-manifest "${GOLDEN_VAL}" \
    --output "${NEW_MODEL}" \
    --epochs 40 --batch-size 32 \
    --encoder-lr 2e-5 --head-lr 2e-4 \
    --focal-gamma 2.0 --patience 12 --device cpu

# Step 4: Evaluate on GOLDEN validation set (not training data!)
echo "Step 4: Evaluating on golden validation set (${GOLDEN_VAL})..."
ACC=$(python3 -c "
import sys; sys.path.insert(0, '.')
from scripts.evaluate_graph2d_v2 import load_model
from scripts.finetune_graph2d_from_pretrained import CachedGraphDataset, collate_finetune
from torch.utils.data import DataLoader
import torch

model, label_map = load_model('${NEW_MODEL}')
model.eval()
dataset = CachedGraphDataset('${GOLDEN_VAL}')
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
