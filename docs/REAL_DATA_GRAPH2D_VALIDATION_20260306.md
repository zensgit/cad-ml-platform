# Real Data Graph2D Validation 2026-03-06

## Scope

This validation run executed the newly merged Graph2D training metrics and
diagnostic tooling against a real local DXF dataset in a clean post-merge
worktree.

History-sequence real-data validation was not executed because no local `.h5`
dataset was available on this machine at validation time.

## Data Used

- Manifest: `reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_ODA_NORMALIZED_CLEANED_20260123.csv`
- DXF directory: local private `训练图纸_dxf_oda_20260123`
- Dataset size: `110` files
- Label count: `8`

Label distribution:

- `传动件`: `21`
- `设备`: `19`
- `罐体`: `18`
- `other`: `14`
- `轴承件`: `11`
- `法兰`: `11`
- `罩盖件`: `8`
- `过滤组件`: `8`

## Commands Run

### Training

```bash
python3 scripts/train_2d_graph.py \
  --config config/graph2d_training.yaml \
  --manifest reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_ODA_NORMALIZED_CLEANED_20260123.csv \
  --dxf-dir /path/to/local/训练图纸_dxf_oda_20260123 \
  --epochs 3 \
  --output models/graph2d_real_validation_oda110_20260306.pth \
  --metrics-out reports/experiments/20260306/graph2d_real_validation_oda110/metrics.json
```

### Default diagnosis

```bash
python3 scripts/diagnose_graph2d_on_dxf_dir.py \
  --dxf-dir /path/to/local/训练图纸_dxf_oda_20260123 \
  --model-path models/graph2d_real_validation_oda110_20260306.pth \
  --manifest-csv reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_ODA_NORMALIZED_CLEANED_20260123.csv \
  --low-conf-threshold 0.2 \
  --output-dir reports/experiments/20260306/graph2d_real_validation_oda110/diagnose_default \
  --max-files 110
```

### Blind diagnosis

```bash
python3 scripts/diagnose_graph2d_on_dxf_dir.py \
  --dxf-dir /path/to/local/训练图纸_dxf_oda_20260123 \
  --model-path models/graph2d_real_validation_oda110_20260306.pth \
  --manifest-csv reports/experiments/20260123/MECH_TRAINING_DXF_LABEL_MANIFEST_ODA_NORMALIZED_CLEANED_20260123.csv \
  --low-conf-threshold 0.2 \
  --output-dir reports/experiments/20260306/graph2d_real_validation_oda110/diagnose_blind \
  --max-files 110 \
  --strip-text-entities \
  --mask-filename
```

## Training Result

- `epochs_requested`: `3`
- `epochs_ran`: `3`
- `train_size`: `84`
- `val_size`: `26`
- `best_epoch`: `2`
- `best_val_acc`: `0.1538`
- `final_val_acc`: `0.0769`
- `final_loss`: `0.3988`
- `class imbalance ratio`: `2.67`

Epoch history:

1. epoch `1`: loss `0.4053`, val_acc `0.1154`
2. epoch `2`: loss `0.3870`, val_acc `0.1538`
3. epoch `3`: loss `0.3988`, val_acc `0.0769`

## Diagnosis Result

### Default signal path

- `sampled_files`: `110`
- `accuracy`: `0.1182`
- `low_conf_threshold`: `0.2`
- `low_conf_count`: `110`
- `low_conf_rate`: `1.0`
- `p50 confidence`: `0.1412`
- `p90 confidence`: `0.1457`
- predicted labels used: `3`

Top predicted labels:

- `过滤组件`: `76`
- `轴承件`: `32`
- `法兰`: `2`

### Blind path (`strip_text_entities + mask_filename`)

- `sampled_files`: `110`
- `accuracy`: `0.1273`
- `low_conf_threshold`: `0.2`
- `low_conf_count`: `110`
- `low_conf_rate`: `1.0`
- `p50 confidence`: `0.1404`
- `p90 confidence`: `0.1441`
- predicted labels used: `3`

Top predicted labels:

- `过滤组件`: `56`
- `轴承件`: `39`
- `法兰`: `15`

## Findings

1. Standalone Graph2D remains weak on this real dataset.
   - Best validation accuracy only reached `0.1538`.
   - End-to-end diagnosis accuracy stayed near random-like low performance.

2. Confidence is uniformly too low.
   - Both default and blind runs produced `low_conf_rate = 1.0`.
   - This confirms the new low-confidence reporting logic is useful in practice:
     the model should be rejected or downweighted for all samples in this run.

3. Prediction collapse remains visible.
   - Ground truth has `8` labels.
   - Model predictions collapsed into only `3` labels in both runs.

4. Removing text and filename signals did not materially worsen results.
   - Default accuracy: `0.1182`
   - Blind accuracy: `0.1273`
   - This suggests the current trained checkpoint is not yet extracting strong
     usable signal from either geometric structure or auxiliary textual hints on
     this dataset.

5. Production implication:
   - Graph2D should not be trusted as a standalone classifier on this dataset.
   - It is currently more suitable as a weak signal feeding hybrid fusion,
     rejection logic, and review-pack generation.

## History-Sequence Blocker

No `.h5` files were found under the local data roots inspected during this run,
so the following real-data validations remain blocked:

- `scripts/build_history_sequence_prototypes.py`
- `scripts/eval_history_sequence_classifier.py`
- `scripts/tune_history_sequence_weights.py`

To unblock that path, prepare a labeled local manifest pointing to real `.h5`
files and rerun the commands documented in
`docs/REAL_DATA_VALIDATION_COMMANDS_20260306.md`.

## Local-Only Artifacts

The following artifacts were generated locally and intentionally not committed:

- `reports/experiments/20260306/graph2d_real_validation_oda110/metrics.json`
- `reports/experiments/20260306/graph2d_real_validation_oda110/diagnose_default/summary.json`
- `reports/experiments/20260306/graph2d_real_validation_oda110/diagnose_blind/summary.json`
- `models/graph2d_real_validation_oda110_20260306.pth`

## Recommended Next Steps

1. Keep Graph2D behind hybrid gating and low-confidence rejection for real DXF.
2. Use this dataset to compare filename/titleblock/hybrid branches instead of
   expecting Graph2D-only classification to carry the task.
3. If standalone Graph2D is still required, improve the dataset before model
   complexity:
   - add more classes with repeated samples,
   - reduce label ambiguity,
   - review whether `other` should be split or downweighted.
4. Prepare real `.h5` data to validate the history-sequence branch end-to-end.
