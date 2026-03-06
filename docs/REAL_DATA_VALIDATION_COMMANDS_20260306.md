# Real Data Validation Commands

## Goal

Run the newly added history-sequence and Graph2D tooling against real local data after the PR stack merges or in isolated feature branches.

## History Sequence

### 1. Build prototypes from a labeled manifest

```bash
python3 scripts/build_history_sequence_prototypes.py \
  --manifest <manifest.json-or-csv> \
  --label-source manifest \
  --output reports/experiments/$(date +%Y%m%d)/history_sequence_real/prototypes.json
```

Expected outputs:

- `prototypes.json`
- stdout summary with `input_pairs`, `used_samples`, `label_count`

### 2. Evaluate classifier on labeled `.h5`

```bash
python3 scripts/eval_history_sequence_classifier.py \
  --manifest <manifest.json-or-csv> \
  --label-source manifest \
  --prototypes-path reports/experiments/$(date +%Y%m%d)/history_sequence_real/prototypes.json \
  --min-seq-len 4 \
  --output-dir reports/experiments/$(date +%Y%m%d)/history_sequence_real/eval
```

Check:

- `summary.json`
- `results.csv`
- `coverage`
- `accuracy_overall`
- `macro_f1_overall`
- `low_conf_rate`

### 3. Tune token/bigram weights

```bash
python3 scripts/tune_history_sequence_weights.py \
  --manifest <manifest.json-or-csv> \
  --label-source manifest \
  --prototypes-path reports/experiments/$(date +%Y%m%d)/history_sequence_real/prototypes.json \
  --objective macro_f1_overall \
  --token-weight-grid 0.5,1.0,1.5 \
  --bigram-weight-grid 0.0,0.5,1.0,1.5 \
  --output-dir reports/experiments/$(date +%Y%m%d)/history_sequence_real/tuning
```

Check:

- `best_config.json`
- `weight_grid.csv`
- `recommended_history_sequence.env`

## Graph2D Training

### 1. Emit training metrics artifact

```bash
python3 scripts/train_2d_graph.py \
  --config config/graph2d_training.yaml \
  --manifest <manifest.csv> \
  --dxf-dir <dxf-dir> \
  --output models/graph2d_real_validation.pth \
  --metrics-out reports/experiments/$(date +%Y%m%d)/graph2d_train/metrics.json
```

Check:

- `metrics.json`
- `best_epoch`
- `final_val_acc`
- `final_loss`
- `epochs_ran`
- `class_stats`
- `sampling_overrides`
- `epoch_history`

## Graph2D Evaluation

### 1. Directory diagnosis with low-confidence stats

```bash
python3 scripts/diagnose_graph2d_on_dxf_dir.py \
  --dxf-dir <dxf-dir> \
  --model-path models/graph2d_real_validation.pth \
  --manifest-csv <manifest.csv> \
  --low-conf-threshold 0.2 \
  --output-dir reports/experiments/$(date +%Y%m%d)/graph2d_diagnose
```

Check:

- `summary.json`
- `predictions.csv`
- `confidence.low_conf_count`
- `confidence.low_conf_rate`
- `accuracy`

### 2. End-to-end eval history generation

```bash
HISTORY_SEQUENCE_EVAL_ENABLE=true \
HISTORY_SEQUENCE_EVAL_MANIFEST=<manifest.json-or-csv> \
HISTORY_SEQUENCE_PROTOTYPES_OUT=reports/experiments/$(date +%Y%m%d)/history_sequence_real/prototypes.json \
HISTORY_SEQUENCE_TUNE_ENABLE=true \
bash scripts/eval_with_history.sh
```

Check:

- OCR eval record under `reports/eval_history/`
- history-sequence eval record under `reports/eval_history/`
- generated env file with selected history weights

## Safety Notes

- Run these commands in a clean worktree, not the dirty main worktree.
- Keep output under `reports/experiments/<date>/...` to avoid mixing runs.
- If using local private data, do not add generated manifests or artifacts to Git.
