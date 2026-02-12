# DEV_GRAPH2D_TITLEBLOCK_DISTILLATION_TRAINING_20260210

## Goal
Train a Graph2D (DXF geometry) checkpoint using **titleblock-only** distillation so we can recover useful fine-label signals in the “anonymous DXF” scenario:
- upload filename is masked (no filename signal), and
- titleblock extraction is disabled in production (or not trusted),
while still leveraging titleblock text as a **teacher signal** during training.

This is the pragmatic bridge between:
- short-term accuracy (filename/titleblock rule fusion), and
- medium-term robustness (geometry can stand on its own when text is unavailable).

## Key Implementation Changes
- Distillation teacher can consume DXF bytes:
  - `src/ml/knowledge_distillation.py`
    - `TeacherModel.generate_soft_labels(..., file_bytes_list=...)` added.
    - `teacher=titleblock` parses entities via `src.utils.dxf_io.read_dxf_entities_from_bytes()` and runs `TitleBlockClassifier.predict(entities)`.
    - Hybrid teacher now accepts `file_bytes` so it can use titleblock/process signals when available.
    - If the teacher predicts a label outside the student label-map and `other` exists, fall back to `other` with low confidence (max `0.25`).
- Manifest dataset surfaces `file_path` so training can read DXF bytes:
  - `src/ml/train/dataset_2d.py` (`DXFManifestDataset.__getitem__` returns `relative_path` + `file_path`).
- Training script wiring:
  - `scripts/train_2d_graph.py`
    - `--teacher` now supports `titleblock`.
    - Added `--distill-mask-filename` to prevent the teacher from “cheating” via filename signal.
    - Collate returns `file_path`; distillation caches teacher logits by file path to avoid repeated DXF parsing.
- Downstream debug metadata:
  - `src/ml/vision_2d.py` Graph2D prediction now returns `label_map_size`, `top2_confidence`, and `margin`.
  - `src/ml/hybrid_classifier.py` applies a conservative dynamic Graph2D min-confidence when label-map size is large.
- Unit coverage:
  - `tests/unit/test_knowledge_distillation_teacher.py` verifies titleblock teacher uses `file_bytes_list` and unknown-label fallback to `other`.

## Run
Dataset:
- `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123` (110 DXF; ODA-normalized)

Build a weak-label manifest (from filename extraction) and clean it:
```bash
PYTHONPYCACHEPREFIX=/tmp/pycache \
MPLCONFIGDIR=/tmp/mplconfig \
XDG_CACHE_HOME=/tmp/xdg-cache \
  .venv-graph/bin/python scripts/build_dxf_label_manifest.py \
    --input-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123" \
    --recursive \
    --output-csv reports/experiments/20260210/graph2d_distill_titleblock_teacher/manifest_raw.csv

PYTHONPYCACHEPREFIX=/tmp/pycache \
MPLCONFIGDIR=/tmp/mplconfig \
XDG_CACHE_HOME=/tmp/xdg-cache \
  .venv-graph/bin/python scripts/clean_dxf_label_manifest.py \
    --input-csv reports/experiments/20260210/graph2d_distill_titleblock_teacher/manifest_raw.csv \
    --min-count 2 \
    --output-csv reports/experiments/20260210/graph2d_distill_titleblock_teacher/manifest_min2.csv
```

Train (EdgeSAGE) with titleblock-only distillation:
```bash
PYTHONPYCACHEPREFIX=/tmp/pycache \
MPLCONFIGDIR=/tmp/mplconfig \
XDG_CACHE_HOME=/tmp/xdg-cache \
  .venv-graph/bin/python scripts/train_2d_graph.py \
    --manifest reports/experiments/20260210/graph2d_distill_titleblock_teacher/manifest_min2.csv \
    --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123" \
    --model edge_sage \
    --epochs 20 \
    --loss focal \
    --distill \
    --teacher titleblock \
    --distill-mask-filename \
    --output models/graph2d_training_dxf_oda_titleblock_distill_20260210.pth \
    --save-best
```

Eval (per-class metrics + overall):
```bash
PYTHONPYCACHEPREFIX=/tmp/pycache \
MPLCONFIGDIR=/tmp/mplconfig \
XDG_CACHE_HOME=/tmp/xdg-cache \
  .venv-graph/bin/python scripts/eval_2d_graph.py \
    --manifest reports/experiments/20260210/graph2d_distill_titleblock_teacher/manifest_min2.csv \
    --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123" \
    --model-path models/graph2d_training_dxf_oda_titleblock_distill_20260210.pth \
    --output-dir reports/experiments/20260210/graph2d_distill_titleblock_teacher
```

Diagnose the trained checkpoint using weak supervision (filename labels):
```bash
PYTHONPYCACHEPREFIX=/tmp/pycache \
MPLCONFIGDIR=/tmp/mplconfig \
XDG_CACHE_HOME=/tmp/xdg-cache \
  .venv-graph/bin/python scripts/diagnose_graph2d.py \
    --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf_oda_20260123" \
    --model-path models/graph2d_training_dxf_oda_titleblock_distill_20260210.pth \
    --output-dir reports/experiments/20260210/graph2d_diagnose_titleblock_distilled \
    --max-files 110 \
    --seed 22
```

Outputs:
- Model (local artifact): `models/graph2d_training_dxf_oda_titleblock_distill_20260210.pth`
- Train/eval:
  - `reports/experiments/20260210/graph2d_distill_titleblock_teacher/train_stdout.txt`
  - `reports/experiments/20260210/graph2d_distill_titleblock_teacher/eval_metrics.csv`
- Diagnose:
  - `reports/experiments/20260210/graph2d_diagnose_titleblock_distilled/summary.json`

Note:
- Per-file outputs and manifests contain original filenames and are gitignored via:
  - `reports/experiments/20260210/graph2d_distill_titleblock_teacher/.gitignore`
  - `reports/experiments/20260210/graph2d_diagnose_titleblock_distilled/.gitignore`

## Results Summary
Training (from `train_stdout.txt`):
- Classes: `47` (min count per class `1`, max `4`)
- Best `val_acc=0.319` (epoch 16)

Eval (from `eval_metrics.csv`):
- Overall accuracy: `0.298` (`14/47`)
- Top-2 accuracy: `0.553`
- Macro-F1: `0.237`, Weighted-F1: `0.237`

Diagnose (from `graph2d_diagnose_titleblock_distilled/summary.json`):
- Weak-supervised accuracy: `0.2727` (filename labels as “true”)
- Confidence distribution: `p50=0.0406`, `p90=0.0909`

Interpretation:
- The checkpoint is **not production-grade** for fine-label classification on anonymous DXF (confidence is still low due to many-class softmax).
- But it breaks the previous “single-label collapse” failure mode and provides a usable geometry signal for:
  - soft override suggestions,
  - human-in-the-loop review,
  - and as a stepping stone toward temperature calibration + additional geometry constraints.

