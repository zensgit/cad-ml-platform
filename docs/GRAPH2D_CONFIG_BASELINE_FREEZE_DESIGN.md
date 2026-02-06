# Graph2D Config + Baseline Freeze Design

## Summary

This update makes Graph2D training/evaluation and hybrid fusion behavior more reproducible:

1. Add YAML-backed runtime config for hybrid classifier (`config/hybrid_classifier.yaml`).
2. Add YAML-backed defaults for Graph2D train/eval scripts.
3. Add a baseline freeze script to package checkpoints with metadata.
4. Add a worktree bootstrap script for parallel branch development.

## Interfaces and Behavior Changes

### Hybrid runtime config

- New file: `config/hybrid_classifier.yaml`
- Existing module `src/ml/hybrid_config.py` now loads:
  - defaults (dataclass)
  - YAML config file
  - env overrides (highest priority)
- Existing class `src/ml/hybrid_classifier.py` now resolves:
  - weights/thresholds from config
  - optional env overrides
  - filename synonyms path from config/env

### Graph2D train/eval script config

- New files:
  - `config/graph2d_training.yaml`
  - `config/graph2d_eval.yaml`
- `scripts/train_2d_graph.py`:
  - new `--config` (default `config/graph2d_training.yaml`)
  - YAML section key: `train_2d_graph`
  - CLI still overrides YAML
  - new optional DXF sampling overrides:
    - `--dxf-max-nodes`
    - `--dxf-sampling-strategy`
    - `--dxf-sampling-seed`
    - `--dxf-text-priority-ratio`
- `scripts/eval_2d_graph.py`:
  - new `--config` (default `config/graph2d_eval.yaml`)
  - YAML section key: `eval_2d_graph`
  - same DXF sampling override flags

### Baseline freeze

- New script: `scripts/freeze_graph2d_baseline.py`
- Input:
  - checkpoint path
  - optional eval metrics CSV
  - baseline name / notes / output dir
- Output:
  - copied checkpoint under a timestamped baseline directory
  - `metadata.json` including hashes, source path, optional metric summary, git SHA

### Worktree bootstrap

- New script: `scripts/bootstrap_worktree.sh`
- Purpose: quickly create branch-specific worktrees for parallel development windows.

## Operational Defaults

- Hybrid config path:
  - default: `config/hybrid_classifier.yaml`
  - override: `HYBRID_CONFIG_PATH`
- Precedence for key runtime values:
  - environment variable > YAML > code default

## Validation Scope

- New unit tests:
  - `tests/unit/test_hybrid_config_loader.py`
  - `tests/unit/test_hybrid_classifier_config_integration.py`
  - `tests/unit/test_graph2d_script_config.py`
  - `tests/unit/test_freeze_graph2d_baseline.py`
- Existing related tests rerun:
  - `tests/unit/test_hybrid_classifier.py`
  - `tests/unit/test_filename_classifier.py`
  - `tests/unit/test_importance_sampling.py`
