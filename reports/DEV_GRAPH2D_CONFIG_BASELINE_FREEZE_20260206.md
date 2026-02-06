# DEV_GRAPH2D_CONFIG_BASELINE_FREEZE_20260206

## Scope

- Implement config-driven Graph2D train/eval defaults.
- Implement config-driven hybrid classifier runtime defaults.
- Add Graph2D baseline freeze utility.
- Add worktree bootstrap utility for parallel development.

## Code Changes

- Added config files:
  - `config/hybrid_classifier.yaml`
  - `config/graph2d_training.yaml`
  - `config/graph2d_eval.yaml`
- Updated:
  - `src/ml/hybrid_config.py`
  - `src/ml/hybrid_classifier.py`
  - `scripts/train_2d_graph.py`
  - `scripts/eval_2d_graph.py`
  - `Makefile`
- Added scripts:
  - `scripts/freeze_graph2d_baseline.py`
  - `scripts/bootstrap_worktree.sh`
- Added tests:
  - `tests/unit/test_hybrid_config_loader.py`
  - `tests/unit/test_hybrid_classifier_config_integration.py`
  - `tests/unit/test_graph2d_script_config.py`
  - `tests/unit/test_freeze_graph2d_baseline.py`
- Added design doc:
  - `docs/GRAPH2D_CONFIG_BASELINE_FREEZE_DESIGN.md`

## Verification Commands

```bash
python3 -m black src/ml/hybrid_config.py src/ml/hybrid_classifier.py scripts/train_2d_graph.py scripts/eval_2d_graph.py scripts/freeze_graph2d_baseline.py tests/unit/test_hybrid_config_loader.py tests/unit/test_hybrid_classifier_config_integration.py tests/unit/test_graph2d_script_config.py tests/unit/test_freeze_graph2d_baseline.py
python3 -m pytest tests/unit/test_hybrid_config_loader.py tests/unit/test_hybrid_classifier_config_integration.py tests/unit/test_graph2d_script_config.py tests/unit/test_freeze_graph2d_baseline.py -q
python3 scripts/train_2d_graph.py --help
python3 scripts/eval_2d_graph.py --help
python3 -m pytest tests/unit/test_hybrid_classifier.py tests/unit/test_filename_classifier.py tests/unit/test_importance_sampling.py -q
python3 scripts/freeze_graph2d_baseline.py --checkpoint /tmp/graph2d_dummy.pth --name smoke --output-dir /tmp/graph2d_baselines_smoke
```

## Verification Results

- Formatting: success.
- New tests: `7 passed`.
- Existing related tests: `40 passed`.
- Train/Eval scripts expose new `--config` and DXF sampling override options.
- Baseline freeze script successfully generated a timestamped bundle and metadata JSON in smoke run.
