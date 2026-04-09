# Eval Coarse Accuracy Validation 2026-03-07

## Goal
Expose both exact-label and coarse-label accuracy in DXF manifest evaluation so benchmark tracking can distinguish:
- exact fine-grained correctness
- coarse taxonomy correctness

## Scope
Updated:
- `scripts/eval_hybrid_dxf_manifest.py`
- `tests/unit/test_eval_hybrid_dxf_manifest.py`

## Delivered
- `eval_hybrid_dxf_manifest.py` now computes:
  - `exact_accuracy`
  - `coarse_accuracy`
- Existing `accuracy` field remains as the coarse-normalized metric for backward compatibility.
- Result rows now include explicit coarse outputs when present:
  - `coarse_part_type`
  - `coarse_fine_part_type`
  - `coarse_graph2d_label`
  - `coarse_filename_label`
  - `coarse_titleblock_label`
  - `coarse_hybrid_label`
  - `true_label_exact`
  - `true_label_coarse`

## Why
Previous reporting mixed coarse-normalized scoring with labels that looked fine-grained. That was enough for debugging but not enough for product or benchmark decisions. The new split makes it explicit when the system is:
- wrong at the fine label level but still right at the part-family level
- genuinely wrong even after coarse normalization

## Validation
Commands run:

```bash
python3 -m py_compile \
  scripts/eval_hybrid_dxf_manifest.py \
  tests/unit/test_eval_hybrid_dxf_manifest.py

flake8 \
  scripts/eval_hybrid_dxf_manifest.py \
  tests/unit/test_eval_hybrid_dxf_manifest.py \
  --max-line-length=100

pytest -q tests/unit/test_eval_hybrid_dxf_manifest.py
```

Results:
- `py_compile`: pass
- `flake8`: pass
- `pytest`: `4 passed`

## Notes
- This is reporting-only. It does not change inference decisions.
- Downstream dashboards and CI can adopt `exact_accuracy` incrementally while continuing to read `accuracy`.
