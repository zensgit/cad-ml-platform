# DEV_HYBRID_CLASSIFIER_FUSION_CONFIDENCE_RESCALING_20260213

## Summary

Adjusted `HybridClassifier` fusion confidence so it better reflects the strongest supporting signal
instead of being scaled down by fusion weights.

This matters because `HybridClassifier` confidence is used by `/api/v1/analyze/` hybrid auto-override
logic; previously, even when `titleblock` was highly confident and `graph2d` agreed on the same
label, the fused confidence could be ~0.3 due to small fusion weights, preventing override of
placeholder base labels.

## Change

- `src/ml/hybrid_classifier.py`
  - Fusion label selection remains weight-based.
  - Fusion **confidence** now uses:
    - `base_conf = max(confidences_of_sources_supporting_best_label)`
    - `+ small bonus` when 2+ sources support the same label (capped)

## Validation

Executed:

```bash
.venv/bin/python -m pytest \
  tests/unit/test_hybrid_classifier_graph2d_guardrails.py \
  tests/unit/test_hybrid_classifier_coverage.py \
  tests/integration/test_analyze_dxf_hybrid_override.py \
  -v

.venv/bin/python scripts/batch_analyze_dxf_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --max-files 30 \
  --mask-filename \
  --output-dir /tmp/dxf_batch_eval_masked_20260213_torch_v2
```

Key results (`/tmp/dxf_batch_eval_masked_20260213_torch_v2/summary.json`):

- `weak_labels.covered_rate = 1.0` (`30/30`)
- `weak_labels.accuracy.fine_part_type.accuracy = 1.0` (`30/30`)
- `weak_labels.accuracy.final_part_type.accuracy = 0.8667` (`26/30`)
  - Improved from `0.7667` (`23/30`) in `/tmp/dxf_batch_eval_masked_20260213_torch/summary.json`
- `weak_labels.accuracy.graph2d_label.accuracy = 0.3` (`9/30`)
- `hybrid.confidence.mean_all = 0.855` (previously ~`0.795` in the masked run)
- `confidence_buckets.gte_0_8 = 30` (no low-confidence final labels in this batch)

## Notes / Caveats

- Weak labels are derived from the **original filenames** via `FilenameClassifier` and are not
  ground truth; treat these as regression indicators.
- Even with `--mask-filename`, titleblock text remains available inside the DXF content and is a
  strong signal in this dataset.
