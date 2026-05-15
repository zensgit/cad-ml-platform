# CAD ML Manufacturing Evidence Detail Quality Verification

Date: 2026-05-12

## Scope

Validated nested manufacturing evidence detail quality labels, aggregate detail
accuracy metrics, and forward scorecard release gating.

## Commands

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile \
  scripts/eval_hybrid_dxf_manifest.py \
  src/core/benchmark/forward_scorecard.py \
  tests/unit/test_eval_hybrid_dxf_manifest.py \
  tests/unit/test_forward_scorecard.py \
  tests/unit/test_forward_scorecard_release_gate.py
```

```bash
.venv311/bin/flake8 \
  scripts/eval_hybrid_dxf_manifest.py \
  src/core/benchmark/forward_scorecard.py \
  tests/unit/test_eval_hybrid_dxf_manifest.py \
  tests/unit/test_forward_scorecard.py \
  tests/unit/test_forward_scorecard_release_gate.py
```

```bash
.venv311/bin/pytest -q \
  tests/unit/test_eval_hybrid_dxf_manifest.py \
  tests/unit/test_forward_scorecard.py \
  tests/unit/test_forward_scorecard_release_gate.py
```

```bash
git diff --check
```

## Results

- Python compile passed for touched exporter, scorecard, and tests.
- Flake8 passed for touched exporter, scorecard, and tests.
- Targeted pytest passed: `29 passed, 7 warnings in 2.40s`.
- `git diff --check` passed.
- Warnings are existing `ezdxf`/`pyparsing` dependency deprecation warnings.

## Verified Behavior

- Manifest payload expectations can include nested `details` objects.
- Manifest payload expectations can include explicit `details.*` fields.
- Convenience columns can label nested detail paths with
  `expected_<source>_detail_<path>` and `__` as the nested dot separator.
- Row output records detail-specific expected, matched, mismatched, and missing field
  counts.
- Summary output records detail-reviewed sample count and detail quality accuracy.
- Per-source payload quality records `detail_accuracy` and detail field counters.
- Numeric zero values in detail payloads are preserved as valid reviewed values.
- Forward scorecard `manufacturing_evidence` requires detail quality evidence for
  `release_ready`.
- Missing detail quality metrics downgrade an otherwise ready manufacturing evidence
  component to `benchmark_ready_with_gap`.
