# CAD ML Manufacturing Evidence Review Manifest Verification

Date: 2026-05-12

## Scope

Validated the manufacturing evidence review manifest builder, reviewed-label
threshold validator, and targeted CLI path.

## Commands

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile \
  scripts/build_manufacturing_review_manifest.py \
  tests/unit/test_build_manufacturing_review_manifest.py
```

```bash
.venv311/bin/flake8 \
  scripts/build_manufacturing_review_manifest.py \
  tests/unit/test_build_manufacturing_review_manifest.py
```

```bash
.venv311/bin/pytest -q \
  tests/unit/test_build_manufacturing_review_manifest.py
```

```bash
.venv311/bin/pytest -q \
  tests/unit/test_build_manufacturing_review_manifest.py \
  tests/unit/test_eval_hybrid_dxf_manifest.py
```

```bash
git diff --check -- \
  scripts/build_manufacturing_review_manifest.py \
  tests/unit/test_build_manufacturing_review_manifest.py
```

## Results

- Python compile passed for the new script and test.
- Flake8 passed for the new script and test.
- Targeted pytest passed: `3 passed, 7 warnings in 2.45s`.
- Compatibility pytest with the existing manufacturing manifest evaluator passed:
  `15 passed, 7 warnings in 2.20s`.
- `git diff --check` passed for the new script and test.
- Warnings are existing `ezdxf`/`pyparsing` dependency deprecation warnings imported
  through existing DXF evaluation helpers.

## Verified Behavior

- Benchmark result rows with `manufacturing_evidence` produce review suggestions.
- Suggested source labels are normalized to the existing manufacturing evidence source
  names.
- Suggested payload labels include top-level `kind`/`label`/`status`.
- Suggested payload labels include nested `details.*` paths.
- Numeric zero values are preserved as valid detail labels.
- Generated review manifests keep reviewed columns blank by default.
- Explicit bootstrap mode can prefill reviewed columns from suggestions.
- Reviewed manifest validation counts source-reviewed samples.
- Reviewed manifest validation counts payload-reviewed samples.
- Reviewed manifest validation counts nested detail-reviewed samples.
- `--fail-under-minimum` can convert insufficient reviewed labels into a non-zero
  process exit for CI/release gates.
