# Local Unit Regression Recovery Verification

Date: 2026-04-14
Environment: `.venv311` / Python 3.11

## Verification Commands

### Focused regressions

```bash
.venv311/bin/python -m pytest -q \
  tests/unit/test_filename_classifier.py \
  tests/unit/test_hybrid_classifier_rejection_policy.py
```

Result: passed

```bash
.venv311/bin/python -m pytest -q \
  tests/unit/test_graph2d_eval_helpers.py \
  tests/unit/test_eval_trend.py
```

Result: passed

```bash
.venv311/bin/python -m pytest -q \
  tests/unit/test_hybrid_calibration_make_targets.py \
  tests/unit/test_print_hybrid_blind_strict_real_gh_template.py
```

Result: `41 passed`

```bash
.venv311/bin/python -m pytest -q \
  tests/unit/test_makefile_targets.py \
  tests/unit/test_hybrid_calibration_make_targets.py \
  tests/unit/test_print_hybrid_blind_strict_real_gh_template.py
```

Result: `44 passed`

```bash
.venv311/bin/python -m pytest -q \
  tests/unit/assistant/test_semantic_retrieval.py \
  tests/unit/test_domain_embedding_provider.py
```

Result: `32 passed`

### Lint

```bash
make lint
```

Result: passed

### Full local unit regression gate

```bash
.venv311/bin/python -m pytest tests/unit -x -q --tb=short
```

Result:

```text
9718 passed, 116 skipped, 26 warnings in 128.54s (0:02:08)
```

## Key Assertions Confirmed

1. `HybridClassifier`
   - explicit filename-only fallback survives the Graph2D non-matching guardrail path
   - generic low-confidence rejection policy still works

2. `analyze.py`
   - Graph2D helper functions are importable again
   - helper recovery did not break the related gate-helper tests

3. `Makefile`
   - restored hybrid calibration / hybrid blind / soft-mode smoke / new-modules wrapper targets
   - `make -n` contract tests for those wrappers pass

4. `semantic_retrieval`
   - explicit `use_transformers=False` returns `SimpleEmbeddingProvider`
   - default factory path still selects `DomainEmbeddingProvider` when available

## Residual Warnings

- `ezdxf` / `pyparsing` deprecation warnings remain in the environment.
- Assistant function-calling tests still emit pre-existing coroutine resource warnings around `FeatureExtractor.extract`; they did not fail the suite in this run.

## Sidecar Tooling

- `Claude Code CLI` was available and callable during the session.
- It was used only for read-only sidecar review; verification results above come from local commands and local test execution.
